"""The agent: a LangGraph loop of model calls and tool calls under limits.

    START → check ──(a limit reached)──────────────────→ finish → END
              ↑   └→ model ──(answer, refusal, error)──→ finish
              │            └→ tools ──┐
              └───────────────────────┘

`check` runs before every model call and stops the run on the first limit reached (limits.py).
The step limit is our own, not LangGraph's recursion_limit: that one raises and loses the state,
while a stop here keeps it and turns it into a partial answer. recursion_limit stays as a guard
against a bug in this graph only.

A tool's failure (invalid arguments, timeout, unknown tool, exception, no text) goes back to the
model as an error result and the run goes on: the model may fix the call or try another way.
A model's failure ends the run: an API error left after the SDK's retries, a malformed reply,
or any stop but end_turn and tool_use (refusal, max_tokens, a full context window, ...).
An API error that means the request itself is wrong (bad key, bad request) is raised, as in
stages 1 and 2.

The tool calls of one turn run in parallel and share one wait of tool_timeout_s. Python cannot
kill a thread, so a call over the timeout is abandoned, not stopped: the agent stops waiting and
goes on, the thread finishes in the background. That is safe for read-only tools; a tool with a
side effect needs an idempotency key (issue #20). A thread that never finishes also keeps the
process from exiting, since Python joins worker threads at exit: real tools must have their own
timeouts (HTTP, Qdrant) shorter than tool_timeout_s.
"""

import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import uuid4

import anthropic
from anthropic.types import MessageParam
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import JsonValue

from src.agents.limits import Limits, StopReason, Usage, stop_reason
from src.agents.model import (
    ChatModel,
    MalformedReply,
    ModelTurn,
    ToolRequest,
    tool_results_message,
)
from src.agents.tools import InvalidArguments, Tool
from src.agents.trace import ToolCall, Trace, TraceStep, append_trace
from src.extraction.retry import is_retryable

USABLE_FINISHES = frozenset({"end_turn", "tool_use"})  # any other stop cannot be acted on
NODES_PER_STEP = 3  # check, model, tools
RECURSION_MARGIN = 5  # the first check and finish, with room to spare
PARTIAL_RESULT_CHARS = 500  # of each tool result quoted in a partial answer
PARTIAL_HEADERS = {
    StopReason.MAX_STEPS: "Зупинився: запит потребує забагато кроків.",
    StopReason.BUDGET: "Зупинився: вичерпано бюджет запиту.",
    StopReason.TIMEOUT: "Зупинився: запит триває задовго.",
    StopReason.ERROR: "Зупинився: модель зараз недоступна.",
}

type ToolRun = Callable[[Mapping[str, JsonValue]], str]
type Update = dict[str, object]


class AgentState(TypedDict):
    started_s: float  # the agent's clock at the start of the request
    messages: list[MessageParam]
    steps: tuple[TraceStep, ...]
    usage: Usage
    pending: tuple[ToolRequest, ...]  # the tool calls of the last model turn, not yet run
    stop_reason: StopReason | None
    answer: str | None
    error: str | None


@dataclass(frozen=True)
class Agent:
    model: ChatModel
    tools: tuple[Tool[Any], ...]  # Any: each tool has its own arguments model
    system: str
    limits: Limits
    prices: tuple[float, float]  # USD per 1M input and output tokens, from models.yaml
    model_name: str  # the models.yaml section, for the trace
    clock: Callable[[], float] = field(default=time.monotonic)

    def __post_init__(self) -> None:
        # the API refuses two tools with one name; here the last one would silently win
        names = [tool.name for tool in self.tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate tool names: {', '.join(duplicates)}")

    def run(self, question: str, trace_file: Path | None = None) -> Trace:
        """Answer one question; append the trace to `trace_file` when one is given."""
        started_at = datetime.now().astimezone()
        recursion_limit = self.limits.max_steps * NODES_PER_STEP + RECURSION_MARGIN
        state = self._graph().invoke(
            self._initial_state(question), {"recursion_limit": recursion_limit}
        )
        trace = Trace(
            request_id=uuid4().hex,
            started_at=started_at,
            model=self.model_name,
            question=question,
            steps=state["steps"],
            usage=state["usage"],
            stop_reason=state["stop_reason"],
            answer=state["answer"],
            error=state["error"],
            latency_s=self.clock() - state["started_s"],
        )
        if trace_file is not None:
            append_trace(trace, trace_file)
        return trace

    def _initial_state(self, question: str) -> AgentState:
        return {
            "started_s": self.clock(),
            "messages": [{"role": "user", "content": question}],
            "steps": (),
            "usage": Usage(),
            "pending": (),
            "stop_reason": None,
            "answer": None,
            "error": None,
        }

    def _graph(self) -> CompiledStateGraph:
        builder = StateGraph(AgentState)
        builder.add_node("check", self._check)
        builder.add_node("model", self._call_model)
        builder.add_node("tools", self._run_tools)
        builder.add_node("finish", self._finish)
        builder.add_edge(START, "check")
        builder.add_conditional_edges("check", _after_check, ["model", "finish"])
        builder.add_conditional_edges("model", _after_model, ["tools", "finish"])
        builder.add_edge("tools", "check")
        builder.add_edge("finish", END)
        return builder.compile()

    # --- nodes: each returns only the keys it changes, as new objects

    def _check(self, state: AgentState) -> Update:
        elapsed_s = self.clock() - state["started_s"]
        reason = stop_reason(state["usage"], self.limits, elapsed_s=elapsed_s)
        return {} if reason is None else {"stop_reason": reason}

    def _call_model(self, state: AgentState) -> Update:
        try:
            turn = self.model.call(
                system=self.system,
                messages=state["messages"],
                tools=[tool.spec for tool in self.tools],
            )
        except (anthropic.APITimeoutError, anthropic.APIConnectionError, MalformedReply) as error:
            return {"stop_reason": StopReason.ERROR, "error": f"{type(error).__name__}: {error}"}
        except anthropic.APIStatusError as error:
            if not is_retryable(error.status_code):
                raise  # the request itself is wrong: a bug to fix, not a farmer's problem
            return {"stop_reason": StopReason.ERROR, "error": f"{type(error).__name__}: {error}"}
        return self._after_turn(state, turn)

    def _after_turn(self, state: AgentState, turn: ModelTurn) -> Update:
        usage = state["usage"].add_call(
            input_tokens=turn.input_tokens, output_tokens=turn.output_tokens, prices=self.prices
        )
        step = TraceStep(
            step=usage.steps,
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
            cost_usd=usage.cost_usd - state["usage"].cost_usd,
            latency_s=turn.latency_s,
            text=turn.text,
        )
        update: Update = {
            "usage": usage,
            "steps": (*state["steps"], step),
            "messages": [*state["messages"], turn.assistant_message],
        }
        if turn.finish not in USABLE_FINISHES:
            error = f"model stopped: {turn.finish}"
            return {**update, "stop_reason": StopReason.ERROR, "error": error}
        if not turn.tool_requests:
            return {**update, "stop_reason": StopReason.ANSWERED, "answer": turn.text}
        return {**update, "pending": turn.tool_requests}

    def _run_tools(self, state: AgentState) -> Update:
        calls = self._execute(state["pending"])
        last = state["steps"][-1].model_copy(update={"tool_calls": calls})
        return {
            "steps": (*state["steps"][:-1], last),
            "messages": [*state["messages"], tool_results_message(calls)],
            "pending": (),
        }

    def _finish(self, state: AgentState) -> Update:
        reason = state["stop_reason"]
        if reason is None or reason == StopReason.ANSWERED:
            return {}
        return {"answer": partial_answer(reason, state["steps"])}

    # --- tools

    def _execute(self, requests: tuple[ToolRequest, ...]) -> tuple[ToolCall, ...]:
        """Run the calls of one turn in parallel; wait for them at most tool_timeout_s."""
        by_name = {tool.name: tool for tool in self.tools}
        tools = [by_name.get(request.name) for request in requests]
        executor = ThreadPoolExecutor(max_workers=len(requests))
        started = time.perf_counter()
        try:
            # a list in the order of the requests, not a dict by id: two calls never merge
            futures = [
                None if tool is None else executor.submit(_timed, tool.run, request.arguments)
                for request, tool in zip(requests, tools, strict=True)
            ]
            wait([f for f in futures if f is not None], timeout=self.limits.tool_timeout_s)
            waited = time.perf_counter() - started
            return tuple(
                self._tool_call(request, tool, future, waited)
                for request, tool, future in zip(requests, tools, futures, strict=True)
            )
        finally:
            # not `with`: leaving a with-block waits for every thread, a stuck one included
            executor.shutdown(wait=False, cancel_futures=True)

    def _tool_call(
        self,
        request: ToolRequest,
        tool: Tool[Any] | None,
        future: Future[tuple[object, float]] | None,
        waited: float,
    ) -> ToolCall:
        def done(latency_s: float = waited, **outcome: str) -> ToolCall:
            return ToolCall(
                id=request.id,
                name=request.name,
                tool_version=None if tool is None else tool.version,
                arguments=request.arguments,
                latency_s=latency_s,
                **outcome,
            )

        if future is None:
            return done(latency_s=0, error=f"unknown tool: {request.name}")
        if not future.done():
            return done(error=f"timeout after {self.limits.tool_timeout_s:g} s")
        error = future.exception()
        if isinstance(error, InvalidArguments):  # already names the tool and every bad field
            return done(error=str(error))
        if error is not None:  # the tool's failure goes back to the model, the run goes on
            return done(error=f"{type(error).__name__}: {error}")
        result, latency_s = future.result()
        if not isinstance(result, str):
            return done(latency_s, error=f"tool returned {type(result).__name__}, not text")
        return done(latency_s, result=result)


def _timed(run: ToolRun, arguments: Mapping[str, JsonValue]) -> tuple[object, float]:
    """A tool's result and its own run time; object, not str: a tool may break its type."""
    started = time.perf_counter()
    result: object = run(arguments)
    return result, time.perf_counter() - started


def _after_check(state: AgentState) -> Literal["model", "finish"]:
    return "finish" if state["stop_reason"] is not None else "model"


def _after_model(state: AgentState) -> Literal["tools", "finish"]:
    return "finish" if state["stop_reason"] is not None else "tools"


def partial_answer(reason: StopReason, steps: tuple[TraceStep, ...]) -> str:
    """What the farmer gets when a limit stops the run: why, and what the tools found so far.

    Built in Python, not by one more model call: the run stopped because it may not spend more.
    """
    found = [
        f"- {call.name}({_arguments(call.arguments)}): {call.result[:PARTIAL_RESULT_CHARS]}"
        for step in steps
        for call in step.tool_calls
        if call.result is not None
    ]
    if not found:
        return f"{PARTIAL_HEADERS[reason]}\n\nНічого не встиг знайти."
    return f"{PARTIAL_HEADERS[reason]}\n\nЩо встиг знайти:\n" + "\n".join(found)


def _arguments(arguments: Mapping[str, JsonValue]) -> str:
    return ", ".join(f"{name}={value}" for name, value in arguments.items())
