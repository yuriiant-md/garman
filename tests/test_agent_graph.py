import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import anthropic
import httpx2
import pytest
from anthropic.types import MessageParam, ToolParam
from pydantic import JsonValue

from src.agents.graph import Agent, AgentTool
from src.agents.limits import Limits, StopReason
from src.agents.model import MalformedReply, ModelTurn, ToolRequest
from src.agents.trace import Trace
from src.rag.documents import read_jsonl

LIMITS = Limits(
    max_steps=5, max_tokens=10_000, max_cost_usd=1.0, tool_timeout_s=1, request_timeout_s=60
)
PRICES = (1.0, 5.0)  # USD per 1M tokens
REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


class Clock:
    """A clock the test moves by hand, so the request timeout needs no real waiting."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeModel:
    """Plays a script: one outcome per call, the last one repeating; keeps every request."""

    def __init__(self, *script: ModelTurn | Exception) -> None:
        self.script = list(script)
        self.requests: list[dict[str, object]] = []

    def call(
        self, *, system: str, messages: Sequence[MessageParam], tools: Sequence[ToolParam]
    ) -> ModelTurn:
        self.requests.append({"system": system, "messages": list(messages), "tools": list(tools)})
        outcome = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def turn(
    text: str = "",
    *requests: ToolRequest,
    tokens: tuple[int, int] = (100, 20),
    finish: str | None = None,
) -> ModelTurn:
    return ModelTurn(
        text=text,
        tool_requests=requests,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        latency_s=0.1,
        finish=finish or ("tool_use" if requests else "end_turn"),
        assistant_message={"role": "assistant", "content": text or "…"},
    )


def ask(name: str = "search_corpus", call_id: str = "t1", **arguments: JsonValue) -> ToolRequest:
    return ToolRequest(id=call_id, name=name, arguments=arguments)


def tool(name: str, run: object) -> AgentTool:
    spec: ToolParam = {"name": name, "description": name, "input_schema": {"type": "object"}}
    return AgentTool(spec=spec, run=run)  # type: ignore[arg-type]


def search(arguments: Mapping[str, JsonValue]) -> str:
    return f"знайдено: {arguments.get('query')}"


SEARCH = tool("search_corpus", search)


def make_agent(
    model: FakeModel,
    tools: tuple[AgentTool, ...] = (SEARCH,),
    limits: Limits = LIMITS,
    clock: Clock | None = None,
) -> Agent:
    return Agent(
        model=model,
        tools=tools,
        system="Ти помічник фермера.",
        limits=limits,
        prices=PRICES,
        model_name="haiku-5.5",
        clock=clock or Clock(),
    )


def tool_results(model: FakeModel, request: int) -> list[dict[str, object]]:
    """The tool_result blocks the model got in its request number `request` (0-based)."""
    messages = model.requests[request]["messages"]
    return messages[-1]["content"]  # type: ignore[index, no-any-return]


# --- answering


def test_a_direct_answer_ends_the_run_as_answered_with_one_step_and_its_cost() -> None:
    model = FakeModel(turn("Через 5 днів [S1]."))

    trace = make_agent(model).run("Коли мені заплатять?")

    assert trace.stop_reason == StopReason.ANSWERED
    assert trace.answer == "Через 5 днів [S1]."
    assert not trace.partial
    assert len(trace.steps) == 1
    assert trace.usage.steps == 1
    assert trace.usage.cost_usd == pytest.approx((100 * 1 + 20 * 5) / 1_000_000)
    assert trace.steps[0].cost_usd == pytest.approx(trace.usage.cost_usd)


def test_the_model_gets_the_system_prompt_the_question_and_the_tool_specs() -> None:
    model = FakeModel(turn("ok"))

    make_agent(model).run("Коли мені заплатять?")

    request = model.requests[0]
    assert request["system"] == "Ти помічник фермера."
    assert request["messages"] == [{"role": "user", "content": "Коли мені заплатять?"}]
    assert request["tools"] == [SEARCH.spec]


def test_a_tool_result_goes_back_to_the_model_and_lands_in_the_trace() -> None:
    model = FakeModel(turn("Шукаю.", ask(query="строк оплати")), turn("Через 5 днів [S1]."))

    trace = make_agent(model).run("Коли мені заплатять?")

    assert trace.stop_reason == StopReason.ANSWERED
    assert tool_results(model, 1) == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "знайдено: строк оплати"}
    ]
    first = trace.steps[0]
    assert first.text == "Шукаю."
    assert [(c.name, c.arguments, c.result) for c in first.tool_calls] == [
        ("search_corpus", {"query": "строк оплати"}, "знайдено: строк оплати")
    ]
    assert trace.steps[1].tool_calls == ()


def test_several_tool_calls_of_one_turn_all_run_and_go_back_in_one_message() -> None:
    model = FakeModel(
        turn("", ask(call_id="a", query="ціна"), ask(call_id="b", query="штраф")),
        turn("готово"),
    )

    make_agent(model).run("Порівняй")

    assert [block["tool_use_id"] for block in tool_results(model, 1)] == ["a", "b"]


# --- limits


def test_the_step_limit_stops_a_model_that_never_answers() -> None:
    model = FakeModel(turn("", ask(query="ще раз")))

    trace = make_agent(model, limits=LIMITS.model_copy(update={"max_steps": 3})).run("?")

    assert trace.stop_reason == StopReason.MAX_STEPS
    assert trace.partial
    assert trace.usage.steps == 3
    assert len(model.requests) == 3  # no fourth call


def test_check_all_documents_stops_on_the_token_budget_with_a_partial_answer() -> None:
    contracts = [f"own_{number:03d}" for number in range(1, 11)]
    model = FakeModel(
        *(turn("", ask(query=contract), tokens=(300, 50)) for contract in contracts)
    )
    limits = LIMITS.model_copy(update={"max_tokens": 1000, "max_steps": 20})

    trace = make_agent(model, limits=limits).run("Перевір усі мої договори")

    assert trace.stop_reason == StopReason.BUDGET
    assert trace.usage.tokens == 1050  # 350 per call: stopped before the fourth call
    assert len(model.requests) == 3
    assert trace.answer is not None
    assert "бюджет" in trace.answer
    for checked in contracts[:3]:
        assert f"знайдено: {checked}" in trace.answer
    assert "own_004" not in trace.answer


def test_the_dollar_budget_stops_the_run() -> None:
    model = FakeModel(turn("", ask(query="x"), tokens=(1000, 1000)))  # 0.006 USD per call
    limits = LIMITS.model_copy(update={"max_cost_usd": 0.01})

    trace = make_agent(model, limits=limits).run("?")

    assert trace.stop_reason == StopReason.BUDGET
    assert trace.usage.steps == 2


def test_the_request_timeout_stops_the_run_before_the_next_model_call() -> None:
    clock = Clock()

    def slow_search(arguments: Mapping[str, JsonValue]) -> str:
        clock.now += 61  # the request has run past its 60 s
        return "знайдено"

    model = FakeModel(turn("", ask(query="x")), turn("never"))
    agent = make_agent(model, tools=(tool("search_corpus", slow_search),), clock=clock)

    trace = agent.run("?")

    assert trace.stop_reason == StopReason.TIMEOUT
    assert len(model.requests) == 1
    assert "задовго" in (trace.answer or "")


# --- tool failures go back to the model, the run goes on


def test_a_tool_over_its_timeout_becomes_an_error_result_and_the_run_goes_on() -> None:
    release = threading.Event()

    def stuck(arguments: Mapping[str, JsonValue]) -> str:
        release.wait(timeout=5)
        return "too late"

    model = FakeModel(turn("", ask(query="x")), turn("Не вдалося знайти."))
    limits = LIMITS.model_copy(update={"tool_timeout_s": 0.05})
    try:
        trace = make_agent(model, tools=(tool("search_corpus", stuck),), limits=limits).run("?")
    finally:
        release.set()

    assert trace.stop_reason == StopReason.ANSWERED
    call = trace.steps[0].tool_calls[0]
    assert (call.result, call.error) == (None, "timeout after 0.05 s")
    assert tool_results(model, 1)[0]["is_error"] is True


def test_an_unknown_tool_becomes_an_error_result() -> None:
    model = FakeModel(turn("", ask(name="send_email", to="x")), turn("Не можу."))

    trace = make_agent(model).run("?")

    assert trace.steps[0].tool_calls[0].error == "unknown tool: send_email"
    assert trace.stop_reason == StopReason.ANSWERED


def test_a_tool_that_returns_no_text_becomes_an_error_result() -> None:
    model = FakeModel(turn("", ask(query="x")), turn("Не вийшло."))

    trace = make_agent(model, tools=(tool("search_corpus", lambda arguments: None),)).run("?")

    assert trace.steps[0].tool_calls[0].error == "tool returned NoneType, not text"
    assert trace.stop_reason == StopReason.ANSWERED


def test_two_calls_with_the_same_id_both_run() -> None:
    model = FakeModel(turn("", ask(query="ціна"), ask(query="штраф")), turn("готово"))

    trace = make_agent(model).run("?")

    assert [call.result for call in trace.steps[0].tool_calls] == [
        "знайдено: ціна",
        "знайдено: штраф",
    ]


def test_each_tool_call_reports_its_own_latency() -> None:
    def slow(arguments: Mapping[str, JsonValue]) -> str:
        time.sleep(0.1)
        return "повільно"

    model = FakeModel(
        turn("", ask(call_id="a", query="x"), ask(name="slow", call_id="b")), turn("готово")
    )

    trace = make_agent(model, tools=(SEARCH, tool("slow", slow))).run("?")

    fast_call, slow_call = trace.steps[0].tool_calls
    assert fast_call.latency_s < 0.05 <= 0.1 <= slow_call.latency_s


def test_a_tool_that_raises_becomes_an_error_result_with_the_exception() -> None:
    def broken(arguments: Mapping[str, JsonValue]) -> str:
        raise ValueError("no such contract")

    model = FakeModel(turn("", ask(query="x")), turn("Не знайшов договір."))

    trace = make_agent(model, tools=(tool("search_corpus", broken),)).run("?")

    assert trace.steps[0].tool_calls[0].error == "ValueError: no such contract"
    assert trace.stop_reason == StopReason.ANSWERED


# --- model failures stop the run


@pytest.mark.parametrize(
    "error, says",
    [
        (
            anthropic.InternalServerError(
                "overloaded", response=httpx2.Response(529, request=REQUEST), body=None
            ),
            "overloaded",
        ),
        (anthropic.APITimeoutError(request=REQUEST), "APITimeoutError"),
        (anthropic.APIConnectionError(request=REQUEST), "APIConnectionError"),
        (MalformedReply("tool input is not JSON"), "tool input is not JSON"),
    ],
)
def test_a_model_failure_left_after_the_sdk_retries_stops_the_run_as_error(
    error: Exception, says: str
) -> None:
    model = FakeModel(turn("", ask(query="x")), error)

    trace = make_agent(model).run("?")

    assert trace.stop_reason == StopReason.ERROR
    assert trace.error is not None and says in trace.error
    assert trace.usage.steps == 1  # the failed call cost nothing
    assert "знайдено: x" in (trace.answer or "")


def test_an_error_that_means_the_request_is_wrong_is_raised() -> None:
    model = FakeModel(
        anthropic.AuthenticationError(
            "bad key", response=httpx2.Response(401, request=REQUEST), body=None
        )
    )

    with pytest.raises(anthropic.AuthenticationError):
        make_agent(model).run("?")


@pytest.mark.parametrize(
    "finish", ["refusal", "max_tokens", "model_context_window_exceeded", "pause_turn"]
)
def test_a_reply_that_is_neither_an_answer_nor_tool_use_stops_the_run_as_error(
    finish: str,
) -> None:
    model = FakeModel(turn("", ask(query="x"), finish=finish))

    trace = make_agent(model).run("?")

    assert trace.stop_reason == StopReason.ERROR
    assert trace.error == f"model stopped: {finish}"
    assert trace.steps[0].tool_calls == ()  # a cut reply's tool calls are not run


# --- the trace file


def test_a_run_appends_its_trace_to_the_file(tmp_path: Path) -> None:
    path = tmp_path / "traces" / "run.jsonl"
    agent = make_agent(FakeModel(turn("ok")))

    first = agent.run("перше", trace_file=path)
    second = agent.run("друге", trace_file=path)

    assert read_jsonl(path, Trace) == (first, second)
    assert first.request_id != second.request_id
