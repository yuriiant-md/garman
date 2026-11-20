"""A scripted model and a hand-moved clock for the agent tests: no API call, no real waiting."""

from collections.abc import Sequence

from anthropic.types import MessageParam, ToolParam
from pydantic import JsonValue

from src.agents.model import ModelTurn, ToolRequest


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


def tool_results(model: FakeModel, request: int) -> list[dict[str, object]]:
    """The tool_result blocks the model got in its request number `request` (0-based)."""
    messages = model.requests[request]["messages"]
    return messages[-1]["content"]  # type: ignore[index, no-any-return]
