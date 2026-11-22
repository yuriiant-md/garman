import json
from dataclasses import replace
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock

from src.agents.model import (
    AnthropicChatModel,
    MalformedReply,
    ModelTurn,
    ToolRequest,
    tool_results_message,
)
from src.agents.trace import ToolCall
from src.extraction.anthropic_extractor import AnthropicConfig

CONFIG = AnthropicConfig(
    model="claude-haiku-5-5", timeout_s=60, max_retries=3, max_tokens=8192, effort="medium"
)
SEARCH_TOOL = {
    "name": "search_corpus",
    "description": "Search the farmer's contract and the law.",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
}
QUESTION = [{"role": "user", "content": "Коли мені заплатять?"}]
REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

THINKING = ThinkingBlock(type="thinking", thinking="", signature="sig")
SAY = TextBlock(type="text", text="Шукаю строк оплати.")
SEARCH = ToolUseBlock(
    type="tool_use", id="toolu_1", name="search_corpus", input={"query": "строк оплати"}
)
type Block = ThinkingBlock | TextBlock | ToolUseBlock


def reply(*content: Block, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=list(content),
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=900, output_tokens=70),
    )


class FakeClient:
    """Answers every call with one outcome and keeps the requests."""

    def __init__(self, outcome: object) -> None:
        self.calls: list[dict[str, object]] = []
        self.messages = SimpleNamespace(create=self._create)
        self._outcome = outcome

    def _create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def call(outcome: object, config: AnthropicConfig = CONFIG) -> tuple[ModelTurn, FakeClient]:
    client = FakeClient(outcome)
    model = AnthropicChatModel(config, client=client)  # type: ignore[arg-type]
    turn = model.call(
        system="Ти помічник фермера.",
        messages=QUESTION,  # type: ignore[arg-type]
        tools=[SEARCH_TOOL],  # type: ignore[list-item]
    )
    return turn, client


# --- the request


def test_the_request_lets_the_model_choose_a_tool_and_sets_the_effort() -> None:
    _, client = call(reply(SAY))
    request = client.calls[0]

    assert request["model"] == "claude-haiku-5-5"
    assert request["max_tokens"] == 8192
    assert request["system"] == "Ти помічник фермера."
    assert request["messages"] == QUESTION
    assert request["tools"] == [SEARCH_TOOL]
    assert request["tool_choice"] == {"type": "auto"}  # the agent decides, nothing is forced
    assert request["output_config"] == {"effort": "medium"}


def test_without_an_effort_the_request_sends_no_output_config() -> None:
    _, client = call(reply(SAY), config=replace(CONFIG, effort=None))

    assert "output_config" not in client.calls[0]


# --- the turn


def test_a_tool_use_reply_becomes_tool_requests_with_their_arguments() -> None:
    turn, _ = call(reply(THINKING, SAY, SEARCH, stop_reason="tool_use"))

    assert turn.tool_requests == (
        ToolRequest(id="toolu_1", name="search_corpus", arguments={"query": "строк оплати"}),
    )
    assert turn.text == "Шукаю строк оплати."
    assert turn.finish == "tool_use"
    assert (turn.input_tokens, turn.output_tokens) == (900, 70)
    assert turn.latency_s >= 0


def test_a_text_reply_is_a_final_answer_with_no_tool_requests() -> None:
    turn, _ = call(reply(TextBlock(type="text", text="Через 5 днів [S1]."), THINKING))

    assert turn.tool_requests == ()
    assert turn.text == "Через 5 днів [S1]."
    assert turn.finish == "end_turn"


def test_the_assistant_message_keeps_every_block_for_the_next_request() -> None:
    turn, _ = call(reply(THINKING, SAY, SEARCH, stop_reason="tool_use"))

    # thinking blocks must go back unchanged, so the whole content is kept, not only the text
    assert turn.assistant_message == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Шукаю строк оплати."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "search_corpus",
                "input": {"query": "строк оплати"},
            },
        ],
    }


def test_the_assistant_message_is_plain_json_a_checkpoint_keeps_as_it_is() -> None:
    turn, _ = call(reply(THINKING, SAY, SEARCH, stop_reason="tool_use"))

    # graph.py checkpoints the messages; an SDK object there would be refused on load
    assert json.loads(json.dumps(turn.assistant_message)) == turn.assistant_message


def test_tool_input_that_is_not_a_json_object_is_a_malformed_reply() -> None:
    # the SDK's own type would refuse it: built unchecked, as a reply off its schema would come
    broken = ToolUseBlock.model_construct(
        type="tool_use", id="toolu_1", name="search_corpus", input="oops"
    )

    with pytest.raises(MalformedReply, match="toolu_1"):
        call(reply(broken, stop_reason="tool_use"))


def test_a_missing_stop_reason_is_named_unknown_not_none() -> None:
    turn, _ = call(SimpleNamespace(**{**vars(reply(SAY)), "stop_reason": None}))

    assert turn.finish == "unknown"


@pytest.mark.parametrize(
    "error",
    [
        anthropic.APITimeoutError(request=REQUEST),
        anthropic.RateLimitError(
            "slow down", response=httpx2.Response(429, request=REQUEST), body=None
        ),
    ],
)
def test_an_error_left_after_the_sdk_retries_reaches_the_caller(error: Exception) -> None:
    with pytest.raises(type(error)):
        call(error)


# --- tool results


def test_all_tool_results_go_back_in_one_user_message_in_order() -> None:
    found = ToolCall(id="toolu_1", name="search_corpus", arguments={}, result="S1", latency_s=1)
    failed = ToolCall(
        id="toolu_2", name="search_corpus", arguments={}, error="timeout after 20 s", latency_s=20
    )

    message = tool_results_message((found, failed))

    assert message == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "S1"},
            {
                "type": "tool_result",
                "tool_use_id": "toolu_2",
                "content": "timeout after 20 s",
                "is_error": True,
            },
        ],
    }
