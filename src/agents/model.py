"""The model the agent thinks with: a narrow interface, and Claude behind it.

The graph sees only `ChatModel.call` and the `ModelTurn` it returns, so the tests drive the graph
with a scripted fake and never pay for a call. `AnthropicChatModel` is the real one.

Unlike the extractor of stage 1, the model is not forced to call a tool (`tool_choice: auto`):
deciding when to search, when to calculate and when to answer is the agent's job.

Retries and timeouts of one call are the SDK's own (models.yaml: timeout_s, max_retries). An error
left after them is not caught here: the graph stops the run with `error` and keeps the trace.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

import anthropic
from anthropic.types import (
    ContentBlockParam,
    MessageParam,
    ToolChoiceAutoParam,
    ToolParam,
    ToolResultBlockParam,
)
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from src.agents.trace import ToolCall
from src.extraction.anthropic_extractor import AnthropicConfig, OutputArgs
from src.extraction.retry import is_retryable


class MalformedReply(Exception):
    """The model's reply cannot be acted on, e.g. a tool input that is not a JSON object."""


def is_transient(error: BaseException) -> bool:
    """A model failure the farmer can be told about: the API busy or out of reach, a broken reply.

    Any other one means the request itself is wrong (a bad key, a bad request): a bug to raise,
    as in stages 1 and 2. The agent (graph.py) and the summary (memory.py) share this rule.
    """
    if isinstance(error, anthropic.APIStatusError):
        return is_retryable(error.status_code)
    return isinstance(error, anthropic.APIConnectionError | MalformedReply)  # timeouts included


class ToolRequest(BaseModel):
    """One tool call the model asked for, before it runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    arguments: dict[str, JsonValue]


@dataclass(frozen=True)
class ModelTurn:
    """One model call as the graph needs it."""

    text: str  # the text blocks joined; the final answer when there are no tool requests
    tool_requests: tuple[ToolRequest, ...]
    input_tokens: int
    output_tokens: int
    latency_s: float
    finish: str  # the API's stop_reason: end_turn, tool_use, max_tokens, refusal, ...
    # The reply as it goes back in the next request. Claude Haiku 5.5 always thinks, and its
    # thinking blocks must be sent back unchanged, so this is the whole content, not the text.
    assistant_message: MessageParam


class ToolingArgs(TypedDict, total=False):
    """tools and tool_choice go into the request only with a tool: the API refuses one alone."""

    tools: Sequence[ToolParam]
    tool_choice: ToolChoiceAutoParam


class ChatModel(Protocol):
    def call(
        self, *, system: str, messages: Sequence[MessageParam], tools: Sequence[ToolParam]
    ) -> ModelTurn: ...


class AnthropicChatModel:
    def __init__(self, config: AnthropicConfig, client: anthropic.Anthropic | None = None) -> None:
        self.config = config
        self._client = client or anthropic.Anthropic(
            timeout=config.timeout_s, max_retries=config.max_retries
        )

    def call(
        self, *, system: str, messages: Sequence[MessageParam], tools: Sequence[ToolParam]
    ) -> ModelTurn:
        effort: OutputArgs = (
            {"output_config": {"effort": self.config.effort}} if self.config.effort else {}
        )
        # with no tools (the dialogue's summary) the call only writes text
        tooling: ToolingArgs = {}
        if tools:
            tooling = {"tools": tools, "tool_choice": {"type": "auto"}}
        started = time.perf_counter()
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=messages,
            **tooling,
            **effort,
        )
        return ModelTurn(
            text="".join(block.text for block in response.content if block.type == "text"),
            tool_requests=tuple(
                _tool_request(block) for block in response.content if block.type == "tool_use"
            ),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=time.perf_counter() - started,
            finish=response.stop_reason or "unknown",
            # every block as plain JSON, unchanged: thinking must go back with its signature, and
            # the graph's checkpoint keeps JSON as it is but refuses an SDK object it does not know
            assistant_message={
                "role": "assistant",
                # the same fields as the SDK's *BlockParam: a response block dumped as a request one
                "content": cast(
                    list[ContentBlockParam],
                    [
                        block.model_dump(mode="json", exclude_none=True)
                        for block in response.content
                    ],
                ),
            },
        )


def _tool_request(block: anthropic.types.ToolUseBlock) -> ToolRequest:
    """The SDK types a tool input as any object: it is validated as JSON here, at the boundary."""
    try:
        return ToolRequest.model_validate(
            {"id": block.id, "name": block.name, "arguments": block.input}
        )
    except ValidationError as error:
        raise MalformedReply(f"tool call {block.id}: input is not a JSON object") from error


def tool_results_message(calls: Sequence[ToolCall]) -> MessageParam:
    """Every result of one model turn in one user message, in the order the model asked.

    One message, not one per call: split results teach the model to stop calling tools in
    parallel. A failed call goes back as is_error, so the model sees what went wrong.
    """
    return {"role": "user", "content": [_tool_result(call) for call in calls]}


def _tool_result(call: ToolCall) -> ToolResultBlockParam:
    if call.error is not None:
        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": call.error,
            "is_error": True,
        }
    return {"type": "tool_result", "tool_use_id": call.id, "content": call.result or ""}
