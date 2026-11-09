"""Answer a question from the documents: search, then Claude, then Python's checks.

An empty context (nothing over the reranker threshold) is «У документах цього немає» at once,
without paying for a model call. Otherwise Claude answers through a forced tool call shaped by
the Answer schema. The answer must pass the schema and cite only labels of the context; an
invalid one is sent back once with the reason, and if the fix is invalid too the result is
`failed`. Python renders the citations and the Commercial Code warnings, as in answer.py.

Retries and timeouts are the SDK's own, as in stage 1: what is left after them becomes an
`error` result, and errors that mean the request itself is wrong stop the run.
"""

import json
import time
from collections.abc import Mapping
from enum import StrEnum
from typing import TypedDict, Unpack

import anthropic
from anthropic.types import MessageParam, ToolParam
from pydantic import BaseModel, ConfigDict, ValidationError

from src.extraction.anthropic_extractor import (
    AnthropicConfig,
    OutputArgs,
    tool_error_turns,
)
from src.extraction.prompts import Prompt
from src.extraction.retry import is_retryable
from src.rag.answer import (
    NOT_IN_DOCUMENTS_TEXT,
    Answer,
    AnswerStatus,
    answer_problem,
    render,
)
from src.rag.chunking import Chunk
from src.rag.context import Context, ContextSettings, Search, build_context, escaped

ANSWER_TOOL: ToolParam = {
    "name": "record_answer",
    "description": (
        "Record the answer: claims, each citing the labels (S1, S2, …) of the documents it "
        "rests on; or status not_in_documents when the documents do not answer the question."
    ),
    "input_schema": Answer.model_json_schema(),
}
NO_FIX_STOP_REASONS = frozenset({"refusal", "max_tokens"})  # asking again meets the same
SAMPLING = "default"  # Haiku 5.5 rejects any temperature/top_p but the defaults


class Outcome(StrEnum):
    ANSWERED = "answered"
    NOT_IN_DOCUMENTS = "not_in_documents"  # nothing over the threshold, or the model said so
    FAILED = "failed"  # the model answered, but neither the answer nor its one fix is valid
    ERROR = "error"  # no answer: timeout, rate limit or server error left after retries


class RagResult(BaseModel):
    """One answered question and everything needed to reproduce and score it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    contract_id: str | None
    outcome: Outcome
    model_called: bool  # False: the threshold left nothing, the answer cost no tokens
    text: str | None = None  # what the farmer sees; None when failed or error
    answer: Answer | None = None
    sources: dict[str, str]  # label -> chunk_id: the whole context
    cited: tuple[str, ...] = ()  # chunk_ids the answer cites, each once
    context_tokens: int
    left_out: tuple[str, ...]
    below_threshold: tuple[str, ...]
    model: str
    prompt_version: str
    prompt_sha256: str
    sampling: str = SAMPLING
    attempts: int = 0  # 0 without a model call, 2 when the first answer was sent back
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    raw_output: str | None = None
    error: str | None = None


class Usage(TypedDict):
    """What every model reply costs, the fix attempt included."""

    attempts: int
    input_tokens: int
    output_tokens: int
    raw_output: str | None


class ModelFields(TypedDict, total=False):
    """What the model's reply fills in a result; the rest comes from the context."""

    text: str | None
    answer: Answer | None
    cited: tuple[str, ...]
    attempts: int
    input_tokens: int
    output_tokens: int
    raw_output: str | None
    error: str | None


def user_turn(question: str, context: Context) -> str:
    """The documents first, then the question, so the question is what the model reads last.

    The question is typed by a user, so it is escaped like the documents are.
    """
    return f"<documents>\n{context.text}\n</documents>\n\nПитання: {escaped(question)}"


def read_answer(
    response: anthropic.types.Message, sources: Mapping[str, Chunk]
) -> tuple[str, Answer | None, str | None]:
    """Raw output, the checked Answer or None, and the reason it cannot be accepted."""
    call = next(
        (b for b in response.content if b.type == "tool_use" and b.name == ANSWER_TOOL["name"]),
        None,
    )
    if call is None:
        raw = "".join(block.text for block in response.content if block.type == "text")
    else:
        raw = json.dumps(call.input, ensure_ascii=False)
    if response.stop_reason in NO_FIX_STOP_REASONS:
        return raw, None, f"stop_reason={response.stop_reason}: the answer is not complete"
    if call is None:
        return raw, None, f"no {ANSWER_TOOL['name']} call"
    try:
        answer = Answer.model_validate(call.input)
    except ValidationError as error:
        return raw, None, str(error)
    problem = answer_problem(answer, sources)
    return raw, (None if problem else answer), problem


def repair_request(error: str) -> str:
    """The message that asks the model to fix its one invalid answer."""
    return (
        "Your previous answer did not pass the check:\n"
        f"{error}\n"
        f"Call {ANSWER_TOOL['name']} again with the complete corrected answer to the same "
        "question, from the same documents."
    )


class Answerer:
    def __init__(
        self,
        search: Search,
        settings: ContextSettings,
        config: AnthropicConfig,
        prompt: Prompt,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.search = search
        self.settings = settings
        self.config = config
        self.prompt = prompt
        self._client = client or anthropic.Anthropic(
            timeout=config.timeout_s, max_retries=config.max_retries
        )

    def ask(self, question: str, contract_id: str | None = None) -> RagResult:
        if not question.strip():
            raise ValueError("the question is empty")
        started = time.perf_counter()
        context = build_context(self.search, question, contract_id, self.settings)
        if not context.sources:
            return self._result(
                question, contract_id, context, started, Outcome.NOT_IN_DOCUMENTS,
                text=NOT_IN_DOCUMENTS_TEXT,
            )
        try:
            responses = self._ask_with_one_fix(question, context)
        except anthropic.APITimeoutError:
            error = f"timed out after {self.config.max_retries} retries"
        except anthropic.APIConnectionError as exc:
            error = f"connection failed after retries: {exc}"
        except anthropic.APIStatusError as exc:
            if not is_retryable(exc.status_code):
                raise
            error = f"API error {exc.status_code} after retries: {exc.message}"
        else:
            return self._answered(question, contract_id, context, started, responses)
        return self._result(
            question, contract_id, context, started, Outcome.ERROR, attempts=1, error=error
        )

    def _create(self, messages: list[MessageParam]) -> anthropic.types.Message:
        effort: OutputArgs = (
            {"output_config": {"effort": self.config.effort}} if self.config.effort else {}
        )
        return self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.prompt.text,
            messages=messages,
            tools=[ANSWER_TOOL],
            tool_choice={"type": "tool", "name": ANSWER_TOOL["name"]},
            **effort,
        )

    def _ask_with_one_fix(
        self, question: str, context: Context
    ) -> list[anthropic.types.Message]:
        """The replies in order: one, or two when the first was invalid and was sent back."""
        messages: list[MessageParam] = [{"role": "user", "content": user_turn(question, context)}]
        first = self._create(messages)
        _, _, error = read_answer(first, context.sources)
        if error is None or first.stop_reason in NO_FIX_STOP_REASONS:
            return [first]
        fix = [*messages, *tool_error_turns(first, repair_request(error))]
        return [first, self._create(fix)]

    def _answered(
        self,
        question: str,
        contract_id: str | None,
        context: Context,
        started: float,
        responses: list[anthropic.types.Message],
    ) -> RagResult:
        raw, answer, error = read_answer(responses[-1], context.sources)
        usage: Usage = {
            "attempts": len(responses),
            "input_tokens": sum(r.usage.input_tokens for r in responses),
            "output_tokens": sum(r.usage.output_tokens for r in responses),
            "raw_output": raw,
        }
        if answer is None:
            return self._result(
                question, contract_id, context, started, Outcome.FAILED, error=error, **usage
            )
        outcome = (
            Outcome.ANSWERED if answer.status is AnswerStatus.ANSWERED else Outcome.NOT_IN_DOCUMENTS
        )
        return self._result(
            question, contract_id, context, started, outcome,
            text=render(answer, context.sources),
            answer=answer,
            cited=_cited(answer, context.sources),
            **usage,
        )

    def _result(
        self,
        question: str,
        contract_id: str | None,
        context: Context,
        started: float,
        outcome: Outcome,
        **fields: Unpack[ModelFields],
    ) -> RagResult:
        return RagResult(
            question=question,
            contract_id=contract_id,
            outcome=outcome,
            model_called=bool(context.sources),  # the model is asked only when something was found
            sources={label: chunk.chunk_id for label, chunk in context.sources.items()},
            context_tokens=context.tokens,
            left_out=context.left_out,
            below_threshold=context.below_threshold,
            model=self.config.model,
            prompt_version=self.prompt.version,
            prompt_sha256=self.prompt.sha256,
            latency_s=round(time.perf_counter() - started, 3),
            **fields,
        )


def _cited(answer: Answer, sources: Mapping[str, Chunk]) -> tuple[str, ...]:
    labels = (label for claim in answer.claims for label in claim.sources)
    return tuple(dict.fromkeys(sources[label].chunk_id for label in labels))
