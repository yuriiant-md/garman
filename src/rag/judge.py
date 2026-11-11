"""The stage 2 LLM judge: two checks of a saved answer, each one call to Claude Sonnet 5.5.

- correct: the answer against the reference answer a person wrote (prompts/judge_correct_v*.md);
- faithful: each claim against the documents it cites, exactly as the answerer read them
  (RagResult.cited_documents), with no reference in sight (prompts/judge_faithful_v*.md).

The judge is a model other than the one that answers (configs/judge.yaml), so it does not grade
its own habits. The verdict comes back as JSON shaped by a schema (output_config.format): Sonnet
5.5 rejects the forced tool call the answerer uses. A verdict Python cannot accept is sent back
once with the reason, and if the fix is not accepted either the judgement is `failed`. Errors left
after the SDK's retries become `error`; errors that mean the request itself is wrong stop the run.
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypedDict

import anthropic
import yaml
from anthropic.types import MessageParam, OutputConfigParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.extraction.anthropic_extractor import Effort
from src.extraction.prompts import Prompt
from src.extraction.retry import is_retryable
from src.rag.answerer import RagResult
from src.rag.context import BLOCK_SEPARATOR

JUDGE_CONFIG_PATH = Path("configs/judge.yaml")
NO_FIX_STOP_REASONS = frozenset({"refusal", "max_tokens"})  # asking again meets the same
# The answer and the question come from a model and a user: neither may close a block or open one
JUDGE_TAG = re.compile(r"<(?=\s*/?\s*(?:question|reference|answer|documents?|claims?)\b)", re.IGNORECASE)


class JudgeModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    max_tokens: int = Field(ge=1)
    effort: Effort
    price_input_per_mtok: float = Field(ge=0)
    price_output_per_mtok: float = Field(ge=0)


class JudgePrompts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    correct: Path
    faithful: Path


class JudgeThresholds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    correct_min: float = Field(ge=0, le=1)
    faithful_min: float = Field(ge=0, le=1)
    agreement_min: float = Field(ge=0, le=1)


class JudgeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: JudgeModel
    prompts: JudgePrompts
    thresholds: JudgeThresholds


def load_judge_config(path: Path = JUDGE_CONFIG_PATH) -> JudgeConfig:
    """The config, or a ValueError that names the file and what is wrong in it."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return JudgeConfig.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error


# The verdict schemas are sent to the API as they are: no length or range constraints, which
# structured outputs may not support; Python checks what the schema cannot (faithful_problem).
class CorrectVerdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conclusion_matches: bool
    citation_fits: bool
    numbers_ok: bool
    reason: str

    @property
    def correct(self) -> bool:
        return self.conclusion_matches and self.citation_fits and self.numbers_ok


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    supported: bool
    reason: str


class FaithfulVerdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verdicts: tuple[ClaimVerdict, ...]


class JudgeStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"  # the judge replied, but neither the verdict nor its one fix is valid
    ERROR = "error"  # no verdict: timeout, rate limit or server error left after retries


class Judgement(BaseModel):
    """One check of one answer and everything needed to reproduce it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: JudgeStatus
    model: str
    prompt_version: str
    prompt_sha256: str
    effort: Effort
    attempts: int
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float
    raw_output: str | None = None
    error: str | None = None


class CorrectJudgement(Judgement):
    verdict: CorrectVerdict | None = None


class FaithfulJudgement(Judgement):
    verdict: FaithfulVerdict | None = None


def judge_escaped(text: str) -> str:
    return JUDGE_TAG.sub("&lt;", text)


def correct_turn(question: str, reference: str, answer: str) -> str:
    blocks = {"question": question, "reference": reference, "answer": answer}
    return "\n\n".join(f"<{tag}>\n{judge_escaped(text)}\n</{tag}>" for tag, text in blocks.items())


def faithful_turn(result: RagResult) -> str:
    """The documents the answer cites, as its model read them, then its claims, numbered."""
    if result.answer is None or not result.answer.claims:
        raise ValueError("the answer has no claims to check")
    cited = {label for claim in result.answer.claims for label in claim.sources}
    if missing := sorted(cited - result.cited_documents.keys()):
        raise ValueError(
            f"cited_documents lacks {', '.join(missing)}: the run was saved before the answers "
            "kept their documents; make a new run"
        )
    documents = BLOCK_SEPARATOR.join(result.cited_documents.values())
    claims = "\n".join(
        f'<claim n="{n}" sources="{",".join(claim.sources)}">\n{judge_escaped(claim.text)}\n</claim>'
        for n, claim in enumerate(result.answer.claims, start=1)
    )
    return f"<documents>\n{documents}\n</documents>\n\n<claims>\n{claims}\n</claims>"


def faithful_problem(verdict: FaithfulVerdict, claims: int) -> str | None:
    numbers = [claim.n for claim in verdict.verdicts]
    if numbers == list(range(1, claims + 1)):
        return None
    return f"expected one verdict per claim, n = 1..{claims} in order; got n = {numbers}"


def repair_request(error: str) -> str:
    return f"The verdict cannot be accepted: {error}. Record the whole verdict again."


class _Logged(TypedDict):
    """What every judgement logs about how it was asked."""

    model: str
    prompt_version: str
    prompt_sha256: str
    effort: Effort
    latency_s: float


class _Usage(TypedDict):
    """What the replies came to and cost, the fix attempt included."""

    status: JudgeStatus
    attempts: int
    input_tokens: int
    output_tokens: int
    raw_output: str | None
    error: str | None


@dataclass(frozen=True)
class _Reply[V: BaseModel]:
    verdict: V | None
    usage: _Usage


def read_verdict[V: BaseModel](
    response: anthropic.types.Message, schema: type[V], problem: Callable[[V], str | None]
) -> tuple[str, V | None, str | None]:
    """Raw output, the checked verdict or None, and the reason it cannot be accepted."""
    raw = "".join(block.text for block in response.content if block.type == "text")
    if response.stop_reason in NO_FIX_STOP_REASONS:
        return raw, None, f"stop_reason={response.stop_reason}: the verdict is not complete"
    try:
        verdict = schema.model_validate_json(raw)
    except ValidationError as error:
        return raw, None, str(error)
    reason = problem(verdict)
    return raw, (None if reason else verdict), reason


class Judge:
    def __init__(
        self,
        config: JudgeModel,
        correct_prompt: Prompt,
        faithful_prompt: Prompt,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.config = config
        self.correct_prompt = correct_prompt
        self.faithful_prompt = faithful_prompt
        self._client = client or anthropic.Anthropic(
            timeout=config.timeout_s, max_retries=config.max_retries
        )

    def correct(self, question: str, reference: str, answer: str) -> CorrectJudgement:
        started = time.perf_counter()
        turn = correct_turn(question, reference, answer)
        reply = self._ask(self.correct_prompt, turn, CorrectVerdict, lambda _: None)
        return CorrectJudgement(
            **self._logged(self.correct_prompt, started), **reply.usage, verdict=reply.verdict
        )

    def faithful(self, result: RagResult) -> FaithfulJudgement:
        started = time.perf_counter()
        turn = faithful_turn(result)  # refuses an answer with no claims or no saved documents
        claims = len(result.answer.claims) if result.answer else 0
        reply = self._ask(
            self.faithful_prompt, turn, FaithfulVerdict, lambda v: faithful_problem(v, claims)
        )
        return FaithfulJudgement(
            **self._logged(self.faithful_prompt, started), **reply.usage, verdict=reply.verdict
        )

    def _logged(self, prompt: Prompt, started: float) -> _Logged:
        return {
            "model": self.config.id,
            "prompt_version": prompt.version,
            "prompt_sha256": prompt.sha256,
            "effort": self.config.effort,
            "latency_s": round(time.perf_counter() - started, 3),
        }

    def _ask[V: BaseModel](
        self, prompt: Prompt, turn: str, schema: type[V], problem: Callable[[V], str | None]
    ) -> _Reply[V]:
        try:
            responses = self._ask_with_one_fix(prompt, turn, schema, problem)
        except anthropic.APITimeoutError:
            error = f"timed out after {self.config.max_retries} retries"
        except anthropic.APIConnectionError as exc:
            error = f"connection failed after retries: {exc}"
        except anthropic.APIStatusError as exc:
            if not is_retryable(exc.status_code):
                raise
            error = f"API error {exc.status_code} after retries: {exc.message}"
        else:
            raw, verdict, problem_ = read_verdict(responses[-1], schema, problem)
            usage: _Usage = {
                "status": JudgeStatus.OK if verdict is not None else JudgeStatus.FAILED,
                "attempts": len(responses),
                "input_tokens": sum(r.usage.input_tokens for r in responses),
                "output_tokens": sum(r.usage.output_tokens for r in responses),
                "raw_output": raw,
                "error": problem_,
            }
            return _Reply(verdict, usage)
        no_reply: _Usage = {
            "status": JudgeStatus.ERROR,
            "attempts": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "raw_output": None,
            "error": error,
        }
        return _Reply(None, no_reply)

    def _ask_with_one_fix[V: BaseModel](
        self, prompt: Prompt, turn: str, schema: type[V], problem: Callable[[V], str | None]
    ) -> list[anthropic.types.Message]:
        """The replies in order: one, or two when the first was invalid and was sent back."""
        messages: list[MessageParam] = [{"role": "user", "content": turn}]
        first = self._create(prompt, messages, schema)
        raw, _, error = read_verdict(first, schema, problem)
        if error is None or first.stop_reason in NO_FIX_STOP_REASONS:
            return [first]
        fix: list[MessageParam] = [
            *messages,
            {"role": "assistant", "content": raw},
            {"role": "user", "content": repair_request(error)},
        ]
        return [first, self._create(prompt, fix, schema)]

    def _create(
        self, prompt: Prompt, messages: list[MessageParam], schema: type[BaseModel]
    ) -> anthropic.types.Message:
        output_config: OutputConfigParam = {
            "effort": self.config.effort,
            "format": {"type": "json_schema", "schema": schema.model_json_schema()},
        }
        return self._client.messages.create(
            model=self.config.id,
            max_tokens=self.config.max_tokens,
            system=prompt.text,
            messages=messages,
            output_config=output_config,
        )
