"""One extraction attempt and everything needed to reproduce and score it."""

import time
from enum import StrEnum
from typing import Any, NotRequired, TypedDict, Unpack

from pydantic import BaseModel, ConfigDict

from src.extraction.prompts import Prompt
from src.extraction.sampling import Sampling, SamplingParams
from src.schemas.deal import Deal


class ExtractionStatus(StrEnum):
    OK = "ok"  # the response passed the Deal schema and its content checks
    FAILED = "failed"  # the model answered, but neither the answer nor its one fix is a valid Deal
    ERROR = "error"  # no answer: timeout, rate limit or server error left after retries


class ExtractionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    prompt_version: str
    prompt_sha256: str
    sampling: str
    temperature: float | None  # values actually sent to the model, not the config ones
    top_p: float | None
    status: ExtractionStatus
    deal: Deal | None = None
    raw_output: str | None = None
    error: str | None = None
    attempts: int = 1  # 2 when the first answer was invalid and the model was asked to fix it
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0


class AnswerFields(TypedDict, total=False):
    """What an extractor knows about the answer itself; the rest of a result make_result fills."""

    deal: Deal | None
    raw_output: str | None
    error: str | None
    attempts: int
    input_tokens: int
    output_tokens: int


class Record(TypedDict):
    """One line of a results *.jsonl file as metrics and run summaries read it: an
    ExtractionResult in JSON form with its document and the start of its run."""

    doc_id: str
    run_at: str
    model: str
    prompt_version: str
    sampling: str
    status: str
    deal: dict[str, Any] | None
    attempts: NotRequired[int]  # missing in results saved before the fix attempt existed
    latency_s: float
    input_tokens: int
    output_tokens: int


def make_result(
    model: str,
    prompt: Prompt,
    sampling: Sampling,
    params: SamplingParams,
    started: float,
    status: ExtractionStatus,
    **fields: Unpack[AnswerFields],
) -> ExtractionResult:
    """The result every extractor logs: `params` are the sampling values it sent, and `started`
    is the time.perf_counter() reading taken before its first request."""
    return ExtractionResult(
        model=model,
        prompt_version=prompt.version,
        prompt_sha256=prompt.sha256,
        sampling=sampling.name,
        temperature=params.get("temperature"),
        top_p=params.get("top_p"),
        status=status,
        latency_s=round(time.perf_counter() - started, 3),
        **fields,
    )
