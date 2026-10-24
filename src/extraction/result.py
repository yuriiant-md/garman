"""One extraction attempt and everything needed to reproduce and score it."""

import time
from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from src.extraction.prompts import Prompt
from src.extraction.sampling import Sampling
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


def make_result(
    model: str,
    prompt: Prompt,
    sampling: Sampling,
    params: Mapping[str, float],
    started: float,
    status: ExtractionStatus,
    **fields: object,
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
