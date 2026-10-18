"""One extraction attempt and everything needed to reproduce and score it."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from src.schemas.deal import Deal


class ExtractionStatus(StrEnum):
    OK = "ok"  # the response passed the Deal schema and its content checks
    INVALID = "invalid"  # the model answered, but the answer is not a valid Deal
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
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
