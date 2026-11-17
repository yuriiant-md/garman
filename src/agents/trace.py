"""The trace of one agent request: every model call, every tool call, what it cost, why it ended.

A launch writes one JSONL file, results/stage3/traces/<start>.jsonl, with one Trace per request,
so a run of the task set is one file and each line can be read on its own. The file is appended
to after each request: if a later request crashes, the traces before it are already on disk.
"""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from src.agents.limits import StopReason, Usage
from src.extraction.run import RUN_STAMP_FORMAT

TRACES_DIR = Path("results/stage3/traces")


class ToolCall(BaseModel):
    """One tool call the model asked for, and what came back."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str  # the model's id of the call: ties the result to the request in the messages
    name: str
    tool_version: str | None = None  # of tools/<name>.yaml; None for a tool the agent lacks
    arguments: dict[str, JsonValue]
    result: str | None = None
    error: str | None = None  # timeout, unknown tool, invalid arguments or the tool's exception
    latency_s: float = Field(ge=0)

    @model_validator(mode="after")
    def _result_or_error(self) -> "ToolCall":
        if (self.result is None) == (self.error is None):
            raise ValueError("a tool call has exactly one of result and error")
        return self


class TraceStep(BaseModel):
    """One model call and the tool calls it asked for."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    latency_s: float = Field(ge=0)  # the model call only; each tool call has its own
    text: str = ""  # what the model wrote in this step, if anything
    tool_calls: tuple[ToolCall, ...] = ()


class Trace(BaseModel):
    """One request from the question to the stop."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    started_at: datetime
    model: str  # a section of configs/models.yaml
    question: str
    steps: tuple[TraceStep, ...]
    usage: Usage
    stop_reason: StopReason
    answer: str | None  # on a stop by a limit: what the agent had so far, or None
    error: str | None = None  # what went wrong when stop_reason is error
    latency_s: float = Field(ge=0)  # the whole request

    @property
    def partial(self) -> bool:
        """True when a limit or an error ended the run before the model's final answer."""
        return self.stop_reason != StopReason.ANSWERED


def trace_path(started: datetime, traces_dir: Path = TRACES_DIR) -> Path:
    """The file of one launch, named after its start like the runs of stages 1 and 2."""
    return traces_dir / f"{started.strftime(RUN_STAMP_FORMAT)}.jsonl"


def append_trace(trace: Trace, path: Path) -> None:
    """Add one line to the launch's file, creating the file and its folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(f"{trace.model_dump_json()}\n")
