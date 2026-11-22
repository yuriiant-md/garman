"""The trace of one agent request: every model call, every tool call, what it cost, why it ended.

A launch writes one JSONL file, results/stage3/traces/<start>.jsonl, with one Trace per request,
so a run of the task set is one file and each line can be read on its own. The file is appended
to after each request: if a later request crashes, the traces before it are already on disk.

A request that waits for the farmer's confirmation (a tool with a side effect, graph.py) is one
thread of several lines: one line awaiting_confirmation per question to the farmer, then the line
of its end. Each line holds the whole thread up to its moment, so the last one is the request:
count the steps and the cost of that line only, never the sum of the thread's lines.
"""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from src.agents.faults import FaultKind
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
    # attempts the call took (faults.py retries a safe tool); 0 when none finished: an unknown
    # tool, or a call abandoned at the turn's timeout
    attempts: int = Field(default=0, ge=0)
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


class PendingAction(BaseModel):
    """A call with a side effect the farmer is asked about; it runs only after a yes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str  # the model's id of the call
    tool: str
    tool_version: str
    arguments: dict[str, JsonValue]  # checked: the call would go through as it is


class Trace(BaseModel):
    """One request from the question to the stop, or to a question to the farmer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str  # of this line
    thread_id: str  # of the request: the same on every line of a confirmed request
    started_at: datetime
    model: str  # a section of configs/models.yaml
    prompt_version: str  # of prompts/agent_v*.md, from its header
    question: str
    steps: tuple[TraceStep, ...]
    usage: Usage
    stop_reason: StopReason
    answer: str | None  # on a stop by a limit: what the agent had so far, or None
    error: str | None = None  # what went wrong when stop_reason is error
    faults: tuple[FaultKind, ...] = ()  # injected by FAULTS for this run (faults.py)
    awaiting: tuple[PendingAction, ...] = ()  # when stopped for a confirmation: what is asked
    paused_s: float = Field(default=0, ge=0)  # the farmer deciding; not in latency_s nor limits
    latency_s: float = Field(ge=0)  # the whole request so far, the pauses left out

    @model_validator(mode="after")
    def _awaiting_only_when_stopped_for_it(self) -> "Trace":
        if bool(self.awaiting) != (self.stop_reason == StopReason.AWAITING_CONFIRMATION):
            raise ValueError("awaiting is set exactly when the run stopped for a confirmation")
        return self

    @property
    def partial(self) -> bool:
        """True when a limit or an error ended the run before the model's final answer."""
        return self.stop_reason not in (StopReason.ANSWERED, StopReason.AWAITING_CONFIRMATION)


def trace_path(started: datetime, traces_dir: Path = TRACES_DIR) -> Path:
    """The file of one launch, named after its start like the runs of stages 1 and 2."""
    return traces_dir / f"{started.strftime(RUN_STAMP_FORMAT)}.jsonl"


def append_trace(trace: Trace, path: Path) -> None:
    """Add one line to the launch's file, creating the file and its folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(f"{trace.model_dump_json()}\n")
