"""Where the agent must stop: configs/agent.yaml and the check before each model call.

The graph calls `stop_reason` before every model call. The first limit reached stops the run:
the agent writes the reason to the trace and returns what it has so far, it does not go on.
The check runs before the call because the cost of a call is known only after it, so a run can
pass its token or dollar budget by at most one model call.

Money is counted here in Python from the prices in configs/models.yaml, never by the model.
"""

from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.agents.faults import RetryPolicy
from src.extraction.metrics import cost_usd as tokens_cost_usd
from src.extraction.run import MODELS

AGENT_CONFIG_PATH = Path("configs/agent.yaml")


class StopReason(StrEnum):
    """Why a run ended; every trace ends with one."""

    ANSWERED = "answered"  # the model gave its final answer
    MAX_STEPS = "max_steps"
    BUDGET = "budget"  # tokens or dollars
    TIMEOUT = "timeout"  # the whole request took too long
    ERROR = "error"  # the model call failed after the SDK's retries


class Limits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_steps: int = Field(gt=0)  # model calls per request
    # input + output summed over every call of the request; each call's input repeats the history,
    # so this is the tokens paid for, not the size of the context
    max_tokens: int = Field(gt=0)
    max_cost_usd: float = Field(gt=0)
    tool_timeout_s: float = Field(gt=0)  # the tool calls of one turn run in parallel within it
    request_timeout_s: float = Field(gt=0)  # the whole request


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str  # a section of configs/models.yaml
    limits: Limits
    retries: RetryPolicy = RetryPolicy()

    @model_validator(mode="after")
    def _retries_fit_the_tool_timeout(self) -> Self:
        # the pauses alone must leave time for the calls: run_guarded starts no pause that would
        # end past the turn's tool_timeout_s, so with longer pauses a tool would never be retried
        waits = self.retries.total_wait_s
        if waits >= self.limits.tool_timeout_s:
            raise ValueError(
                f"retries waits {waits:g} s, not less than tool_timeout_s "
                f"{self.limits.tool_timeout_s:g} s: the turn would abandon the call mid-retry"
            )
        return self


def load_agent_config(path: Path = AGENT_CONFIG_PATH) -> AgentConfig:
    """The config, or a ValueError that names the file and what is wrong in it."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        config = AgentConfig.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error
    if config.model not in MODELS:
        raise ValueError(f"{path}: model {config.model!r} is not a section of configs/models.yaml")
    return config


class Usage(BaseModel):
    """What a run has spent so far. Immutable: each model call makes a new Usage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: int = 0  # model calls made
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add_call(
        self, *, input_tokens: int, output_tokens: int, prices: tuple[float, float]
    ) -> "Usage":
        """This usage plus one model call; prices are USD per 1M input and output tokens."""
        return Usage(
            steps=self.steps + 1,
            input_tokens=self.input_tokens + input_tokens,
            output_tokens=self.output_tokens + output_tokens,
            cost_usd=self.cost_usd + tokens_cost_usd(input_tokens, output_tokens, *prices),
        )


def stop_reason(usage: Usage, limits: Limits, *, elapsed_s: float) -> StopReason | None:
    """The limit the run has reached, or None to go on.

    When several are reached at once, timeout wins, then budget, then steps: a run that hung
    is the first thing to see in a trace, and money matters more than the number of steps.
    """
    if elapsed_s >= limits.request_timeout_s:
        return StopReason.TIMEOUT
    if usage.tokens >= limits.max_tokens or usage.cost_usd >= limits.max_cost_usd:
        return StopReason.BUDGET
    if usage.steps >= limits.max_steps:
        return StopReason.MAX_STEPS
    return None
