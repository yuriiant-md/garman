"""Every tool call of the agent goes through run_guarded: faults on demand, retries, output schema.

Faults. FAULTS=timeout,429,503,schema (env) names one fault per attempt of every tool call, in
order; the attempts after the list are real. FAULTS=503 makes each call fail once and succeed on
the retry; FAULTS=503,503,503 uses up three attempts. An injected timeout, 429 or 503 replaces the
attempt, so the tool is not called; `schema` calls the tool and renames the first field of its
answer, the way a changed API would. A fault past the last attempt is never reached: with three
attempts, FAULTS=timeout,429,503,schema ends as "unavailable" on the 503, and a tool with a side
effect, tried once, meets only the first fault. Every tool's first field is required (pinned by
the tests), so a renamed one always fails the schema.

Retries. A transient error (timeout, 429, 5xx, a lost connection; injected or real) is tried
again after a pause that doubles (backoff_s, 2×, …), up to max_attempts, but only for a tool
without a side effect: a tool that changes something may have done it before failing, so it is
never repeated automatically. Any other error (a refusal, bad arguments, a bug) is not cured by
waiting and is returned at once. When the attempts run out, the model reads that the tool is
unavailable and must say so to the farmer instead of answering from memory. The same happens
when the next pause would end past the turn's deadline (graph.py waits tool_timeout_s for the
calls of a turn): a call the agent has stopped waiting for must not go on retrying.

Schema. Every answer is checked by the tool's output schema (tools.AgentTool.check_output); an
answer off its schema is an error in the trace, never a result with a field silently missing.
"""

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import httpx
from mcp.shared.exceptions import MCPError
from mcp.types import REQUEST_TIMEOUT
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.agents.tools import AgentTool

FAULTS_ENV = "FAULTS"
DRIFT_SUFFIX = "_v2"  # a renamed field, as a new version of an API would name it

RETRYABLE_STATUS = frozenset({408, 429})  # and every 5xx; not 409: a conflict is no busy server

type Sleep = Callable[[float], None]
type Clock = Callable[[], float]


class FaultKind(StrEnum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "429"
    UNAVAILABLE = "503"
    SCHEMA = "schema"


INJECTED_STATUS = {FaultKind.RATE_LIMIT: 429, FaultKind.UNAVAILABLE: 503}


class FaultPlan(BaseModel):
    """The faults of the attempts of every tool call, the first attempt first."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kinds: tuple[FaultKind, ...] = ()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "FaultPlan":
        """FAULTS=timeout,429 → (timeout, 429); an unknown fault stops the start."""
        names = [name.strip() for name in environ.get(FAULTS_ENV, "").split(",")]
        known = {kind.value: kind for kind in FaultKind}
        kinds = []
        for name in filter(None, names):
            if name not in known:
                raise ValueError(
                    f"{FAULTS_ENV}: unknown fault {name!r}; known: {', '.join(known)}"
                )
            kinds.append(known[name])
        return cls(kinds=tuple(kinds))

    def at(self, attempt: int) -> FaultKind | None:
        """The fault of attempt number `attempt` (from 1), or None for a real attempt."""
        return self.kinds[attempt - 1] if attempt <= len(self.kinds) else None


NO_FAULTS = FaultPlan()


class RetryPolicy(BaseModel):
    """configs/agent.yaml `retries`: how often a safe tool is tried and how long to wait."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(default=3, ge=1)
    backoff_s: float = Field(default=0.5, ge=0)  # before the 2nd attempt; doubles after

    def wait_s(self, attempt: int) -> float:
        """The pause after failed attempt number `attempt`."""
        return float(self.backoff_s * 2 ** (attempt - 1))

    @property
    def total_wait_s(self) -> float:
        """All the pauses of a call that uses every attempt."""
        return sum(self.wait_s(attempt) for attempt in range(1, self.max_attempts))


class InjectedStatus(Exception):
    """An HTTP status the fault plan put in place of the tool's answer."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code} (injected)")
        self.status_code = status_code


class InjectedTimeout(TimeoutError):
    """A timeout the fault plan put in place of the tool's answer."""


class ToolUnavailable(Exception):
    """A transient failure the agent will not or may not retry any more: words for the model."""


@dataclass(frozen=True)
class Outcome:
    """What a guarded call ended with: a result or an error, and the attempts it took."""

    result: object = None  # object: a broken tool may return anything but text
    error: BaseException | None = None
    attempts: int = 1


def run_guarded(
    tool: AgentTool,
    arguments: Mapping[str, JsonValue],
    faults: FaultPlan,
    policy: RetryPolicy,
    *,
    deadline_s: float | None = None,
    clock: Clock = time.monotonic,
    sleep: Sleep = time.sleep,
) -> Outcome:
    """Call the tool with the faults of the plan, retrying a safe tool on a transient error.

    deadline_s is on `clock`: no pause is started that would end past it.
    """
    for attempt in range(1, policy.max_attempts + 1):
        try:
            result = _attempt(tool, arguments, faults.at(attempt))
            if isinstance(result, str):  # anything else the graph reports as not text
                tool.check_output(result)
            return Outcome(result=result, attempts=attempt)
        # Exception, not BaseException: a cancelled MCP call or KeyboardInterrupt goes up and the
        # graph reports it as a failure of the guard itself
        except Exception as error:
            if not is_transient(error):
                return Outcome(error=error, attempts=attempt)
            if tool.side_effect:
                return Outcome(error=_not_retried(tool, error), attempts=attempt)
            wait_s = policy.wait_s(attempt)
            out_of_time = deadline_s is not None and clock() + wait_s >= deadline_s
            if attempt == policy.max_attempts or out_of_time:
                return Outcome(error=_unavailable(tool, error, attempt), attempts=attempt)
            sleep(wait_s)
    raise AssertionError("unreachable: the last attempt always returns")


def is_transient(error: BaseException) -> bool:
    """An error another attempt may not meet: a timeout, a busy or down service, a lost line."""
    if isinstance(error, TimeoutError | ConnectionError | InjectedStatus):
        return True
    if isinstance(error, ResponseHandlingException):  # also wraps a reply it could not parse
        return isinstance(error.source, httpx.TransportError | OSError)
    if isinstance(error, UnexpectedResponse):
        code = error.status_code
        return code is not None and (code in RETRYABLE_STATUS or code >= 500)
    if isinstance(error, MCPError):  # a closed connection is not: it is opened once per session
        return error.error.code == REQUEST_TIMEOUT
    return False


def drift(text: str) -> str:
    """The answer as a changed API would give it: its first field renamed."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = text
    if isinstance(data, dict) and data:
        first = next(iter(data))
        changed = {(f"{key}{DRIFT_SUFFIX}" if key == first else key): v for key, v in data.items()}
        return json.dumps(changed, ensure_ascii=False)
    return json.dumps({"drifted": data}, ensure_ascii=False)


def _attempt(
    tool: AgentTool, arguments: Mapping[str, JsonValue], fault: FaultKind | None
) -> object:
    match fault:
        case None:
            return tool.run(arguments)
        case FaultKind.TIMEOUT:
            raise InjectedTimeout("timeout (injected)")
        case FaultKind.RATE_LIMIT | FaultKind.UNAVAILABLE:
            raise InjectedStatus(INJECTED_STATUS[fault])
        case FaultKind.SCHEMA:
            return drift(tool.run(arguments))


def _cause(error: BaseException) -> str:
    if isinstance(error, InjectedStatus):
        return str(error.status_code)
    if isinstance(error, TimeoutError):
        return "timeout"
    return f"{type(error).__name__}: {error}"


def _unavailable(tool: AgentTool, error: BaseException, attempts: int) -> ToolUnavailable:
    return ToolUnavailable(
        f"{tool.name} недоступний ({_cause(error)} після {attempts} спроб). Скажи фермеру, що "
        "сервіс зараз недоступний; не відповідай на цю частину з пам'яті."
    )


def _not_retried(tool: AgentTool, error: BaseException) -> ToolUnavailable:
    return ToolUnavailable(
        f"{tool.name} не відповів ({_cause(error)}). Автоматично не повторю: він міг уже щось "
        "змінити. Скажи фермеру, що дія могла не виконатися, і не кажи, що її виконано."
    )
