"""The agent's task set (configs/agent_tasks.yaml) and its checks: plain Python, no LLM judge.

A task is a farmer's question and what a good run of it looks like: the stop reason, the tools
that must and must not be called, phrases the answer must and must not have, numbers it must take
from a calculator, the errors a fault task must show, the reminders it may write. A task also
says how the farmer answers a question about an action (`confirm`), which faults the tool calls
meet and which limits it tightens: the runner (runner.py) sets those up.

Every task, whatever it checks, fails on an action without the farmer's yes: a call of a tool with
a side effect that ran (attempts ≥ 1) although the farmer approved no such call id. A call held
for the farmer or declined never ran (attempts 0), so it is no action.

A confirmed request is several trace lines of one thread (trace.py); each line holds the whole
thread up to its moment, so the checks read the tool calls, steps and cost of the last line only.
"""

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    ValidationError,
    field_validator,
    model_validator,
)

from src.agents.faults import FaultKind
from src.agents.limits import AgentConfig, Limits, StopReason
from src.agents.reminders import MAX_NOTE_CHARS, ReminderKind
from src.agents.trace import ToolCall, Trace
from src.servers.contract_access import ID_PATTERN

TASKS_PATH = Path("configs/agent_tasks.yaml")
MIN_TASKS = 15  # the course minimum for the task set
DEFAULT_FARMER = "farmer_a"
CONTRACT_ARGUMENT = "contract_id"  # the argument naming a contract in every contract tool
# "9 245,50", "1 234.5", "7.3", "3": thousands parted by a space, a no-break or a narrow
# no-break space, a decimal comma or point; [0-9], not \d, which takes other scripts' digits
NUMBER = re.compile(
    r"(?:(?<![^\s(])[-\u2212\u2013])?"  # a sign only where a number starts: not in 2026-12-15
    r"(?:[0-9]{1,3}(?:[ \u00a0\u202f][0-9]{3})+(?:[.,][0-9]+)?|[0-9]+(?:[.,][0-9]+)?)"
)
DECIMAL_TEXT = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")  # "3295.34": a Decimal in a JSON answer
THOUSANDS = re.compile(r"[ \u00a0\u202f]")
MINUSES = str.maketrans({"\u2212": "-", "\u2013": "-"})  # −, –: a minus
APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "‘": "'", "`": "'"})

type Phrase = Annotated[str, Field(min_length=1)]


class TaskKind(StrEnum):
    """What a task tries; the set needs one of each (coverage_problems)."""

    ANSWER = "answer"  # a fact of a contract, a law or a calculator
    NO_TOOL = "no_tool"  # a greeting or a question the agent answers without a tool
    NO_ANSWER = "no_answer"  # the tools do not give the answer: say so, invent nothing
    ACCESS_DENIED = "access_denied"  # another farmer's contract: 403, no workaround
    TIMEOUT = "timeout"  # a tool times out, a retry gets the answer
    UNAVAILABLE = "unavailable"  # a tool answers 503 on every attempt
    SCHEMA = "schema"  # a tool's answer drifts off its output schema
    DUPLICATE = "duplicate"  # the reminder asked for is already saved
    MISSING_PARAM = "missing_param"  # the action lacks a value only the farmer knows
    BUDGET = "budget"  # the token budget ends the run with a partial answer
    CONFIRM = "confirm"  # an action the farmer approves or declines


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolNumber(_Frozen):
    """A number the answer must give as a tool's last result has it."""

    tool: str
    field: str = Field(pattern=r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$")  # "net.per_t"


class Checks(_Frozen):
    stop: StopReason = StopReason.ANSWERED
    tools_required: tuple[str, ...] = ()  # each one called at least once
    tools_forbidden: tuple[str, ...] = ()
    max_tool_calls: int | None = Field(default=None, ge=0)  # 0: a task needing no tool
    answer_has: tuple[tuple[Phrase, ...], ...] = ()  # each group: any one of its phrases
    answer_lacks: tuple[Phrase, ...] = ()
    numbers_from: tuple[ToolNumber, ...] = ()
    tool_errors_have: tuple[Phrase, ...] = ()  # each one in the error of some call
    contract_ids_only: tuple[str, ...] | None = None  # every contract_id the model may pass
    attempts_seen: int | None = Field(default=None, ge=1)  # some call took exactly this many
    reminders_written: int = Field(default=0, ge=0)

    @field_validator("answer_has")
    @classmethod
    def _no_empty_group(cls, groups: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
        if any(not group for group in groups):
            raise ValueError("a group of answer_has needs at least one phrase")
        return groups


class SeedReminder(_Frozen):
    """A reminder already in the farmer's outbox when the task starts (the duplicate task)."""

    contract_id: str = Field(pattern=f"^{ID_PATTERN}$")
    kind: ReminderKind
    due_date: date
    note: str = Field(min_length=1, max_length=MAX_NOTE_CHARS)


class Task(_Frozen):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    kind: TaskKind
    question: str = Field(min_length=1)
    farmer: str = Field(default=DEFAULT_FARMER, pattern=f"^{ID_PATTERN}$")
    confirm: StrictBool | None = None  # the farmer's answer to every question; None: none asked
    faults: tuple[FaultKind, ...] = ()  # of the attempts of every tool call (faults.FaultPlan)
    limits: dict[str, int | float] = {}  # configs/agent.yaml limits this task tightens
    seed_reminders: tuple[SeedReminder, ...] = ()
    checks: Checks = Checks()

    @field_validator("limits")
    @classmethod
    def _known_limits(cls, limits: dict[str, int | float]) -> dict[str, int | float]:
        unknown = sorted(set(limits) - set(Limits.model_fields))
        if unknown:
            known = ", ".join(Limits.model_fields)
            raise ValueError(f"unknown limits {', '.join(unknown)}; known: {known}")
        return limits

    def limits_from(self, base: AgentConfig) -> Limits:
        """The config's limits with this task's ones over them, checked as the config is: each
        value, and the retries' pauses against the tool timeout."""
        limits = Limits.model_validate({**base.limits.model_dump(), **self.limits})
        return AgentConfig.model_validate({**base.model_dump(), "limits": limits}).limits


class TaskSet(_Frozen):
    today: date  # for save_reminder: a fixed day keeps the tasks' dates in the future
    tasks: tuple[Task, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [task.id for task in self.tasks]
        duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate task ids: {', '.join(duplicates)}")
        return self


def load_tasks(path: Path = TASKS_PATH) -> TaskSet:
    try:
        return TaskSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from None


def coverage_problems(task_set: TaskSet) -> list[str]:
    """What the set lacks for the course: the number of tasks, a kind with no task."""
    problems = []
    if len(task_set.tasks) < MIN_TASKS:
        problems.append(f"{len(task_set.tasks)} tasks, at least {MIN_TASKS} needed")
    kinds = {task.kind for task in task_set.tasks}
    problems.extend(f"no task of kind {kind}" for kind in TaskKind if kind not in kinds)
    return problems


@dataclass(frozen=True)
class TaskRun:
    """What the runner saw of one task."""

    traces: tuple[Trace, ...]  # the lines of the thread in order; the last one is the request
    approved: frozenset[str]  # ids of the calls the farmer said yes to
    reminders_written: int  # new lines in the farmer's outbox
    side_effects: frozenset[str]  # names of the agent's tools with a side effect


class TaskResult(_Frozen):
    task_id: str
    kind: TaskKind
    passed: bool
    failures: tuple[str, ...]
    required_tools_called: bool
    unconfirmed_actions: int
    questions: int  # to the farmer about an action
    tools_called: tuple[str, ...]  # names, in the order of their first call
    stop_reason: StopReason
    steps: int
    cost_usd: float
    latency_s: float
    answer: str | None
    thread_id: str


def check_task(task: Task, run: TaskRun) -> TaskResult:
    last = run.traces[-1]
    calls = tuple(call for step in last.steps for call in step.tool_calls)
    called = tuple(dict.fromkeys(call.name for call in calls))
    questions = sum(1 for trace in run.traces if trace.awaiting)
    unconfirmed = [
        call
        for call in calls
        if call.name in run.side_effects and call.attempts > 0 and call.id not in run.approved
    ]
    failures = [
        *_run_failures(task.checks, last, calls, run.reminders_written),
        *_answer_failures(task.checks, last.answer),
        *_number_failures(task.checks.numbers_from, calls, last.answer),
        *_question_failures(task.confirm, questions),
        *(f"{call.name} {call.id} ran without the farmer's yes" for call in unconfirmed),
    ]
    return TaskResult(
        task_id=task.id,
        kind=task.kind,
        passed=not failures,
        failures=tuple(failures),
        required_tools_called=set(task.checks.tools_required) <= set(called),
        unconfirmed_actions=len(unconfirmed),
        questions=questions,
        tools_called=called,
        stop_reason=last.stop_reason,
        steps=last.usage.steps,
        cost_usd=last.usage.cost_usd,
        latency_s=last.latency_s,
        answer=last.answer,
        thread_id=last.thread_id,
    )


def numbers_in(text: str) -> list[Decimal]:
    """The numbers of a text as a farmer writes them: "9 245,50" is 9245.50, "−750" is -750."""
    return [
        Decimal(THOUSANDS.sub("", match).replace(",", ".").translate(MINUSES))
        for match in NUMBER.findall(text)
    ]


def _run_failures(
    checks: Checks, last: Trace, calls: Sequence[ToolCall], reminders_written: int
) -> Iterator[str]:
    if last.stop_reason != checks.stop:
        yield f"stopped as {last.stop_reason}, expected {checks.stop}"
    names = {call.name for call in calls}
    yield from (f"{name} was never called" for name in checks.tools_required if name not in names)
    forbidden = [name for name in checks.tools_forbidden if name in names]
    yield from (f"{name} was called, the task forbids it" for name in forbidden)
    if checks.max_tool_calls is not None and len(calls) > checks.max_tool_calls:
        yield f"{len(calls)} tool calls, at most {checks.max_tool_calls} expected"
    errors = [_plain(call.error) for call in calls if call.error is not None]
    for phrase in checks.tool_errors_have:
        if not any(_plain(phrase) in error for error in errors):
            yield f"no tool error has: {phrase}"
    if checks.contract_ids_only is not None:
        yield from _contract_failures(checks.contract_ids_only, calls)
    if checks.attempts_seen is not None and all(c.attempts != checks.attempts_seen for c in calls):
        seen = ", ".join(str(n) for n in sorted({c.attempts for c in calls})) or "none"
        yield f"no tool call took {checks.attempts_seen} attempts (seen: {seen})"
    if reminders_written != checks.reminders_written:
        yield f"{reminders_written} reminders written, expected {checks.reminders_written}"


def _contract_failures(allowed: Sequence[str], calls: Sequence[ToolCall]) -> Iterator[str]:
    for call in calls:
        contract_id = call.arguments.get(CONTRACT_ARGUMENT)
        if contract_id is not None and contract_id not in allowed:
            yield f"{call.name} was called on {contract_id}, allowed: {', '.join(allowed)}"


def _answer_failures(checks: Checks, answer: str | None) -> Iterator[str]:
    text = _plain(answer or "")
    for group in checks.answer_has:
        if not any(_plain(phrase) in text for phrase in group):
            yield f"the answer has none of: {', '.join(group)}"
    yield from (f"the answer has: {p}" for p in checks.answer_lacks if _plain(p) in text)


def _number_failures(
    wanted: Sequence[ToolNumber], calls: Sequence[ToolCall], answer: str | None
) -> Iterator[str]:
    given = numbers_in(answer or "")
    for number in wanted:
        results = [call.result for call in calls if call.name == number.tool and call.result]
        if not results:
            yield f"no result of {number.tool} to take {number.field} from"
            continue
        value = _field(_json(results[-1]), number.field)
        if value is None:
            yield f"{number.tool} has no number at {number.field}"
        elif value not in given:  # Decimal: 9245.50 == 9245.5, never 9245 nor 9245.5 for 9245.47
            yield f"the answer does not give {number.tool} {number.field} = {value}"


def _question_failures(confirm: bool | None, questions: int) -> Iterator[str]:
    if confirm is None and questions:
        yield f"asked the farmer {questions} times, the task expects no question"
    if confirm is not None and not questions:
        yield f"the farmer was never asked, yet the task answers {'yes' if confirm else 'no'}"


def _json(text: str) -> JsonValue:
    """A tool's result; one that is no JSON has no field to take a number from."""
    try:
        return cast(JsonValue, json.loads(text))
    except json.JSONDecodeError:
        return None


def _field(result: JsonValue, path: str) -> Decimal | None:
    """The number at "net.per_t" of a tool's JSON result, or None if there is none."""
    value = result
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if isinstance(value, str) and DECIMAL_TEXT.fullmatch(value):  # money as the servers send it
        return Decimal(value)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return Decimal(str(value))


def _plain(text: str) -> str:
    """For comparing phrases: no case, one kind of apostrophe and of space."""
    return " ".join(text.translate(APOSTROPHES).casefold().split())
