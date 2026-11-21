import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import httpx
import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import CONNECTION_CLOSED, REQUEST_TIMEOUT
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.agents.faults import (
    FAULTS_ENV,
    NO_FAULTS,
    FaultKind,
    FaultPlan,
    InjectedStatus,
    RetryPolicy,
    ToolUnavailable,
    drift,
    is_transient,
    run_guarded,
)
from src.agents.graph import Agent
from src.agents.limits import AgentConfig, Limits, StopReason
from src.agents.tools import InvalidArguments, SchemaMismatch, Tool, ToolArgs, ToolDescription
from tests.agent_fixtures import Clock, FakeModel, ask, tool_results, turn

LIMITS = Limits(
    max_steps=5, max_tokens=10_000, max_cost_usd=1.0, tool_timeout_s=5, request_timeout_s=60
)
POLICY = RetryPolicy(max_attempts=3, backoff_s=0.5)


class QueryArgs(ToolArgs):
    query: str


class Found(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hits: list[str]


@dataclass
class Counting:
    """A tool's function that answers or raises by a script, and counts its calls."""

    script: list[Exception | str] = field(default_factory=list)
    calls: int = 0

    def __call__(self, args: QueryArgs) -> str:
        self.calls += 1
        outcome = self.script.pop(0) if self.script else Found(hits=[args.query]).model_dump_json()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_tool(func: Callable[[QueryArgs], str], *, side_effect: bool = False) -> Tool[QueryArgs]:
    description = ToolDescription(
        name="search_corpus", version="v1", side_effect=side_effect, description="test"
    )
    return Tool(description=description, args=QueryArgs, func=func, output=Found)


@dataclass
class Sleeps:
    waits: list[float] = field(default_factory=list)

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def guarded(
    tool: Tool[QueryArgs],
    faults: FaultPlan = NO_FAULTS,
    arguments: Mapping[str, JsonValue] | None = None,
    sleep: Sleeps | None = None,
) -> tuple[object, BaseException | None, int]:
    outcome = run_guarded(
        tool, arguments or {"query": "пеня"}, faults, POLICY, sleep=sleep or Sleeps()
    )
    return outcome.result, outcome.error, outcome.attempts


# --- the plan of faults


@pytest.mark.parametrize(
    ("value", "kinds"),
    [
        ("", ()),
        ("503", (FaultKind.UNAVAILABLE,)),
        ("timeout, 429,503 ,schema", tuple(FaultKind)),
        ("429,,429", (FaultKind.RATE_LIMIT, FaultKind.RATE_LIMIT)),
    ],
)
def test_the_env_flag_lists_one_fault_per_attempt_in_order(
    value: str, kinds: tuple[FaultKind, ...]
) -> None:
    assert FaultPlan.from_env({FAULTS_ENV: value}).kinds == kinds


def test_no_env_flag_means_no_faults() -> None:
    assert FaultPlan.from_env({}).kinds == ()


def test_an_unknown_fault_stops_the_start_and_names_the_known_ones() -> None:
    with pytest.raises(ValueError, match="unknown fault '500'.*timeout, 429, 503, schema"):
        FaultPlan.from_env({FAULTS_ENV: "429,500"})


def test_the_plan_gives_the_fault_of_each_attempt_then_none() -> None:
    plan = FaultPlan(kinds=(FaultKind.TIMEOUT, FaultKind.RATE_LIMIT))

    assert [plan.at(attempt) for attempt in (1, 2, 3)] == [
        FaultKind.TIMEOUT,
        FaultKind.RATE_LIMIT,
        None,
    ]


# --- retries of a safe tool


def test_without_faults_a_tool_answers_on_the_first_attempt() -> None:
    func = Counting()

    result, error, attempts = guarded(make_tool(func))

    assert (error, attempts, func.calls) == (None, 1, 1)
    assert Found.model_validate_json(str(result)).hits == ["пеня"]


@pytest.mark.parametrize("kind", ["timeout", "429", "503"])
def test_a_transient_fault_is_retried_and_the_real_answer_comes_next(kind: str) -> None:
    func = Counting()
    sleeps = Sleeps()

    result, error, attempts = guarded(
        make_tool(func), FaultPlan.from_env({FAULTS_ENV: kind}), sleep=sleeps
    )

    assert (error, attempts) == (None, 2)
    assert func.calls == 1  # the faulty attempt never reached the tool
    assert sleeps.waits == [0.5]
    assert Found.model_validate_json(str(result)).hits == ["пеня"]


def test_the_backoff_doubles_between_attempts() -> None:
    sleeps = Sleeps()
    plan = FaultPlan(kinds=(FaultKind.TIMEOUT, FaultKind.RATE_LIMIT))

    _, error, attempts = guarded(make_tool(Counting()), plan, sleep=sleeps)

    assert (error, attempts) == (None, 3)
    assert sleeps.waits == [0.5, 1.0]


def test_after_the_last_attempt_the_tool_is_unavailable_with_words_for_the_model() -> None:
    func = Counting()
    sleeps = Sleeps()
    plan = FaultPlan.from_env({FAULTS_ENV: "timeout,429,503,schema"})

    result, error, attempts = guarded(make_tool(func), plan, sleep=sleeps)

    assert result is None
    assert isinstance(error, ToolUnavailable)
    assert attempts == 3
    assert func.calls == 0
    assert sleeps.waits == [0.5, 1.0]  # no wait after the last attempt
    text = str(error)
    assert "search_corpus" in text and "503" in text and "3" in text
    assert "недоступн" in text and "не відповідай" in text


def test_no_pause_is_started_past_the_turns_deadline() -> None:
    func = Counting()
    sleeps = Sleeps()
    plan = FaultPlan.from_env({FAULTS_ENV: "503"})

    outcome = run_guarded(
        make_tool(func), {"query": "пеня"}, plan, POLICY,
        deadline_s=10.0, clock=lambda: 9.6, sleep=sleeps,
    )  # fmt: skip

    assert isinstance(outcome.error, ToolUnavailable)
    assert outcome.attempts == 1
    assert sleeps.waits == []  # 9.6 + 0.5 s would end past 10.0
    assert func.calls == 0


def test_a_pause_that_ends_before_the_deadline_is_taken() -> None:
    sleeps = Sleeps()
    plan = FaultPlan.from_env({FAULTS_ENV: "503"})

    outcome = run_guarded(
        make_tool(Counting()), {"query": "пеня"}, plan, POLICY,
        deadline_s=10.0, clock=lambda: 9.4, sleep=sleeps,
    )  # fmt: skip

    assert (outcome.error, outcome.attempts, sleeps.waits) == (None, 2, [0.5])


def test_a_schema_fault_after_a_transient_one_is_met_on_the_retry() -> None:
    plan = FaultPlan.from_env({FAULTS_ENV: "503,schema"})

    _, error, attempts = guarded(make_tool(Counting()), plan)

    assert isinstance(error, SchemaMismatch)
    assert attempts == 2


def test_a_side_effect_tool_meets_only_the_first_fault() -> None:
    plan = FaultPlan.from_env({FAULTS_ENV: "schema,503"})

    _, error, attempts = guarded(make_tool(Counting(), side_effect=True), plan)

    assert isinstance(error, SchemaMismatch)
    assert attempts == 1


def test_a_real_transient_error_of_the_tool_is_retried_too() -> None:
    func = Counting(script=[TimeoutError("qdrant read timeout")])

    _, error, attempts = guarded(make_tool(func))

    assert (error, attempts, func.calls) == (None, 2, 2)


# --- what is never retried


def test_a_tool_with_a_side_effect_is_never_retried() -> None:
    func = Counting()
    sleeps = Sleeps()

    _, error, attempts = guarded(
        make_tool(func, side_effect=True), FaultPlan.from_env({FAULTS_ENV: "429"}), sleep=sleeps
    )

    assert isinstance(error, ToolUnavailable)
    assert attempts == 1
    assert sleeps.waits == []
    assert "не повторю" in str(error)


def test_a_side_effect_tool_failing_on_its_own_is_not_retried_either() -> None:
    func = Counting(script=[ConnectionError("reset")])

    _, error, attempts = guarded(make_tool(func, side_effect=True))

    assert isinstance(error, ToolUnavailable)
    assert (attempts, func.calls) == (1, 1)


@pytest.mark.parametrize(
    "failure", [ValueError("no such contract"), PermissionError("403"), KeyError("x")]
)
def test_an_error_that_is_not_transient_is_returned_at_once(failure: Exception) -> None:
    func = Counting(script=[failure])

    _, error, attempts = guarded(make_tool(func))

    assert error is failure
    assert (attempts, func.calls) == (1, 1)


def test_invalid_arguments_are_not_retried() -> None:
    func = Counting()

    _, error, attempts = guarded(make_tool(func), arguments={"query": 5})

    assert isinstance(error, InvalidArguments)
    assert (attempts, func.calls) == (1, 0)


# --- the output schema


def test_the_schema_fault_renames_a_field_and_the_answer_is_refused() -> None:
    func = Counting()
    sleeps = Sleeps()

    result, error, attempts = guarded(
        make_tool(func), FaultPlan.from_env({FAULTS_ENV: "schema"}), sleep=sleeps
    )

    assert result is None
    assert isinstance(error, SchemaMismatch)
    assert "hits" in str(error) and "search_corpus" in str(error)
    assert (attempts, func.calls) == (1, 1)  # the tool ran: its answer was changed
    assert sleeps.waits == []  # a changed schema is not cured by waiting


def test_a_real_answer_off_its_schema_is_refused_not_passed_on() -> None:
    func = Counting(script=['{"results": ["ЦК ст. 625"]}'])

    result, error, _ = guarded(make_tool(func))

    assert result is None
    assert isinstance(error, SchemaMismatch)


def test_an_answer_that_is_not_json_is_refused() -> None:
    _, error, _ = guarded(make_tool(Counting(script=["знайдено щось"])))

    assert isinstance(error, SchemaMismatch)


@pytest.mark.parametrize(
    ("text", "changed"),
    [
        ('{"hits": [1], "total": 1}', {"hits_v2": [1], "total": 1}),
        ('{"result": []}', {"result_v2": []}),
        ("[1, 2]", {"drifted": [1, 2]}),
        ("{}", {"drifted": {}}),
        ("не json", {"drifted": "не json"}),
    ],
)
def test_drift_renames_the_first_field_or_wraps_what_has_none(
    text: str, changed: dict[str, object]
) -> None:
    assert json.loads(drift(text)) == changed


# --- which errors are transient


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError(),
        ConnectionError(),
        InjectedStatus(429),
        InjectedStatus(503),
        MCPError(code=REQUEST_TIMEOUT, message="Timed out after 20s"),
        UnexpectedResponse(503, "Service Unavailable", b"", httpx.Headers()),
        UnexpectedResponse(429, "Too Many Requests", b"", httpx.Headers()),
        UnexpectedResponse(408, "Request Timeout", b"", httpx.Headers()),
        ResponseHandlingException(httpx.ConnectError("refused")),
        ResponseHandlingException(httpx.ReadTimeout("slow")),
    ],
)
def test_transient_errors(error: Exception) -> None:
    assert is_transient(error)


@pytest.mark.parametrize(
    "error",
    [
        ValueError(),
        MCPError(code=CONNECTION_CLOSED, message="Connection closed"),
        UnexpectedResponse(404, "Not Found", b"", httpx.Headers()),
        UnexpectedResponse(409, "Conflict", b"", httpx.Headers()),
        ResponseHandlingException(ValueError("reply is not JSON")),
        UnexpectedResponse(None, "?", b"", httpx.Headers()),
    ],
)
def test_errors_that_are_not_transient(error: Exception) -> None:
    assert not is_transient(error)


# --- the policy in configs/agent.yaml


def test_the_retry_policy_must_fit_in_the_tool_timeout() -> None:
    data = {
        "model": "haiku-5.5",
        "limits": LIMITS.model_copy(update={"tool_timeout_s": 1.0}).model_dump(),
        "retries": {"max_attempts": 3, "backoff_s": 0.5},  # waits 0.5 + 1.0 s
    }

    with pytest.raises(ValidationError, match="waits 1.5 s"):
        AgentConfig.model_validate(data)


@pytest.mark.parametrize("bad", [{"max_attempts": 0}, {"backoff_s": -1}, {"extra": 1}])
def test_a_bad_retry_policy_is_refused(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RetryPolicy.model_validate({"max_attempts": 3, "backoff_s": 0.5} | bad)


# --- in the agent


def make_agent(model: FakeModel, tool: Tool[QueryArgs], faults: FaultPlan) -> Agent:
    return Agent(
        model=model,
        tools=(tool,),
        system="Ти помічник фермера.",
        limits=LIMITS,
        prices=(1.0, 5.0),
        model_name="haiku-5.5",
        clock=Clock(),
        retry=POLICY,
        faults=faults,
        sleep=Sleeps(),
    )


def test_the_trace_shows_the_attempts_and_the_faults_of_the_run() -> None:
    model = FakeModel(turn("", ask(query="пеня")), turn("ЦК ст. 625."))
    plan = FaultPlan.from_env({FAULTS_ENV: "503"})

    trace = make_agent(model, make_tool(Counting()), plan).run("?")

    [call] = trace.steps[0].tool_calls
    assert (call.attempts, call.error) == (2, None)
    assert trace.faults == (FaultKind.UNAVAILABLE,)
    assert trace.stop_reason == StopReason.ANSWERED


def test_an_unavailable_tool_reaches_the_model_as_an_error_it_can_explain() -> None:
    model = FakeModel(turn("", ask(query="пеня")), turn("Сервіс зараз недоступний."))
    plan = FaultPlan.from_env({FAULTS_ENV: "503,503,503"})

    trace = make_agent(model, make_tool(Counting()), plan).run("?")

    [call] = trace.steps[0].tool_calls
    [result] = tool_results(model, 1)
    assert result["is_error"] is True
    assert result["content"] == call.error
    assert call.attempts == 3
    assert call.error is not None and call.error.startswith("search_corpus недоступний")


def test_a_schema_mismatch_is_in_the_trace_never_a_silent_result() -> None:
    model = FakeModel(turn("", ask(query="пеня")), turn("Не вийшло."))
    plan = FaultPlan.from_env({FAULTS_ENV: "schema"})

    trace = make_agent(model, make_tool(Counting()), plan).run("?")

    [call] = trace.steps[0].tool_calls
    assert call.result is None
    assert call.error is not None and "schema" in call.error


def test_a_long_tool_error_is_cut_for_the_model() -> None:
    func = Counting(script=[ValueError("x" * 5000)])
    model = FakeModel(turn("", ask(query="пеня")), turn("Не вийшло."))

    trace = make_agent(model, make_tool(func), NO_FAULTS).run("?")

    error = trace.steps[0].tool_calls[0].error
    assert error is not None and len(error) <= 500 and error.endswith("…")


def test_the_agent_starts_no_retry_pause_past_the_tool_timeout_of_the_turn() -> None:
    sleeps = Sleeps()
    model = FakeModel(turn("", ask(query="пеня")), turn("Сервіс недоступний."))
    agent = Agent(
        model=model,
        tools=(make_tool(Counting()),),
        system="Ти помічник фермера.",
        limits=LIMITS.model_copy(update={"tool_timeout_s": 0.2}),  # shorter than the 0.5 s pause
        prices=(1.0, 5.0),
        model_name="haiku-5.5",
        clock=Clock(),
        retry=POLICY,
        faults=FaultPlan.from_env({FAULTS_ENV: "503"}),
        sleep=sleeps,
    )

    trace = agent.run("?")

    [call] = trace.steps[0].tool_calls
    assert call.attempts == 1
    assert call.error is not None and call.error.startswith("search_corpus недоступний")
    assert sleeps.waits == []
