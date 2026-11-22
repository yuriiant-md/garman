from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from src.agents.faults import RetryPolicy
from src.agents.limits import AgentConfig, Limits, StopReason
from src.agents.reminders import REMINDERS_FILE, Reminder, ReminderKind
from src.agents.runner import (
    OUTBOX_NAME,
    RESULTS_NAME,
    SUMMARY_NAME,
    TRACES_NAME,
    RunnerSetup,
    run_task,
    run_tasks,
    summarize,
)
from src.agents.tasks import Checks, SeedReminder, Task, TaskKind, TaskResult
from src.agents.tools import AgentTool
from src.agents.trace import Trace
from src.rag.documents import read_jsonl
from tests.agent_fixtures import Clock, FakeModel, ask, turn
from tests.test_agent_graph import PRICES, NoArgs, found, tool

TODAY = date(2026, 11, 25)
STARTED = datetime(2026, 11, 25, 10, 0, 0)
RUN_STAMP = "2026-11-25T10-00-00"
CONFIG = AgentConfig(
    model="haiku-5.5",
    limits=Limits(
        max_steps=5, max_tokens=10_000, max_cost_usd=1.0, tool_timeout_s=5, request_timeout_s=60
    ),
    retries=RetryPolicy(max_attempts=3, backoff_s=0.25),  # not the default: the config is used
)
REMIND = {"contract_id": "own_001", "kind": "payment", "due_date": "2026-12-15", "note": "оплата"}


def list_contracts(args: NoArgs) -> str:
    return found("own_001, own_002")


LIST_CONTRACTS = tool("list_contracts", list_contracts, NoArgs)


@pytest.fixture
def contracts(tmp_path: Path) -> Path:
    root = tmp_path / "farmers"
    for farmer, contract_id in (("farmer_a", "own_001"), ("farmer_b", "own_004")):
        (root / farmer).mkdir(parents=True)
        (root / farmer / f"{contract_id}.pdf").write_bytes(b"%PDF-1.7")
    return root


class Setup:
    """Builds a RunnerSetup with a scripted model; keeps the farmers tools were asked for."""

    def __init__(self, contracts: Path) -> None:
        self.contracts = contracts
        self.farmers: list[str] = []
        self.sleeps: list[float] = []

    def __call__(
        self,
        model: FakeModel,
        tools: Callable[[str], tuple[AgentTool, ...]] | None = None,
        today: date = TODAY,
    ) -> RunnerSetup:
        def tools_for(farmer: str) -> tuple[AgentTool, ...]:
            self.farmers.append(farmer)
            return tools(farmer) if tools else (LIST_CONTRACTS,)

        return RunnerSetup(
            model=model,
            model_name="haiku-5.5",
            prices=PRICES,
            system="Ти помічник фермера.",
            prompt_version="v1",
            config=CONFIG,
            tools_for=tools_for,
            contracts_root=self.contracts,
            today=today,
            clock=Clock(),
            sleep=self.sleeps.append,
        )


@pytest.fixture
def setup(contracts: Path) -> Setup:
    return Setup(contracts)


def task(checks: Checks | None = None, **fields: Any) -> Task:
    defaults: dict[str, Any] = {"id": "t01", "kind": TaskKind.ANSWER, "question": "Мої договори?"}
    return Task(**{**defaults, **fields, "checks": checks or Checks()})


def reminders(outbox: Path, task_id: str, farmer: str = "farmer_a") -> list[Reminder]:
    path = outbox / task_id / farmer / REMINDERS_FILE
    return read_jsonl(path, Reminder) if path.exists() else []


def saving_model(answer: str = "Нагадування збережено.") -> FakeModel:
    return FakeModel(turn("", ask("save_reminder", "r1", **REMIND)), turn(answer))


# --- one task


def test_a_task_runs_the_agent_on_its_question_and_checks_the_run(
    setup: Setup, tmp_path: Path
) -> None:
    # Arrange
    model = FakeModel(turn("", ask("list_contracts", "c1")), turn("У вас own_001 і own_002."))
    checks = Checks(tools_required=("list_contracts",), answer_has=(("own_002",),))

    # Act
    result = run_task(task(checks), setup(model), tmp_path / "outbox", tmp_path / "traces.jsonl")

    # Assert
    assert result.passed, result.failures
    assert model.requests[0]["messages"] == [{"role": "user", "content": "Мої договори?"}]
    traces = read_jsonl(tmp_path / "traces.jsonl", Trace)
    assert [t.thread_id for t in traces] == [result.thread_id]
    assert traces[0].prompt_version == "v1"


def test_the_tools_are_those_of_the_tasks_farmer_plus_save_reminder(
    setup: Setup, tmp_path: Path
) -> None:
    model = FakeModel(turn("Добрий день!"))

    run_task(task(farmer="farmer_b"), setup(model), tmp_path / "outbox", None)

    assert setup.farmers == ["farmer_b"]
    names = [spec["name"] for spec in model.requests[0]["tools"]]  # type: ignore[attr-defined]
    assert names == ["list_contracts", "save_reminder"]


def test_a_yes_of_the_task_lets_the_reminder_be_written_once(
    setup: Setup, tmp_path: Path
) -> None:
    checks = Checks(tools_required=("save_reminder",), reminders_written=1)

    result = run_task(task(checks, confirm=True), setup(saving_model()), tmp_path / "ob", None)

    assert result.passed, result.failures
    assert (result.questions, result.unconfirmed_actions) == (1, 0)
    assert [r.due_date for r in reminders(tmp_path / "ob", "t01")] == [date(2026, 12, 15)]


def test_a_no_of_the_task_keeps_the_outbox_empty(setup: Setup, tmp_path: Path) -> None:
    result = run_task(
        task(confirm=False), setup(saving_model("Добре, не зберігаю.")), tmp_path / "ob", None
    )

    assert result.passed, result.failures
    assert result.questions == 1
    assert reminders(tmp_path / "ob", "t01") == []


def test_a_question_the_task_does_not_expect_is_answered_no_and_fails_it(
    setup: Setup, tmp_path: Path
) -> None:
    result = run_task(task(), setup(saving_model()), tmp_path / "ob", None)

    assert not result.passed
    assert "asked the farmer 1 times, the task expects no question" in result.failures
    assert reminders(tmp_path / "ob", "t01") == []


def test_a_seeded_reminder_is_found_again_and_nothing_new_is_written(
    setup: Setup, tmp_path: Path
) -> None:
    seed = SeedReminder(
        contract_id="own_001", kind=ReminderKind.PAYMENT, due_date=date(2026, 12, 15), note="раніше"
    )
    duplicate = task(
        Checks(tools_required=("save_reminder",), reminders_written=0),
        kind=TaskKind.DUPLICATE,
        confirm=True,
        seed_reminders=(seed,),
    )

    result = run_task(duplicate, setup(saving_model()), tmp_path / "ob", None)

    assert result.passed, result.failures
    assert [r.note for r in reminders(tmp_path / "ob", "t01")] == ["раніше"]


def test_save_reminder_judges_a_date_by_the_task_sets_day_not_the_clock(
    setup: Setup, tmp_path: Path
) -> None:
    late = setup(saving_model(), today=date(2030, 1, 1))

    result = run_task(task(confirm=True), late, tmp_path / "ob", None)

    # the past date is refused by the tool's check: the farmer is never asked
    assert "the farmer was never asked, yet the task answers yes" in result.failures
    assert reminders(tmp_path / "ob", "t01") == []


def test_the_faults_of_a_task_meet_its_tool_calls_and_are_retried(
    setup: Setup, tmp_path: Path
) -> None:
    model = FakeModel(turn("", ask("list_contracts", "c1")), turn("Сервіс договорів недоступний."))
    flaky = task(Checks(attempts_seen=3), kind=TaskKind.UNAVAILABLE, faults=("503",) * 3)

    result = run_task(flaky, setup(model), tmp_path / "ob", tmp_path / "traces.jsonl")

    assert result.passed, result.failures
    assert setup.sleeps == [0.25, 0.5]  # the retry policy of the config
    assert read_jsonl(tmp_path / "traces.jsonl", Trace)[0].faults == ("503",) * 3


def test_the_limits_of_a_task_tighten_the_configs(setup: Setup, tmp_path: Path) -> None:
    model = FakeModel(turn("", ask("list_contracts", "c1"), tokens=(900, 200)), turn("own_001"))
    tight = task(Checks(stop=StopReason.BUDGET), kind=TaskKind.BUDGET, limits={"max_tokens": 1000})

    result = run_task(tight, setup(model), tmp_path / "ob", None)

    assert result.passed, result.failures
    assert result.steps == 1


# --- a run of the set


def test_a_finished_run_is_a_folder_with_results_summary_traces_and_outbox(
    setup: Setup, tmp_path: Path
) -> None:
    tasks = (task(), task(Checks(reminders_written=1), id="t02", confirm=True))
    model = FakeModel(
        turn("own_001"), turn("", ask("save_reminder", "r1", **REMIND)), turn("Збережено.")
    )

    run_dir = run_tasks(tasks, setup(model), tmp_path / "tasks", STARTED)

    assert run_dir == tmp_path / "tasks" / RUN_STAMP
    assert sorted(path.name for path in run_dir.iterdir()) == sorted(
        [OUTBOX_NAME, RESULTS_NAME, SUMMARY_NAME, TRACES_NAME]
    )
    results = read_jsonl(run_dir / RESULTS_NAME, TaskResult)
    assert [(r.task_id, r.passed) for r in results] == [("t01", True), ("t02", True)]
    assert len(read_jsonl(run_dir / TRACES_NAME, Trace)) == 3  # t02 asked, then finished
    assert not list((tmp_path / "tasks").glob("*.partial"))
    assert reminders(run_dir / OUTBOX_NAME, "t01") == []  # each task has its own outbox
    assert len(reminders(run_dir / OUTBOX_NAME, "t02")) == 1


def test_the_summary_names_the_run_its_metrics_and_each_tasks_failures(
    setup: Setup, tmp_path: Path
) -> None:
    tasks = (task(), task(Checks(answer_has=(("own_009",),)), id="t02"))

    run_dir = run_tasks(tasks, setup(FakeModel(turn("own_001"))), tmp_path / "tasks", STARTED)

    summary = (run_dir / SUMMARY_NAME).read_text(encoding="utf-8")
    assert "haiku-5.5" in summary and "v1" in summary
    assert "1 з 2" in summary
    assert "the answer has none of: own_009" in summary


def test_a_run_that_crashes_keeps_its_partial_folder_and_leaves_no_finished_one(
    setup: Setup, tmp_path: Path
) -> None:
    def tools(farmer: str) -> tuple[AgentTool, ...]:
        if farmer == "farmer_b":
            raise RuntimeError("the MCP server of farmer_b did not start")
        return (LIST_CONTRACTS,)

    tasks = (task(), task(id="t02", farmer="farmer_b"))

    with pytest.raises(RuntimeError, match="farmer_b"):
        run_tasks(tasks, setup(FakeModel(turn("own_001")), tools), tmp_path / "tasks", STARTED)

    partial = tmp_path / "tasks" / f"{RUN_STAMP}.partial"
    assert [r.task_id for r in read_jsonl(partial / RESULTS_NAME, TaskResult)] == ["t01"]
    assert not (tmp_path / "tasks" / RUN_STAMP).exists()


@pytest.mark.parametrize("left", [RUN_STAMP, f"{RUN_STAMP}.partial"])
def test_a_run_never_overwrites_a_run_of_the_same_second(
    setup: Setup, tmp_path: Path, left: str
) -> None:
    (tmp_path / "tasks" / left).mkdir(parents=True)
    (tmp_path / "tasks" / left / RESULTS_NAME).write_text("kept\n", encoding="utf-8")
    model = FakeModel(turn("own_001"))

    with pytest.raises(FileExistsError):
        run_tasks((task(),), setup(model), tmp_path / "tasks", STARTED)

    assert (tmp_path / "tasks" / left / RESULTS_NAME).read_text(encoding="utf-8") == "kept\n"
    assert model.requests == []  # refused before any paid call


# --- metrics


def result(task_id: str, **fields: Any) -> TaskResult:
    defaults: dict[str, Any] = {
        "task_id": task_id,
        "kind": TaskKind.ANSWER,
        "passed": True,
        "failures": (),
        "required_tools_called": True,
        "unconfirmed_actions": 0,
        "questions": 0,
        "tools_called": (),
        "stop_reason": StopReason.ANSWERED,
        "steps": 2,
        "cost_usd": 0.002,
        "latency_s": 3.0,
        "answer": "так",
        "thread_id": task_id,
    }
    return TaskResult(**{**defaults, **fields})


def test_the_metrics_of_a_run() -> None:
    results = [
        result("t01"),
        result("t02", passed=False, failures=("x",), required_tools_called=False, steps=4),
        result("t03", passed=False, failures=("y",), unconfirmed_actions=1, cost_usd=0.004),
        result("t04", latency_s=9.0),
    ]

    summary = summarize(results)

    assert (summary.tasks, summary.passed) == (4, 2)
    assert summary.pass_rate == 0.5
    assert summary.required_tools_rate == 0.75
    assert summary.unconfirmed_actions == 1
    assert (summary.steps_mean, summary.steps_max) == (2.5, 4)
    assert summary.cost_usd == pytest.approx(0.010)
    assert summary.cost_per_task_usd == pytest.approx(0.0025)
    assert summary.latency_p50_s == 3.0
