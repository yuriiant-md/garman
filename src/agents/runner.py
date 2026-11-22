"""Runs the agent's task set (tasks.py) and keeps each run in a folder of its own.

Each task gets an agent of its own: its faults (faults.FaultPlan) and its tightened limits, the
read-only tools of its farmer (setup.tools_for: the MCP servers and the corpus search) and a
save_reminder writing to an outbox of the task alone, so a reminder of one task is never found
by the next one and the duplicate task finds only the reminder it was seeded with. save_reminder
judges a date by the task set's day, so the tasks' dates stay in the future.

When the agent asks the farmer about an action, the runner answers with the task's `confirm`.
A question the task does not expect is answered no, and check_task fails the task for it.

A run is written to results/stage3/tasks/<start>.partial/: results.jsonl gets a line as each task
ends, traces.jsonl every trace line, outbox/ the reminders. summary.md is written last, then the
folder is renamed to <start>/. A run that crashes stays .partial, for a look at what it did;
a folder of the same start, finished or not, is never overwritten.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.agents.faults import FaultPlan, Sleep
from src.agents.graph import Agent
from src.agents.limits import AgentConfig
from src.agents.model import ChatModel
from src.agents.reminder_tool import save_reminder_tool
from src.agents.reminders import ReminderStore
from src.agents.tasks import SeedReminder, Task, TaskResult, TaskRun, check_task
from src.agents.tools import AgentTool
from src.agents.trace import Trace
from src.extraction.metrics import percentile
from src.extraction.run import PARTIAL_SUFFIX, RUN_STAMP_FORMAT
from src.servers.contract_access import ContractShelf

RESULTS_DIR = Path("results/stage3/tasks")
RESULTS_NAME = "results.jsonl"
SUMMARY_NAME = "summary.md"
TRACES_NAME = "traces.jsonl"
OUTBOX_NAME = "outbox"


@dataclass(frozen=True)
class RunnerSetup:
    model: ChatModel
    model_name: str  # the models.yaml section
    prices: tuple[float, float]  # USD per 1M input and output tokens
    system: str  # the agent prompt with what the MCP servers ask of a client
    prompt_version: str
    config: AgentConfig  # configs/agent.yaml: the limits a task may tighten, the retries
    tools_for: Callable[[str], tuple[AgentTool, ...]]  # a farmer's read-only tools
    contracts_root: Path  # the shelf save_reminder checks a contract on
    today: date  # the task set's day
    clock: Callable[[], float] = field(default=time.monotonic)
    sleep: Sleep = field(default=time.sleep)


class RunSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tasks: int
    passed: int
    pass_rate: float  # 0..1
    required_tools_rate: float  # of the tasks with every required tool called
    unconfirmed_actions: int  # over all tasks; any one is a bug of the agent
    steps_mean: float
    steps_max: int
    cost_usd: float
    cost_per_task_usd: float
    latency_p50_s: float
    latency_p95_s: float


def run_task(task: Task, setup: RunnerSetup, outbox: Path, trace_file: Path | None) -> TaskResult:
    """One task from the question to the check; the outbox gets a folder of this task."""
    store = ReminderStore(outbox / task.id, task.farmer)
    _seed(store, task.seed_reminders)
    before = len(store.all())
    shelf = ContractShelf(setup.contracts_root, task.farmer)
    reminder = save_reminder_tool(store, shelf, today=lambda: setup.today)
    tools = (*setup.tools_for(task.farmer), reminder)
    agent = Agent(
        model=setup.model,
        tools=tools,
        system=setup.system,
        limits=task.limits_from(setup.config),
        prices=setup.prices,
        model_name=setup.model_name,
        prompt_version=setup.prompt_version,
        clock=setup.clock,
        retry=setup.config.retries,
        faults=FaultPlan(kinds=task.faults),
        sleep=setup.sleep,
    )
    traces, approved = _converse(agent, task, trace_file)
    run = TaskRun(
        traces=traces,
        approved=approved,
        reminders_written=len(store.all()) - before,
        side_effects=frozenset(tool.name for tool in tools if tool.side_effect),
    )
    return check_task(task, run)


def run_tasks(
    tasks: Sequence[Task], setup: RunnerSetup, results_dir: Path, started: datetime
) -> Path:
    """Run every task in order; returns the finished run's folder."""
    stamp = started.strftime(RUN_STAMP_FORMAT)
    finished = results_dir / stamp
    if finished.exists():
        raise FileExistsError(f"{finished}: a finished run is never overwritten")
    partial = results_dir / f"{stamp}{PARTIAL_SUFFIX}"
    partial.mkdir(parents=True)  # a crashed run of this second is kept as it is: FileExistsError
    results = []
    with (partial / RESULTS_NAME).open("w", encoding="utf-8") as file:
        for task in tasks:
            result = run_task(task, setup, partial / OUTBOX_NAME, partial / TRACES_NAME)
            file.write(f"{result.model_dump_json()}\n")
            file.flush()
            results.append(result)
            print(_progress(result), flush=True)
    summary = render_summary(summarize(results), results, setup, started)
    (partial / SUMMARY_NAME).write_text(summary, encoding="utf-8")
    partial.rename(finished)
    return finished


def summarize(results: Sequence[TaskResult]) -> RunSummary:
    if not results:
        raise ValueError("no task results to summarize")
    count = len(results)
    steps = [result.steps for result in results]
    latencies = [result.latency_s for result in results]
    cost = sum(result.cost_usd for result in results)
    passed = sum(result.passed for result in results)
    return RunSummary(
        tasks=count,
        passed=passed,
        pass_rate=passed / count,
        required_tools_rate=sum(result.required_tools_called for result in results) / count,
        unconfirmed_actions=sum(result.unconfirmed_actions for result in results),
        steps_mean=sum(steps) / count,
        steps_max=max(steps),
        cost_usd=cost,
        cost_per_task_usd=cost / count,
        latency_p50_s=percentile(latencies, 50),
        latency_p95_s=percentile(latencies, 95),
    )


def render_summary(
    summary: RunSummary, results: Sequence[TaskResult], setup: RunnerSetup, started: datetime
) -> str:
    header = (
        f"Прогін: {started:%Y-%m-%d %H:%M} · модель: {setup.model_name} · "
        f"промпт: {setup.prompt_version} · день задач: {setup.today.isoformat()} · "
        f"задач: {summary.tasks}"
    )
    metrics = [
        ("Пройдено", f"{summary.passed} з {summary.tasks} ({summary.pass_rate:.0%})"),
        ("Потрібні інструменти викликано", f"{summary.required_tools_rate:.0%} задач"),
        ("Дій без підтвердження фермера", str(summary.unconfirmed_actions)),
        ("Кроків на задачу", f"{summary.steps_mean:.1f} у середньому, до {summary.steps_max}"),
        ("Вартість", f"${summary.cost_usd:.4f} разом, ${summary.cost_per_task_usd:.4f} на задачу"),
        ("Час відповіді", f"p50 {summary.latency_p50_s:.1f} с, p95 {summary.latency_p95_s:.1f} с"),
    ]
    lines = [
        "# Набір задач агента",
        "",
        header,
        "",
        "| Метрика | Значення |",
        "| --- | --- |",
        *(f"| {name} | {value} |" for name, value in metrics),
        "",
        "| Задача | Тип | Результат | Кроки | Вартість, $ | Час, с | Інструменти | Що не так |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *(_row(result) for result in results),
    ]
    return "\n".join(lines) + "\n"


def _converse(
    agent: Agent, task: Task, trace_file: Path | None
) -> tuple[tuple[Trace, ...], frozenset[str]]:
    """The run and every answer of the farmer; the ids of the calls the farmer said yes to."""
    trace = agent.run(task.question, trace_file)
    traces = [trace]
    approved: set[str] = set()
    # each question costs a model call, so a run asks at most max_steps times
    for _ in range(agent.limits.max_steps):
        if not trace.awaiting:
            break
        yes = task.confirm is True  # a question the task does not expect is declined
        if yes:
            approved.update(action.call_id for action in trace.awaiting)
        trace = agent.resume(trace.thread_id, approved=yes, trace_file=trace_file)
        traces.append(trace)
    return tuple(traces), frozenset(approved)


def _seed(store: ReminderStore, seeds: Sequence[SeedReminder]) -> None:
    now = datetime.now().astimezone()
    for seed in seeds:
        store.save(
            contract_id=seed.contract_id,
            kind=seed.kind,
            due_date=seed.due_date,
            note=seed.note,
            now=now,
        )


def _progress(result: TaskResult) -> str:
    mark = "✓" if result.passed else "✗"
    line = f"{result.task_id}: {mark} {result.steps} steps, ${result.cost_usd:.4f}"
    return f"{line} · {'; '.join(result.failures)}" if result.failures else line


def _row(result: TaskResult) -> str:
    cells = (
        result.task_id,
        result.kind.value,
        "✓" if result.passed else "✗",
        str(result.steps),
        f"{result.cost_usd:.4f}",
        f"{result.latency_s:.1f}",
        ", ".join(result.tools_called) or "—",
        "; ".join(result.failures) or "—",
    )
    return "| " + " | ".join(_cell(cell) for cell in cells) + " |"


def _cell(text: str) -> str:
    """A table cell: a "|" or a line break would end it early."""
    return " ".join(text.replace("|", "\\|").split())
