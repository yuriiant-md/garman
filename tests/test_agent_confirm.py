"""A tool with a side effect runs only after the farmer's yes, given from outside the model."""

import json
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from pydantic import JsonValue

from src.agents.graph import DECLINED, Agent, NothingToConfirm
from src.agents.limits import Limits, StopReason
from src.agents.reminder_tool import save_reminder_tool
from src.agents.reminders import ReminderStore
from src.agents.tools import AgentTool, Tool, ToolDescription, ToolRefused
from src.agents.trace import PendingAction, SummaryCall, Trace
from src.rag.documents import read_jsonl
from src.servers.contract_access import ContractShelf
from tests.agent_fixtures import Clock, FakeModel, ask, tool_results, turn
from tests.test_agent_graph import PRICES, SEARCH, Found, QueryArgs, found

LIMITS = Limits(
    max_steps=6, max_tokens=10_000, max_cost_usd=1.0, tool_timeout_s=1, request_timeout_s=60
)
TODAY = date(2026, 11, 22)
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
REMINDER: dict[str, JsonValue] = {
    "contract_id": "k_001",
    "kind": "payment",
    "due_date": "2026-12-15",
    "note": "оплата за пшеницю, п. 5.2",
}


def remind(call_id: str = "r1", **changes: JsonValue) -> object:
    return ask("save_reminder", call_id, **{**REMINDER, **changes})


@pytest.fixture
def store(tmp_path: Path) -> ReminderStore:
    return ReminderStore(root=tmp_path / "outbox", farmer_id="farmer_a")


@pytest.fixture
def reminder_tool(tmp_path: Path, store: ReminderStore) -> AgentTool:
    folder = tmp_path / "contracts" / "farmer_a"
    folder.mkdir(parents=True)
    (folder / "k_001.pdf").write_bytes(b"%PDF-1.7")
    shelf = ContractShelf(root=tmp_path / "contracts", farmer_id="farmer_a")
    return save_reminder_tool(store, shelf, today=lambda: TODAY, now=lambda: NOW)


def make_agent(model: FakeModel, reminder_tool: AgentTool, clock: Clock | None = None) -> Agent:
    return Agent(
        model=model,
        tools=(SEARCH, reminder_tool),
        system="Ти помічник фермера.",
        limits=LIMITS,
        prices=PRICES,
        model_name="haiku-5.5",
        prompt_version="v1",
        clock=clock or Clock(),
    )


def asks_to_remind(*calls: object) -> FakeModel:
    return FakeModel(
        turn("Збережу нагадування.", *calls),  # type: ignore[arg-type]
        turn("Готово: нагадаю 15 грудня."),
    )


# --- the run stops before the side effect


def test_a_side_effect_waits_for_the_farmer_and_nothing_is_written(
    reminder_tool: AgentTool, store: ReminderStore
) -> None:
    model = asks_to_remind(remind())

    trace = make_agent(model, reminder_tool).run("Нагадай про оплату 15 грудня")

    assert trace.stop_reason == StopReason.AWAITING_CONFIRMATION
    assert trace.answer is None
    assert not trace.partial  # not stopped by a limit: waiting for the farmer
    assert trace.awaiting == (
        PendingAction(
            call_id="r1", tool="save_reminder", tool_version="v1", arguments=REMINDER
        ),
    )
    assert store.all() == ()
    assert len(model.requests) == 1  # the model is not called while the farmer decides


def test_yes_runs_the_call_once_and_the_model_reads_its_result(
    reminder_tool: AgentTool, store: ReminderStore
) -> None:
    model = asks_to_remind(remind())
    agent = make_agent(model, reminder_tool)
    waiting = agent.run("Нагадай про оплату 15 грудня")

    trace = agent.resume(waiting.thread_id, approved=True)

    assert trace.stop_reason == StopReason.ANSWERED
    assert trace.answer == "Готово: нагадаю 15 грудня."
    assert trace.thread_id == waiting.thread_id
    assert trace.awaiting == ()
    [reminder] = store.all()
    [result] = tool_results(model, 1)
    assert json.loads(str(result["content"]))["status"] == "saved"
    [call] = trace.steps[0].tool_calls
    assert (call.name, call.attempts, call.error) == ("save_reminder", 1, None)
    assert json.loads(str(call.result))["key"] == reminder.key


def test_after_the_yes_the_model_reads_the_same_context_and_the_summary_stays(
    reminder_tool: AgentTool,
) -> None:
    model = asks_to_remind(remind())
    agent = make_agent(model, reminder_tool)
    summary = SummaryCall(prompt_version="v1", folded_turns=2, cost_usd=0.0003)
    history = ({"role": "user", "content": "Яка пеня?"}, {"role": "assistant", "content": "0,1 %"})
    waiting = agent.run(
        "Нагадай про оплату 15 грудня",
        history=history,
        context="# Farmer profile\n- region: Одеська область",
        summary=summary,
    )

    trace = agent.resume(waiting.thread_id, approved=True)

    # the request goes on where it stopped: the context is the run's, not built again
    assert model.requests[1]["system"] == model.requests[0]["system"]
    assert "Одеська область" in str(model.requests[1]["system"])
    assert model.requests[1]["messages"][:2] == list(history)  # type: ignore[index]
    assert waiting.summary == trace.summary == summary  # each line of the thread is the request


def test_no_writes_nothing_and_tells_the_model_not_to_retry(
    reminder_tool: AgentTool, store: ReminderStore
) -> None:
    model = asks_to_remind(remind())
    agent = make_agent(model, reminder_tool)
    waiting = agent.run("Нагадай про оплату 15 грудня")

    trace = agent.resume(waiting.thread_id, approved=False)

    assert store.all() == ()
    [result] = tool_results(model, 1)
    assert result["is_error"] is True
    assert result["content"] == DECLINED
    [call] = trace.steps[0].tool_calls
    assert (call.error, call.attempts) == (DECLINED, 0)
    assert trace.stop_reason == StopReason.ANSWERED


def test_a_second_yes_to_the_same_question_is_refused_and_writes_nothing_more(
    reminder_tool: AgentTool, store: ReminderStore
) -> None:
    agent = make_agent(asks_to_remind(remind()), reminder_tool)
    waiting = agent.run("Нагадай про оплату 15 грудня")
    agent.resume(waiting.thread_id, approved=True)

    with pytest.raises(NothingToConfirm):
        agent.resume(waiting.thread_id, approved=True)  # a double tap in the bot
    assert len(store.all()) == 1


def test_an_unknown_thread_has_nothing_to_confirm(reminder_tool: AgentTool) -> None:
    agent = make_agent(asks_to_remind(remind()), reminder_tool)

    with pytest.raises(NothingToConfirm):
        agent.resume("no-such-thread", approved=True)


def test_the_same_reminder_confirmed_in_two_requests_is_saved_once(
    reminder_tool: AgentTool, store: ReminderStore
) -> None:
    model = FakeModel(
        turn("Збережу.", remind()),
        turn("Готово."),
        turn("Збережу ще раз.", remind(note="ще раз про оплату")),
        turn("Таке нагадування вже є."),
    )
    agent = make_agent(model, reminder_tool)
    agent.resume(agent.run("Нагадай про оплату").thread_id, approved=True)

    agent.resume(agent.run("Нагадай ще раз про оплату").thread_id, approved=True)

    assert len(store.all()) == 1
    [result] = tool_results(model, 3)
    assert json.loads(str(result["content"]))["status"] == "exists"


@pytest.mark.parametrize("answer", ["ні", "так", 1, None])
def test_only_a_bool_is_an_answer_and_nothing_is_written_on_another(
    reminder_tool: AgentTool, store: ReminderStore, answer: object
) -> None:
    agent = make_agent(asks_to_remind(remind()), reminder_tool)
    waiting = agent.run("Нагадай про оплату")

    with pytest.raises(TypeError, match="bool"):
        agent.resume(waiting.thread_id, approved=answer)  # type: ignore[arg-type]
    assert store.all() == ()
    assert agent.resume(waiting.thread_id, approved=False).stop_reason == StopReason.ANSWERED


class Gate:
    """A side-effect tool whose check passes only the queries let through, and may be slow."""

    def __init__(self, passing: set[str], check_s: float = 0.0) -> None:
        self.passing = passing
        self.check_s = check_s
        self.writes: list[str] = []

    def tool(self) -> AgentTool:
        def check(args: QueryArgs) -> None:
            time.sleep(self.check_s)
            if args.query not in self.passing:
                raise ToolRefused(f"{args.query}: ще не можна")

        def write(args: QueryArgs) -> str:
            self.writes.append(args.query)
            return found(args.query)

        description = ToolDescription(
            name="send", version="v1", side_effect=True, description="send for tests"
        )
        return Tool(description=description, args=QueryArgs, func=write, output=Found, check=check)


def test_a_yes_approves_only_what_the_farmer_was_shown() -> None:
    gate = Gate(passing={"a"})
    both = turn("", ask("send", "A", query="a"), ask("send", "B", query="b"))
    model = FakeModel(both, turn("ok"))
    agent = make_agent(model, gate.tool())
    waiting = agent.run("Надішли обидва")
    assert [action.call_id for action in waiting.awaiting] == ["A"]

    gate.passing.add("b")  # B would pass now, but the farmer was never asked about it
    agent.resume(waiting.thread_id, approved=True)

    assert gate.writes == ["a"]
    assert tool_results(model, 1)[1]["is_error"] is True


def test_a_double_tap_at_once_runs_the_call_once() -> None:
    gate = Gate(passing={"a"}, check_s=0.2)  # both taps arrive while the first is checking
    model = FakeModel(turn("", ask("send", "A", query="a")), turn("ok"))
    agent = make_agent(model, gate.tool())
    thread_id = agent.run("Надішли").thread_id
    outcomes: list[object] = []

    def tap() -> None:
        try:
            outcomes.append(agent.resume(thread_id, approved=True).stop_reason)
        except NothingToConfirm as error:
            outcomes.append(type(error))

    taps = [threading.Thread(target=tap) for _ in range(2)]
    for one in taps:
        one.start()
    for one in taps:
        one.join()

    assert gate.writes == ["a"]
    assert sorted(map(str, outcomes)) == sorted(map(str, [StopReason.ANSWERED, NothingToConfirm]))
    assert len(model.requests) == 2


def test_a_finished_request_is_dropped_from_the_checkpoint(reminder_tool: AgentTool) -> None:
    agent = make_agent(asks_to_remind(remind()), reminder_tool)
    waiting = agent.run("Нагадай про оплату")
    config = {"configurable": {"thread_id": waiting.thread_id}}
    assert agent.checkpointer.get_tuple(config) is not None  # type: ignore[arg-type]

    agent.resume(waiting.thread_id, approved=True)

    assert agent.checkpointer.get_tuple(config) is None  # type: ignore[arg-type]


# --- refused before the farmer is asked


def test_a_missing_argument_goes_back_to_the_model_without_asking_the_farmer(
    reminder_tool: AgentTool, store: ReminderStore
) -> None:
    call = ask("save_reminder", "r1", contract_id="k_001", kind="payment", note="оплата")
    model = FakeModel(turn("Збережу.", call), turn("На яку дату нагадати?"))

    trace = make_agent(model, reminder_tool).run("Нагадай про оплату")

    assert trace.stop_reason == StopReason.ANSWERED
    assert trace.answer == "На яку дату нагадати?"
    [result] = tool_results(model, 1)
    assert result["is_error"] is True
    assert "due_date" in str(result["content"])
    assert store.all() == ()


@pytest.mark.parametrize(
    ("changes", "words"),
    [({"due_date": "2026-11-01"}, "минула"), ({"contract_id": "k_404"}, "404")],
)
def test_a_call_the_tool_would_refuse_is_not_put_to_the_farmer(
    reminder_tool: AgentTool,
    store: ReminderStore,
    changes: dict[str, JsonValue],
    words: str,
) -> None:
    model = FakeModel(turn("Збережу.", remind(**changes)), turn("Не вийшло."))

    trace = make_agent(model, reminder_tool).run("Нагадай про оплату")

    assert trace.stop_reason == StopReason.ANSWERED
    [result] = tool_results(model, 1)
    assert words in str(result["content"])
    assert store.all() == ()


# --- several calls in one turn


def test_two_reminders_in_one_turn_are_put_to_the_farmer_together(
    reminder_tool: AgentTool, store: ReminderStore
) -> None:
    model = asks_to_remind(remind("r1"), remind("r2", kind="delivery", due_date="2026-12-01"))
    agent = make_agent(model, reminder_tool)

    waiting = agent.run("Нагадай про оплату і поставку")
    assert [action.call_id for action in waiting.awaiting] == ["r1", "r2"]

    agent.resume(waiting.thread_id, approved=False)
    assert store.all() == ()
    assert [result["content"] for result in tool_results(model, 1)] == [DECLINED, DECLINED]


def test_a_read_only_call_in_the_same_turn_waits_too_and_runs_after_the_answer(
    reminder_tool: AgentTool, store: ReminderStore
) -> None:
    model = asks_to_remind(ask("search_corpus", "s1", query="строк оплати"), remind())
    agent = make_agent(model, reminder_tool)
    waiting = agent.run("Нагадай про оплату")

    assert [action.call_id for action in waiting.awaiting] == ["r1"]  # only the side effect
    agent.resume(waiting.thread_id, approved=False)

    search, reminder = tool_results(model, 1)
    assert "знайдено" in str(search["content"])
    assert reminder["content"] == DECLINED


def test_a_bad_call_beside_a_good_one_does_not_reach_the_farmer(
    reminder_tool: AgentTool, store: ReminderStore
) -> None:
    model = asks_to_remind(remind("r1"), remind("r2", due_date="2026-11-01"))
    agent = make_agent(model, reminder_tool)

    waiting = agent.run("Нагадай двічі")
    assert [action.call_id for action in waiting.awaiting] == ["r1"]

    agent.resume(waiting.thread_id, approved=True)
    good, bad = tool_results(model, 1)
    assert json.loads(str(good["content"]))["status"] == "saved"
    assert "минула" in str(bad["content"])
    assert len(store.all()) == 1


# --- time and trace


def test_the_farmers_time_to_decide_does_not_count_against_the_request_timeout(
    reminder_tool: AgentTool,
) -> None:
    clock = Clock()
    agent = make_agent(asks_to_remind(remind()), reminder_tool, clock=clock)
    waiting = agent.run("Нагадай про оплату")

    clock.now += 600  # ten minutes in the bot, past the request's 60 s
    trace = agent.resume(waiting.thread_id, approved=True)

    assert trace.stop_reason == StopReason.ANSWERED
    assert trace.paused_s == 600
    assert trace.latency_s == 0  # the hand-moved clock: the agent itself took no time


def test_the_pauses_of_two_questions_in_one_request_add_up(reminder_tool: AgentTool) -> None:
    clock = Clock()
    model = FakeModel(
        turn("Збережу оплату.", remind("r1")),
        turn("Збережу поставку.", remind("r2", kind="delivery", due_date="2026-12-01")),
        turn("Готово."),
    )
    agent = make_agent(model, reminder_tool, clock=clock)
    thread_id = agent.run("Нагадай про оплату і поставку").thread_id

    clock.now += 100
    second_question = agent.resume(thread_id, approved=True)
    clock.now += 100
    trace = agent.resume(thread_id, approved=True)

    assert second_question.stop_reason == StopReason.AWAITING_CONFIRMATION
    assert trace.paused_s == 200
    assert trace.stop_reason == StopReason.ANSWERED


def test_the_run_and_its_continuation_are_two_lines_of_one_thread(
    reminder_tool: AgentTool, tmp_path: Path
) -> None:
    agent = make_agent(asks_to_remind(remind()), reminder_tool)
    traces = tmp_path / "traces.jsonl"
    waiting = agent.run("Нагадай про оплату", trace_file=traces)

    agent.resume(waiting.thread_id, approved=True, trace_file=traces)

    first, second = read_jsonl(traces, Trace)
    assert first.thread_id == second.thread_id == waiting.thread_id
    assert first.request_id != second.request_id
    assert (first.stop_reason, second.stop_reason) == (
        StopReason.AWAITING_CONFIRMATION,
        StopReason.ANSWERED,
    )
    assert second.question == "Нагадай про оплату"


def test_a_trace_awaits_a_confirmation_only_when_it_stopped_for_one() -> None:
    action = PendingAction(call_id="r1", tool="save_reminder", tool_version="v1", arguments={})
    fields: dict[str, object] = {
        "request_id": "a",
        "thread_id": "t",
        "started_at": NOW,
        "model": "haiku-5.5",
        "prompt_version": "v1",
        "question": "?",
        "steps": (),
        "usage": {},
        "answer": None,
        "latency_s": 0,
    }

    with pytest.raises(ValueError, match="awaiting"):
        Trace.model_validate(
            {**fields, "stop_reason": StopReason.ANSWERED, "awaiting": (action,)}
        )
    with pytest.raises(ValueError, match="awaiting"):
        Trace.model_validate({**fields, "stop_reason": StopReason.AWAITING_CONFIRMATION})
