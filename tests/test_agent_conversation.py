"""A conversation of several questions and several sessions: the agent with both memories."""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.agents.conversation import Memory, ask, confirm
from src.agents.graph import PARTIAL_HEADERS, Agent
from src.agents.limits import Limits, MemoryConfig, StopReason
from src.agents.memory import Dialogue, Summarizer
from src.agents.profile import FactKey, ProfileStore
from src.agents.profile_tool import profile_tools
from tests.agent_fixtures import Clock, FakeModel, turn
from tests.agent_fixtures import ask as tool_request
from tests.test_agent_graph import PRICES, SEARCH

LIMITS = Limits(
    max_steps=3, max_tokens=10_000, max_cost_usd=1.0, tool_timeout_s=1, request_timeout_s=60
)
NOW = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
TTL_DAYS = {key: 365 for key in FactKey}
SYSTEM = "Ти помічник фермера."
REMEMBER_REGION = tool_request("remember_fact", "m1", key="region", value="Одеська область")


@pytest.fixture
def profile(tmp_path: Path) -> ProfileStore:
    return ProfileStore(tmp_path / "profile.sqlite", "farmer_a", TTL_DAYS)


def make_agent(model: FakeModel, profile: ProfileStore) -> Agent:
    """A session: a new agent each time, with the same profile file as the last one."""
    return Agent(
        model=model,
        tools=(SEARCH, *profile_tools(profile, now=lambda: NOW)),
        system=SYSTEM,
        limits=LIMITS,
        prices=PRICES,
        model_name="haiku-5.5",
        prompt_version="v3",
        clock=Clock(),
    )


def make_memory(
    profile: ProfileStore, summary_model: FakeModel | None = None, cap: int = 2000
) -> Memory:
    config = MemoryConfig(
        history_max_tokens=cap, summary_max_tokens=30, keep_turns=1, fact_ttl_days=TTL_DAYS
    )
    summarizer = Summarizer(
        model=summary_model or FakeModel(turn("не має викликатися")),
        prompt="Стисни розмову.",
        prompt_version="v1",
        prices=PRICES,
        memory=config,
    )
    return Memory(summarizer=summarizer, profile=profile, now=lambda: NOW)


def system_of(model: FakeModel, request: int = 0) -> str:
    return str(model.requests[request]["system"])


# --- one session, several questions


def test_the_first_question_has_no_context_and_becomes_the_first_turn(
    profile: ProfileStore,
) -> None:
    model = FakeModel(turn("Протягом 5 днів, п. 5.2."))

    trace, dialogue = ask(make_agent(model, profile), make_memory(profile), Dialogue(), "Коли?")

    assert system_of(model) == SYSTEM  # nothing remembered yet: the prompt as it is
    assert trace.stop_reason == StopReason.ANSWERED
    assert [(t.question, t.answer) for t in dialogue.turns] == [
        ("Коли?", "Протягом 5 днів, п. 5.2.")
    ]


def test_the_next_question_reads_the_dialogue_so_far(profile: ProfileStore) -> None:
    model = FakeModel(turn("Протягом 5 днів, п. 5.2."), turn("Тоді до 15 грудня."))
    agent, memory = make_agent(model, profile), make_memory(profile)
    _, dialogue = ask(agent, memory, Dialogue(), "Коли мені заплатять?")

    _, dialogue = ask(agent, memory, dialogue, "А якщо поставка 10 грудня?")

    assert model.requests[1]["messages"] == [
        {"role": "user", "content": "Коли мені заплатять?"},
        {"role": "assistant", "content": "Протягом 5 днів, п. 5.2."},
        {"role": "user", "content": "А якщо поставка 10 грудня?"},
    ]
    assert len(dialogue.turns) == 2


def test_a_dialogue_over_the_cap_is_summarized_before_the_question(
    profile: ProfileStore,
) -> None:
    summary_model = FakeModel(turn("Фермер питав про пеню own_001: 0,1 % за день."))
    model = FakeModel(turn("Так."))
    long = Dialogue().with_turn("Яка пеня? " + "п" * 200, "0,1 % за день. " + "в" * 200)
    dialogue = long.with_turn("А строк?", "5 днів.")

    trace, after = ask(
        make_agent(model, profile), make_memory(profile, summary_model, cap=100), dialogue, "Ок?"
    )

    messages = model.requests[0]["messages"]
    assert system_of(model) == SYSTEM  # the summary is data: among the messages, not the system
    assert "Фермер питав про пеню own_001" in str(messages[0]["content"])  # type: ignore[index]
    assert messages[1] == {"role": "user", "content": "А строк?"}  # type: ignore[index]
    assert trace.summary is not None and trace.summary.folded_turns == 1
    assert after.summary == "Фермер питав про пеню own_001: 0,1 % за день."
    assert [t.question for t in after.turns] == ["А строк?", "Ок?"]


def test_a_partial_answer_enters_the_dialogue_without_the_tool_results(
    profile: ProfileStore,
) -> None:
    model = FakeModel(turn("Шукаю.", tool_request(query="пеня")))  # never answers

    trace, dialogue = ask(make_agent(model, profile), make_memory(profile), Dialogue(), "Пеня?")

    assert trace.stop_reason == StopReason.MAX_STEPS
    [only] = dialogue.turns
    assert only.answer == PARTIAL_HEADERS[StopReason.MAX_STEPS]
    assert "знайдено" not in only.answer  # the raw tool text stays in the trace


# --- a question to the farmer


def test_the_dialogue_waits_for_the_farmers_answer(profile: ProfileStore) -> None:
    model = FakeModel(turn("Запам'ятаю.", REMEMBER_REGION), turn("Запам'ятав."))
    agent, memory = make_agent(model, profile), make_memory(profile)

    waiting, dialogue = ask(agent, memory, Dialogue(), "Я з Одещини, запам'ятай.")

    assert waiting.stop_reason == StopReason.AWAITING_CONFIRMATION
    assert dialogue.turns == ()  # no answer yet
    trace, dialogue = confirm(agent, dialogue, waiting, approved=True)
    assert trace.stop_reason == StopReason.ANSWERED
    assert [(t.question, t.answer) for t in dialogue.turns] == [
        ("Я з Одещини, запам'ятай.", "Запам'ятав.")
    ]


# --- across sessions: the profile


def test_a_fact_from_session_one_is_used_in_session_two(profile: ProfileStore) -> None:
    first = FakeModel(turn("Запам'ятаю.", REMEMBER_REGION), turn("Запам'ятав."))
    agent = make_agent(first, profile)
    waiting, dialogue = ask(agent, make_memory(profile), Dialogue(), "Я з Одещини, запам'ятай.")
    confirm(agent, dialogue, waiting, approved=True)

    second = FakeModel(turn("Для Одеської області найближчий порт — Одеса."))
    trace, _ = ask(make_agent(second, profile), make_memory(profile), Dialogue(), "Який порт?")

    system = system_of(second)
    assert "# Farmer profile" in system
    assert "- region: Одеська область (записано 26.11.2026)" in system
    assert second.requests[0]["messages"] == [{"role": "user", "content": "Який порт?"}]
    assert trace.stop_reason == StopReason.ANSWERED


def test_after_a_no_nothing_reaches_session_two(profile: ProfileStore) -> None:
    first = FakeModel(turn("Запам'ятаю.", REMEMBER_REGION), turn("Добре, не запам'ятовую."))
    agent = make_agent(first, profile)
    waiting, dialogue = ask(agent, make_memory(profile), Dialogue(), "Я з Одещини.")
    confirm(agent, dialogue, waiting, approved=False)

    second = FakeModel(turn("ok"))
    ask(make_agent(second, profile), make_memory(profile), Dialogue(), "Який порт?")

    assert system_of(second) == SYSTEM
    assert profile.facts(now=NOW) == ()


def test_a_fact_forgotten_in_session_two_is_gone_in_session_three(
    profile: ProfileStore,
) -> None:
    profile.remember(FactKey.REGION, "Одеська область", now=NOW)
    forget = tool_request("forget_fact", "f1", key="region")
    second = FakeModel(turn("Забуду.", forget), turn("Забув."))
    agent = make_agent(second, profile)
    waiting, dialogue = ask(agent, make_memory(profile), Dialogue(), "Забудь мій регіон.")
    assert "Одеська область" in system_of(second)  # it was there before
    confirm(agent, dialogue, waiting, approved=True)

    third = FakeModel(turn("ok"))
    ask(make_agent(third, profile), make_memory(profile), Dialogue(), "Який порт?")

    assert "Одеська область" not in system_of(third)


def test_a_stale_fact_is_not_in_the_context(profile: ProfileStore) -> None:
    profile.remember(FactKey.REGION, "Одеська область", now=NOW - timedelta(days=365))
    model = FakeModel(turn("ok"))

    ask(make_agent(model, profile), make_memory(profile), Dialogue(), "Який порт?")

    assert system_of(model) == SYSTEM


def test_a_profile_that_cannot_be_opened_does_not_stop_the_answer(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder = tmp_path / "profile.sqlite"
    folder.mkdir()  # a folder where the file should be: sqlite3 cannot open it
    profile = ProfileStore(folder, "farmer_a", TTL_DAYS)
    model = FakeModel(turn("Протягом 5 днів."))

    trace, _ = ask(make_agent(model, profile), make_memory(profile), Dialogue(), "Коли?")

    assert trace.stop_reason == StopReason.ANSWERED
    assert system_of(model) == SYSTEM
    assert "OperationalError" in caplog.text


def test_a_broken_profile_does_not_stop_the_answer(
    profile: ProfileStore, caplog: pytest.LogCaptureFixture
) -> None:
    profile.remember(FactKey.REGION, "Одеська область", now=NOW)
    connection = sqlite3.connect(profile.path)
    with connection:
        connection.execute("UPDATE facts SET key = 'favourite_colour'")
    connection.close()
    model = FakeModel(turn("Протягом 5 днів."))

    trace, _ = ask(make_agent(model, profile), make_memory(profile), Dialogue(), "Коли?")

    assert trace.stop_reason == StopReason.ANSWERED
    assert system_of(model) == SYSTEM  # answered without the profile
    assert "favourite_colour" in caplog.text
