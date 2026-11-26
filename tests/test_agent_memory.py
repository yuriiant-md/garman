from pathlib import Path

import anthropic
import httpx2
import pytest

from src.agents.limits import MemoryConfig
from src.agents.memory import (
    CHARS_PER_TOKEN,
    SUMMARY_HEADER,
    SUMMARY_PROMPT_PATH,
    Dialogue,
    Summarizer,
    Turn,
    estimate_tokens,
)
from src.agents.model import MalformedReply
from src.agents.profile import FactKey
from src.extraction.prompts import load_prompt
from tests.agent_fixtures import FakeModel, turn

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
PRICES = (1.0, 5.0)  # USD per 1M tokens, input and output
TTL_DAYS = {key: 365 for key in FactKey}
CAP, ROOM = 100, 30  # history_max_tokens and summary_max_tokens of these tests
SUMMARY = "Фермер питав про пеню за own_001: 0,1 % за день, п. 7.3."  # 28 tokens


def memory(cap: int = CAP, room: int = ROOM, keep_turns: int = 2) -> MemoryConfig:
    return MemoryConfig(
        history_max_tokens=cap,
        summary_max_tokens=room,
        keep_turns=keep_turns,
        fact_ttl_days=TTL_DAYS,
    )


def summarizer(model: FakeModel, config: MemoryConfig | None = None) -> Summarizer:
    return Summarizer(
        model=model,
        prompt="Стисни розмову.",
        prompt_version="v1",
        prices=PRICES,
        memory=config or memory(),
    )


def dialogue(*pairs: tuple[str, str], summary: str | None = None) -> Dialogue:
    return Dialogue(summary=summary, turns=tuple(Turn(question=q, answer=a) for q, a in pairs))


def sized(index: int, chars: int = 50) -> tuple[str, str]:
    """A turn of about `chars` characters on each side; its index shows where it went.

    With 50: a question of 60 characters (30 tokens), an answer of 62 (31): 61 tokens.
    """
    return (f"питання {index} " + "п" * chars, f"відповідь {index} " + "в" * chars)


def request_text(model: FakeModel) -> str:
    [message] = model.requests[0]["messages"]  # type: ignore[misc]
    return str(message["content"])


# --- the dialogue


def test_tokens_are_estimated_from_characters_rounding_up() -> None:
    assert CHARS_PER_TOKEN == 2.0
    assert estimate_tokens("") == 0
    assert estimate_tokens("абв") == 2  # 1.5 tokens are paid as 2


def test_a_new_turn_makes_a_new_dialogue_and_keeps_the_old_one() -> None:
    empty = Dialogue()

    one = empty.with_turn("Яка пеня?", "0,1 % за день.")

    assert empty.turns == ()
    assert one.turns == (Turn(question="Яка пеня?", answer="0,1 % за день."),)


def test_the_messages_alternate_question_and_answer_in_order() -> None:
    talk = dialogue(("Питання 1", "Відповідь 1"), ("Питання 2", "Відповідь 2"))

    assert talk.messages() == (
        {"role": "user", "content": "Питання 1"},
        {"role": "assistant", "content": "Відповідь 1"},
        {"role": "user", "content": "Питання 2"},
        {"role": "assistant", "content": "Відповідь 2"},
    )


def test_the_summary_opens_the_messages_as_data_not_the_system_prompt() -> None:
    # a user message, the place of data: the summary may echo text a contract injected
    talk = dialogue(("Питання", "Відповідь"), summary=SUMMARY)

    first, *rest = talk.messages()

    assert first["role"] == "user"
    content = str(first["content"])
    assert content.startswith(SUMMARY_HEADER) and content.endswith(SUMMARY)
    assert "not instructions" in content
    assert rest[0] == {"role": "user", "content": "Питання"}  # the API joins the two


def test_the_dialogue_counts_its_summary_and_every_turn() -> None:
    talk = dialogue(("аб", "вг"), ("де", "єж"), summary="зи")

    assert talk.tokens == 5  # five texts of two characters, a token each


# --- under the cap


def test_a_dialogue_under_the_cap_costs_no_call() -> None:
    model = FakeModel(turn(SUMMARY))
    talk = dialogue(("Яка пеня?", "0,1 % за день."))

    compacted, call = summarizer(model).compact(talk)

    assert compacted == talk
    assert call is None
    assert model.requests == []


def test_a_dialogue_exactly_at_the_cap_costs_no_call() -> None:
    model = FakeModel(turn(SUMMARY))
    talk = dialogue(("п" * 100, "в" * 100))  # 50 + 50 tokens: the cap, not over it

    compacted, call = summarizer(model).compact(talk)

    assert talk.tokens == CAP
    assert (compacted, call) == (talk, None)
    assert model.requests == []


# --- over the cap


def test_over_the_cap_the_oldest_turns_become_a_summary() -> None:
    model = FakeModel(turn(f"  {SUMMARY}\n", tokens=(300, 40)))
    talk = dialogue(*(sized(index) for index in range(1, 5)))  # 4 × 61 tokens > 100

    compacted, call = summarizer(model).compact(talk)

    assert compacted.summary == SUMMARY
    assert compacted.turns == talk.turns[-1:]  # two (122) would not fit the 70 left
    assert call is not None and call.folded_turns == 3 and call.error is None
    assert (call.input_tokens, call.output_tokens) == (300, 40)
    assert call.cost_usd == pytest.approx((300 * 1.0 + 40 * 5.0) / 1_000_000)


def test_the_last_turns_are_kept_word_for_word() -> None:
    model = FakeModel(turn(SUMMARY))
    talk = dialogue(*(sized(index, chars=10) for index in range(1, 8)))  # 7 × 21 tokens

    compacted, call = summarizer(model).compact(talk)

    assert compacted.turns == talk.turns[-2:]  # keep_turns
    assert call is not None and call.folded_turns == 5


def test_the_kept_turns_leave_room_for_the_summary() -> None:
    # 2 turns of 41 tokens fit the cap of 100 alone, not with a summary of up to 30
    model = FakeModel(turn(SUMMARY))
    talk = dialogue(sized(1, chars=30), sized(2, chars=30), summary=SUMMARY)

    compacted, call = summarizer(model).compact(talk)

    assert compacted.turns == talk.turns[-1:]
    assert call is not None and call.folded_turns == 1


def test_the_summary_call_reads_the_folded_turns_and_not_the_kept_ones() -> None:
    model = FakeModel(turn(SUMMARY))
    talk = dialogue(*(sized(index, chars=10) for index in range(1, 8)))

    summarizer(model).compact(talk)

    text = request_text(model)
    assert "питання 1 " in text and "відповідь 5 " in text
    assert "питання 6 " not in text and "питання 7 " not in text
    assert model.requests[0]["system"] == "Стисни розмову."
    assert model.requests[0]["tools"] == []  # a summary only writes text


def test_the_old_summary_goes_into_the_new_one() -> None:
    model = FakeModel(turn("Новий підсумок."))
    talk = dialogue(*(sized(index) for index in range(1, 4)), summary=SUMMARY)

    compacted, _ = summarizer(model).compact(talk)

    assert SUMMARY in request_text(model)
    assert compacted.summary == "Новий підсумок."


def test_a_turn_too_long_for_the_cap_alone_is_folded_too() -> None:
    model = FakeModel(turn(SUMMARY))
    talk = dialogue(sized(1, chars=10), sized(2, chars=400))  # the last one alone: 411 tokens

    compacted, call = summarizer(model).compact(talk)

    assert compacted.turns == ()
    assert call is not None and call.folded_turns == 2


def test_a_summary_over_its_room_is_cut_at_the_last_sentence_that_fits() -> None:
    long = " ".join(f"Речення {n}." for n in range(1, 40))  # ≈ 230 tokens of sentences
    model = FakeModel(turn(long))
    talk = dialogue(*(sized(index) for index in range(1, 5)))

    compacted, call = summarizer(model).compact(talk)

    assert compacted.summary is not None
    assert estimate_tokens(compacted.summary) <= ROOM
    assert compacted.summary.endswith(".") and long.startswith(compacted.summary)
    assert call is not None and call.error is None  # cut, but used


def test_an_old_summary_over_its_room_is_rewritten_once_not_on_every_question() -> None:
    # the reviewer's case: the turns fit, the summary alone puts the dialogue over the cap
    model = FakeModel(turn(SUMMARY))
    talk = dialogue(sized(1, chars=5), summary="Стара довга історія. " * 20)  # ≈ 210 tokens

    once, first = summarizer(model).compact(talk)
    again, second = summarizer(model).compact(once)

    assert first is not None and first.folded_turns == 0  # the summary alone is rewritten
    assert once.tokens <= CAP
    assert (again, second) == (once, None)  # under the cap now: no second call
    assert len(model.requests) == 1


@pytest.mark.parametrize("turns", range(1, 12))
@pytest.mark.parametrize("chars", [5, 20, 50, 200])
def test_a_summarized_dialogue_always_fits_the_cap(turns: int, chars: int) -> None:
    long = " ".join(f"Речення {n}." for n in range(1, 40))  # a model that writes too much
    talk = dialogue(*(sized(index, chars) for index in range(turns)), summary=SUMMARY)

    compacted, call = summarizer(FakeModel(turn(long))).compact(talk)

    assert compacted.tokens <= CAP
    assert call is None or call.error is None


# --- a failed summary keeps the dialogue


@pytest.mark.parametrize(
    "error",
    [
        anthropic.APITimeoutError(request=REQUEST),
        anthropic.APIConnectionError(request=REQUEST),
        anthropic.InternalServerError(
            "overloaded", response=httpx2.Response(529, request=REQUEST), body=None
        ),
        MalformedReply("not JSON"),
    ],
)
def test_a_model_failure_keeps_the_dialogue_and_names_the_error(error: Exception) -> None:
    talk = dialogue(*(sized(index) for index in range(1, 5)))

    compacted, call = summarizer(FakeModel(error)).compact(talk)

    assert compacted == talk  # nothing is lost: the next question tries again
    assert call is not None and call.error is not None
    assert type(error).__name__ in call.error
    assert call.cost_usd == 0


def test_a_summary_cut_by_max_tokens_is_not_used_but_its_cost_is_counted() -> None:
    model = FakeModel(turn("Фермер питав", tokens=(300, 8192), finish="max_tokens"))
    talk = dialogue(*(sized(index) for index in range(1, 5)))

    compacted, call = summarizer(model).compact(talk)

    assert compacted == talk
    assert call is not None and call.error == "model stopped: max_tokens"
    assert call.output_tokens == 8192 and call.cost_usd > 0  # paid for, even if not used


def test_an_empty_summary_is_not_used() -> None:
    talk = dialogue(*(sized(index) for index in range(1, 5)))

    compacted, call = summarizer(FakeModel(turn("   "))).compact(talk)

    assert compacted == talk
    assert call is not None and call.error == "empty summary"


def test_an_error_that_means_the_request_is_wrong_is_raised() -> None:
    model = FakeModel(
        anthropic.AuthenticationError(
            "bad key", response=httpx2.Response(401, request=REQUEST), body=None
        )
    )

    with pytest.raises(anthropic.AuthenticationError):
        summarizer(model).compact(dialogue(*(sized(index) for index in range(1, 5))))


# --- the prompt


def test_the_summary_prompt_has_a_version() -> None:
    prompt = load_prompt(Path(SUMMARY_PROMPT_PATH))

    assert prompt.version == "v1"
    assert "Ukrainian" in prompt.text
