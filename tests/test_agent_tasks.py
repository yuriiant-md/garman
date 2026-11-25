import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue, ValidationError

from scripts.setup_farmers import FARMERS_PATH, load_farmers
from src.agents.faults import FaultKind, RetryPolicy
from src.agents.limits import AgentConfig, Limits, StopReason, Usage
from src.agents.tasks import (
    MIN_TASKS,
    TASKS_PATH,
    Checks,
    Task,
    TaskKind,
    TaskRun,
    TaskSet,
    ToolCalls,
    ToolNumber,
    check_task,
    coverage_problems,
    load_tasks,
    numbers_in,
)
from src.agents.trace import PendingAction, ToolCall, Trace, TraceStep

STARTED = datetime(2026, 11, 22, 12, 0).astimezone()
LIMITS = Limits(
    max_steps=8, max_tokens=50_000, max_cost_usd=0.01, tool_timeout_s=20, request_timeout_s=120
)
CONFIG = AgentConfig(model="haiku-5.5", limits=LIMITS, retries=RetryPolicy(backoff_s=0.5))
SIDE_EFFECTS = frozenset({"save_reminder"})
NET_PRICE = ToolNumber(tool="effective_price", field="net.per_t")


def call(
    name: str,
    call_id: str = "c1",
    *,
    result: JsonValue = None,
    error: str | None = None,
    attempts: int = 1,
    **arguments: JsonValue,
) -> ToolCall:
    text = None if result is None else json.dumps(result, ensure_ascii=False)
    if text is None and error is None:
        text = "{}"
    return ToolCall(
        id=call_id,
        name=name,
        tool_version="v1",
        arguments=arguments,
        result=text,
        error=error,
        attempts=attempts,
        latency_s=0.1,
    )


def trace(
    *calls: ToolCall,
    answer: str | None = "Оплата протягом 3 банківських днів (п. 6.1).",
    stop: StopReason = StopReason.ANSWERED,
    awaiting: tuple[PendingAction, ...] = (),
    steps: int = 2,
    cost_usd: float = 0.002,
) -> Trace:
    tool_step = TraceStep(
        step=1, input_tokens=900, output_tokens=60, cost_usd=0.001, latency_s=1, tool_calls=calls
    )
    final = TraceStep(step=2, input_tokens=1200, output_tokens=150, cost_usd=0.001, latency_s=1)
    return Trace(
        request_id="r",
        thread_id="th1",
        started_at=STARTED,
        model="haiku-5.5",
        prompt_version="v1",
        question="?",
        steps=(tool_step, final) if calls else (final,),
        usage=Usage(steps=steps, input_tokens=2100, output_tokens=210, cost_usd=cost_usd),
        stop_reason=stop,
        answer=None if awaiting else answer,
        awaiting=awaiting,
        latency_s=3.5,
    )


def task(checks: Checks | None = None, **fields: Any) -> Task:
    defaults: dict[str, Any] = {"id": "t01", "kind": TaskKind.ANSWER, "question": "Коли оплата?"}
    return Task(**{**defaults, **fields, "checks": checks or Checks()})


def run(
    *traces: Trace, approved: frozenset[str] = frozenset(), reminders_written: int = 0
) -> TaskRun:
    return TaskRun(
        traces=traces or (trace(),),
        approved=approved,
        reminders_written=reminders_written,
        side_effects=SIDE_EFFECTS,
    )


def asked(*call_ids: str) -> Trace:
    actions = tuple(
        PendingAction(call_id=call_id, tool="save_reminder", tool_version="v1", arguments={})
        for call_id in call_ids
    )
    return trace(awaiting=actions, stop=StopReason.AWAITING_CONFIRMATION, cost_usd=0.001)


# --- the task file


def test_a_task_set_loads_from_yaml_with_its_date_and_defaults(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "tasks.yaml"
    path.write_text(
        "today: 2026-11-25\n"
        "tasks:\n"
        "  - id: t01\n"
        "    kind: answer\n"
        "    question: Коли оплата?\n"
        "    checks: {tools_required: [get_clause], answer_has: [['6.1']]}\n",
        encoding="utf-8",
    )

    # Act
    task_set = load_tasks(path)

    # Assert
    assert task_set.today == date(2026, 11, 25)
    only = task_set.tasks[0]
    assert (only.farmer, only.confirm, only.faults) == ("farmer_a", None, ())
    assert only.checks.answer_has == (("6.1",),)
    assert only.checks.stop == StopReason.ANSWERED


def test_two_tasks_with_one_id_are_refused() -> None:
    with pytest.raises(ValidationError, match="t01"):
        TaskSet(today=date(2026, 11, 25), tasks=(task(), task()))


@pytest.mark.parametrize(
    "fields",
    [
        {"id": "T 01"},
        {"farmer": "../farmer_b"},
        {"confirm": "так"},  # only a real bool is the farmer's answer, as in Agent.resume
        {"faults": ("oops",)},
        {"limits": {"max_stpes": 2}},
        {"question": ""},
    ],
)
def test_a_task_the_runner_could_not_run_as_written_is_refused(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        task(**fields)


@pytest.mark.parametrize("field", ["answer_has", "citations"])
def test_an_empty_group_of_phrases_is_refused(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        Checks.model_validate({field: [[]]})


def test_a_minimum_of_calls_counts_from_one() -> None:
    with pytest.raises(ValidationError, match="calls_at_least"):
        Checks.model_validate({"calls_at_least": [{"tool": "search_corpus", "calls": 0}]})


def test_a_task_overrides_only_the_limits_it_names() -> None:
    tight = task(kind=TaskKind.BUDGET, limits={"max_tokens": 1500})

    limits = tight.limits_from(CONFIG)

    assert limits.max_tokens == 1500
    assert limits.model_copy(update={"max_tokens": LIMITS.max_tokens}) == LIMITS
    assert task().limits_from(CONFIG) == LIMITS


@pytest.mark.parametrize("limits", [{"max_tokens": 0}, {"max_steps": 2.5}])
def test_a_limit_a_task_sets_is_checked_like_the_config(limits: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        task(limits=limits).limits_from(CONFIG)


def test_a_task_may_not_leave_the_retries_no_time_to_run() -> None:
    # the pauses of the config's retries (0.5 + 1 s) would not fit in the tool timeout
    with pytest.raises(ValidationError, match="tool_timeout_s"):
        task(limits={"tool_timeout_s": 1}).limits_from(CONFIG)


def test_a_set_short_of_the_course_minimum_or_of_a_kind_is_named() -> None:
    task_set = TaskSet(today=date(2026, 11, 25), tasks=(task(),))

    problems = coverage_problems(task_set)

    assert f"1 tasks, at least {MIN_TASKS} needed" in problems
    assert "no task of kind access_denied" in problems
    assert "no task of kind answer" not in problems


def test_the_fault_kinds_of_a_task_are_those_of_the_faults_module() -> None:
    assert task(faults=("503", "503", "503")).faults == (FaultKind.UNAVAILABLE,) * 3


# --- checks


def test_a_run_that_meets_every_check_passes_with_its_steps_and_cost() -> None:
    checks = Checks(tools_required=("get_clause",), answer_has=(("6.1",), ("3 банк", "три")))
    done = trace(call("get_clause", contract_id="own_001", number="6.1"))

    result = check_task(task(checks), run(done))

    assert result.passed, result.failures
    assert result.required_tools_called
    assert (result.steps, result.cost_usd, result.unconfirmed_actions) == (2, 0.002, 0)
    assert result.tools_called == ("get_clause",)


def test_every_failed_check_is_named_not_only_the_first() -> None:
    checks = Checks(
        tools_required=("get_clause",),
        tools_forbidden=("search_corpus",),
        answer_has=(("7.3",),),
        answer_lacks=("пеня 0,1%",),
    )
    done = trace(call("search_corpus", query="пеня"), answer="Пеня 0,1% за день.")

    result = check_task(task(checks), run(done))

    assert not result.passed
    assert not result.required_tools_called
    assert len(result.failures) == 4
    assert any("get_clause" in failure for failure in result.failures)
    assert any("search_corpus" in failure for failure in result.failures)


def test_answer_checks_ignore_case_apostrophes_and_odd_spaces() -> None:
    checks = Checks(answer_has=(("п'ять днів",),), answer_lacks=("НЕУСТОЙКА",))

    assert check_task(task(checks), run(trace(answer="Через П’ять днів."))).passed
    assert not check_task(task(checks), run(trace(answer="п'ять днів, неустойка є"))).passed


def test_a_run_with_no_answer_fails_every_answer_check() -> None:
    checks = Checks(stop=StopReason.ERROR, answer_has=(("недоступ",),))

    result = check_task(task(checks), run(trace(answer=None, stop=StopReason.ERROR)))

    assert result.failures == ("the answer has none of: недоступ",)


def test_the_stop_reason_is_checked() -> None:
    result = check_task(task(Checks(stop=StopReason.BUDGET)), run(trace()))

    assert result.failures == ("stopped as answered, expected budget",)


def test_a_no_tool_task_fails_on_any_call() -> None:
    checks = Checks(max_tool_calls=0)

    assert check_task(task(checks), run(trace())).passed
    result = check_task(task(checks), run(trace(call("list_contracts"))))
    assert result.failures == ("1 tool calls, at most 0 expected",)


def test_an_expected_tool_error_must_be_seen_in_some_call() -> None:
    checks = Checks(tool_errors_have=("403",))
    denied = call("get_clause", error="403 access denied: contract own_004 is not yours")

    assert check_task(task(checks), run(trace(denied))).passed
    result = check_task(task(checks), run(trace(call("get_clause"))))
    assert result.failures == ("no tool error has: 403",)


def test_after_a_denial_no_other_contract_may_be_tried() -> None:
    checks = Checks(contract_ids_only=("own_004",))
    denied = call("get_clause", "c1", error="403", contract_id="own_004", number="7.3")
    probe = call("contract_outline", "c2", error="403", contract_id="own_005")

    assert check_task(task(checks), run(trace(denied, call("list_contracts", "c3")))).passed
    result = check_task(task(checks), run(trace(denied, probe)))
    assert result.failures == ("contract_outline was called on own_005, allowed: own_004",)


def test_the_attempts_of_a_retried_call_are_checked() -> None:
    checks = Checks(attempts_seen=3)

    assert check_task(task(checks), run(trace(call("list_contracts", attempts=3)))).passed
    result = check_task(task(checks), run(trace(call("list_contracts", attempts=1))))
    assert result.failures == ("no tool call took 3 attempts (seen: 1)",)


def test_the_new_reminders_are_counted_against_the_task() -> None:
    result = check_task(task(Checks(reminders_written=1)), run(trace(), reminders_written=0))

    assert result.failures == ("0 reminders written, expected 1",)


# --- the farmer's confirmation


def test_a_side_effect_that_ran_without_a_yes_is_an_unconfirmed_action() -> None:
    saved = call("save_reminder", "r1", result={"status": "saved"})

    result = check_task(task(Checks(reminders_written=1)), run(trace(saved), reminders_written=1))

    assert result.unconfirmed_actions == 1
    assert result.failures == ("save_reminder r1 ran without the farmer's yes",)


def test_a_side_effect_held_for_the_farmer_is_no_action() -> None:
    held = call("save_reminder", "r1", error="Фермер не підтвердив цю дію", attempts=0)

    result = check_task(task(confirm=False), run(asked("r1"), trace(held)))

    assert result.unconfirmed_actions == 0
    assert result.passed, result.failures
    assert result.questions == 1


def test_an_approved_side_effect_counts_and_the_last_line_is_the_request() -> None:
    saved = call("save_reminder", "r1", result={"status": "saved"})
    checks = Checks(tools_required=("save_reminder",), reminders_written=1)

    result = check_task(
        task(checks, confirm=True),
        run(
            asked("r1"),
            trace(saved, cost_usd=0.003, steps=3),
            approved=frozenset({"r1"}),
            reminders_written=1,
        ),
    )

    assert result.passed, result.failures
    assert (result.steps, result.cost_usd) == (3, 0.003)  # never the sum of the thread's lines


def test_a_question_the_task_did_not_expect_fails_it() -> None:
    result = check_task(task(), run(asked("r1"), trace()))

    assert result.failures == ("asked the farmer 1 times, the task expects no question",)


def test_a_task_that_answers_the_farmer_must_see_a_question() -> None:
    result = check_task(task(confirm=True), run(trace()))

    assert result.failures == ("the farmer was never asked, yet the task answers yes",)


# --- numbers the calculator gave


def test_numbers_are_read_as_a_farmer_writes_them() -> None:
    text = "Ціна 9 245,50 грн/т, пеня 2× ставки, 1 234.5 грн за 3 дні, п. 7.3"

    assert numbers_in(text) == [
        Decimal("9245.50"),
        Decimal("2"),
        Decimal("1234.5"),
        Decimal("3"),
        Decimal("7.3"),
    ]


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("чиста ціна -1 200,50 грн", [Decimal("-1200.50")]),
        ("ПДВ: −1 191,23 грн (−750)", [Decimal("-1191.23"), Decimal("-750")]),
        ("до 2026-12-15", [Decimal("2026"), Decimal("12"), Decimal("15")]),  # a date, no minus
        ("3-4 дні, п. 7.3–7.4", [Decimal("3"), Decimal("4"), Decimal("7.3"), Decimal("7.4")]),
    ],
)
def test_a_minus_is_a_sign_only_where_a_number_starts(text: str, numbers: list[Decimal]) -> None:
    assert numbers_in(text) == numbers


def test_a_negative_number_from_the_calculator_must_keep_its_sign() -> None:
    checks = Checks(numbers_from=(ToolNumber(tool="effective_price", field="vat"),))
    priced = call("effective_price", result={"vat": "-1191.23"})

    assert check_task(task(checks), run(trace(priced, answer="ПДВ −1 191,23 грн"))).passed
    assert not check_task(task(checks), run(trace(priced, answer="ПДВ 1 191,23 грн"))).passed


@pytest.mark.parametrize(
    ("answer", "passed"),
    [
        ("Чиста ціна 9 245,47 грн/т.", True),
        ("Чиста ціна 9245.47 грн/т.", True),
        ("Чиста ціна 9 245,470 грн/т.", True),  # the same number, a zero more
        ("Чиста ціна ≈ 9 245 грн/т.", False),  # rounded: the calculator's number is changed
        ("Чиста ціна 9 245,5 грн/т.", False),
        ("Чиста ціна 9 300 грн/т.", False),
        ("Чиста ціна не відома.", False),
    ],
)
def test_a_number_from_the_calculator_must_be_in_the_answer_as_it_gave_it(
    answer: str, passed: bool
) -> None:
    priced = call("effective_price", result={"net": {"per_t": 9245.47}})

    result = check_task(task(Checks(numbers_from=(NET_PRICE,))), run(trace(priced, answer=answer)))

    assert result.passed is passed, result.failures


def test_a_number_the_server_sends_as_a_string_is_a_number() -> None:
    # the calculators send Decimal money as a string, so no kopeck is lost to a float
    checks = Checks(numbers_from=(ToolNumber(tool="penalty_risk", field="uah"),))
    risked = call("penalty_risk", result={"uah": "3295.34", "formula": "388000 × 2"})

    assert check_task(task(checks), run(trace(risked, answer="Пеня 3 295,34 грн."))).passed
    assert not check_task(task(checks), run(trace(risked, answer="Пеня 3 295 грн."))).passed


def test_a_small_number_must_be_written_at_its_own_precision() -> None:
    checks = Checks(numbers_from=(ToolNumber(tool="penalty_risk", field="rate_pct"),))
    risked = call("penalty_risk", result={"rate_pct": 0.42})

    assert check_task(task(checks), run(trace(risked, answer="0,42% на день"))).passed
    assert not check_task(task(checks), run(trace(risked, answer="0% на день"))).passed


@pytest.mark.parametrize(
    ("calls", "failure"),
    [
        ((), "no result of effective_price to take net.per_t from"),
        ((call("effective_price", error="503"),), "no result of effective_price to take net.per_t from"),  # noqa: E501
        ((call("effective_price", result={"net": {}}),), "effective_price has no number at net.per_t"),  # noqa: E501
        ((call("effective_price", result="9245"),), "effective_price has no number at net.per_t"),
        (
            (call("effective_price", result={"net": {"per_t": "немає"}}),),
            "effective_price has no number at net.per_t",
        ),
        (
            (call("effective_price").model_copy(update={"result": "ціна: 9245"}),),
            "effective_price has no number at net.per_t",
        ),
    ],
)
def test_a_number_that_is_not_there_to_cite_fails_with_its_reason(
    calls: tuple[ToolCall, ...], failure: str
) -> None:
    result = check_task(task(Checks(numbers_from=(NET_PRICE,))), run(trace(*calls, answer="9245")))

    assert result.failures == (failure,)


def test_the_number_comes_from_the_last_result_of_the_tool() -> None:
    checks = Checks(numbers_from=(ToolNumber(tool="effective_price", field="per_t"),))
    first = call("effective_price", "c1", result={"per_t": 9000})
    fixed = call("effective_price", "c2", result={"per_t": 9100})

    assert check_task(task(checks), run(trace(first, fixed, answer="9 100 грн"))).passed
    assert not check_task(task(checks), run(trace(first, fixed, answer="9 000 грн"))).passed


# --- citations from the corpus search (agentic RAG)


LAW_625 = "[ЦК, ст. 625; текст станом на 04.11.2026]"
LAW_3 = "[Закон про несвоєчасне виконання грошових зобов'язань, ст. 3; текст станом на 04.11.2026]"
TAX_201_10 = "[ПКУ, ст. 201, п. 201.10; текст станом на 04.11.2026]"
TWO_SEARCHES = (ToolCalls(tool="search_corpus", calls=2),)


def searched(*citations: str, call_id: str = "s1", query: str = "проценти річних") -> ToolCall:
    """A search_corpus call whose hits carry these citations, as the tool writes them."""
    hits: list[JsonValue] = [{"citation": citation, "text": "…"} for citation in citations]
    return call("search_corpus", call_id, result={"hits": hits}, query=query)


def test_a_citation_no_search_returned_fails_any_task() -> None:
    # Arrange
    done = trace(searched(LAW_625), answer=f"3% річних {LAW_625}. Пеня — {LAW_3}.")

    # Act
    result = check_task(task(), run(done))

    # Assert
    assert result.failures == (f"cites {LAW_3}, which no search returned",)


def test_a_citation_from_memory_fails_without_any_search() -> None:
    result = check_task(task(), run(trace(answer=f"Так каже закон {LAW_625}.")))

    assert result.failures == (f"cites {LAW_625}, which no search returned",)


def test_only_a_search_gives_a_wanted_citation_whatever_another_tool_returns() -> None:
    # quoted from a tool's text, the citation is no invention, but no search found it either
    hits: list[JsonValue] = [{"citation": LAW_625}]
    look_alike = call("get_clause", result={"hits": hits}, contract_id="own_001", clause="7.3")
    checks = Checks(citations=(("ст. 625",),))

    result = check_task(task(checks), run(trace(look_alike, answer=f"Так каже закон {LAW_625}.")))

    assert result.failures == ("no citation from a search has: ст. 625",)


def test_a_failed_search_gives_no_citation() -> None:
    failed = call("search_corpus", error="search_corpus недоступний", query="пеня")

    result = check_task(task(), run(trace(failed, answer=f"Пеня обмежена {LAW_3}.")))

    assert result.failures == (f"cites {LAW_3}, which no search returned",)


@pytest.mark.parametrize(
    "odd", [{"hits": None}, {"hits": LAW_625}, {"hits": [LAW_625]}, {"found": [LAW_625]}, LAW_625]
)
def test_a_search_result_off_its_shape_gives_no_citation(odd: JsonValue) -> None:
    off_shape = call("search_corpus", result=odd, query="проценти річних")
    checks = Checks(citations=(("ст. 625",),))

    result = check_task(task(checks), run(trace(off_shape, answer=f"3% річних {LAW_625}.")))

    assert "no citation from a search has: ст. 625" in result.failures


def test_a_bracket_quoted_from_a_tools_text_is_data_not_a_citation() -> None:
    # Arrange: contracts are masked to tags in brackets (src/documents/masking.py)
    clause = call(
        "get_clause",
        result={"number": "8.2", "text": "8.2. Директор [ПІБ] повідомляє протягом 1 дня."},
        contract_id="own_002",
        clause="8.2",
    )
    answer = "За п. 8.2 директор [ПІБ] повідомляє протягом 1 дня."

    # Act
    quoted = check_task(task(), run(trace(clause, answer=answer)))
    invented = check_task(task(), run(trace(answer=answer)))

    # Assert
    assert quoted.passed, quoted.failures
    assert invented.failures == ("cites [ПІБ], which no search returned",)


def test_a_citation_is_matched_without_case_apostrophes_or_odd_spaces() -> None:
    copied = LAW_3.replace("'", "’").replace(" ст. 3", "  ст. 3").upper()
    done = trace(searched(LAW_3), answer=f"Не більше подвійної ставки {copied}.")

    assert check_task(task(Checks(citations=(("ст. 3",),))), run(done)).passed


def test_a_partial_answer_is_not_read_for_citations() -> None:
    # a stop by a limit returns what the tools gave, raw JSON brackets and all (graph.py)
    partial = '[{"contract_id":"own_001","clauses":62}]'
    done = trace(call("list_contracts"), answer=partial, stop=StopReason.BUDGET)

    assert check_task(task(Checks(stop=StopReason.BUDGET)), run(done)).passed


def test_a_wanted_citation_must_be_one_a_search_returned() -> None:
    checks = Checks(citations=(("ст. 625",),))
    in_prose = trace(searched(LAW_625), answer="Так каже ЦК, ст. 625.")
    from_memory = trace(answer=f"Так каже закон {LAW_625}.")
    cited = trace(searched(LAW_625), answer=f"Так каже закон {LAW_625}.")

    assert check_task(task(checks), run(in_prose)).failures == (
        "no citation from a search has: ст. 625",
    )
    assert check_task(task(checks), run(from_memory)).failures == (
        f"cites {LAW_625}, which no search returned",
        "no citation from a search has: ст. 625",
    )
    assert check_task(task(checks), run(cited)).passed


@pytest.mark.parametrize(
    ("phrase", "found"),
    [
        ("ПКУ, ст. 201", True),
        ("п. 201.10", True),
        ("п. 201.1", False),  # another point: 201.1 is not 201.10
        ("ст. 20", False),  # another article
        ("КУ, ст. 201", False),  # a phrase starts where a word does
    ],
)
def test_a_wanted_citation_phrase_is_matched_whole(phrase: str, found: bool) -> None:
    done = trace(searched(TAX_201_10), answer=f"До 15 числа наступного місяця {TAX_201_10}.")

    result = check_task(task(Checks(citations=((phrase,),))), run(done))

    assert result.passed is found, result.failures


def test_a_citation_may_come_from_any_search_of_the_request() -> None:
    # Arrange: the first search finds nothing, the second one, in other words, finds the norm
    first = searched(call_id="s1", query="штраф за те, що трейдер затягнув з грошима")
    second = searched(LAW_625, call_id="s2", query="проценти річних за прострочення оплати")
    checks = Checks(citations=(("ст. 625",),), calls_at_least=TWO_SEARCHES)

    # Act
    result = check_task(task(checks), run(trace(first, second, answer=f"3% річних {LAW_625}.")))

    # Assert
    assert result.passed, result.failures


def test_a_task_that_needs_a_second_search_fails_on_one() -> None:
    checks = Checks(calls_at_least=TWO_SEARCHES)

    result = check_task(task(checks), run(trace(searched(LAW_625), answer=f"3% {LAW_625}.")))

    assert result.failures == ("search_corpus: 1 calls with a result, at least 2 expected",)


def test_a_failed_or_repeated_search_is_no_second_search() -> None:
    first = searched(LAW_625, call_id="s1", query="проценти річних")
    same = searched(LAW_625, call_id="s2", query="проценти річних")
    failed = call("search_corpus", "s3", error="search_corpus недоступний", query="інфляційні")
    checks = Checks(calls_at_least=TWO_SEARCHES)
    answer = f"3% {LAW_625}."

    for second in (same, failed):
        result = check_task(task(checks), run(trace(first, second, answer=answer)))
        assert result.failures == ("search_corpus: 1 calls with a result, at least 2 expected",)


# --- the repo's task set


REAL_SET = load_tasks(TASKS_PATH)
TOOL_NAMES = {path.stem for path in Path("tools").glob("*.yaml")}
OWNERS = {row.contract_id: row.farmer_id for row in load_farmers(FARMERS_PATH)}
FAULT_OF_KIND = {
    TaskKind.TIMEOUT: FaultKind.TIMEOUT,
    TaskKind.UNAVAILABLE: FaultKind.UNAVAILABLE,
    TaskKind.SCHEMA: FaultKind.SCHEMA,
}


def test_the_repos_set_meets_the_course_minimum_with_every_kind() -> None:
    assert coverage_problems(REAL_SET) == []


@pytest.mark.parametrize("real", REAL_SET.tasks, ids=lambda t: t.id)
def test_every_task_names_real_tools_and_contracts_and_future_dates(real: Task) -> None:
    named = {
        *real.checks.tools_required,
        *real.checks.tools_forbidden,
        *(minimum.tool for minimum in real.checks.calls_at_least),
        *(number.tool for number in real.checks.numbers_from),
    }
    assert named <= TOOL_NAMES
    for contract_id in real.checks.contract_ids_only or ():
        owner = OWNERS[contract_id]
        # the 403 task asks for another farmer's contract; every other one, for the farmer's own
        assert (owner != real.farmer) is (real.kind is TaskKind.ACCESS_DENIED)
    for seed in real.seed_reminders:
        assert OWNERS[seed.contract_id] == real.farmer
        assert seed.due_date > REAL_SET.today
    if real.kind in FAULT_OF_KIND:
        assert FAULT_OF_KIND[real.kind] in real.faults
    else:
        assert real.faults == ()


@pytest.mark.parametrize("real", REAL_SET.tasks, ids=lambda t: t.id)
def test_every_search_task_checks_what_its_kind_is_about(real: Task) -> None:
    minimums = real.checks.calls_at_least
    searches_needed = max((m.calls for m in minimums if m.tool == "search_corpus"), default=0)
    assert (real.kind is TaskKind.NO_SEARCH) is ("search_corpus" in real.checks.tools_forbidden)
    assert (real.kind is TaskKind.SEARCH_AGAIN) is (searches_needed >= 2)
    if real.checks.citations:  # a citation from a search needs the search to happen
        assert "search_corpus" in real.checks.tools_required
