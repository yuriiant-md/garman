from pathlib import Path

import pytest

from src.rag.answer import Answer, AnswerStatus, Claim
from src.rag.answer_scoring import (
    EVAL_CONFIG_PATH,
    Ratio,
    Scorecard,
    Thresholds,
    check_thresholds,
    claims_text,
    forbidden_facts_present,
    load_thresholds,
    missing_key_facts,
    repealed_problems,
    score_run,
)
from src.rag.answerer import Outcome, RagResult
from src.rag.chunking import Chunk
from src.rag.sources import DocType
from tests.test_answer_eval import question, result

REPEALED_ID = "law_commercial_code#006"
THRESHOLDS = Thresholds(
    source_min=0.85,
    key_facts_min=0.80,
    not_in_documents_min=1.0,
    forbidden_facts_max=0,
    repealed_as_in_force_max=0,
)


def chunk(chunk_id: str = "own_003#030", **fields: object) -> Chunk:
    values: dict[str, object] = {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split("#")[0],
        "doc_type": DocType.CONTRACT,
        "title": "Договір",
        "short_title": "Договір",
        "text_sha256": "abc",
        "ref": "п. 7.2",
        "text": "текст",
    }
    return Chunk.model_validate(values | fields)


def repealed_chunk() -> Chunk:
    return chunk(REPEALED_ID, doc_type=DocType.LAW, in_force=False, short_title="ГК", ref="ст. 232")


def answered(*claims: str, source: str = "S1", **fields: object) -> RagResult:
    answer = Answer(
        status=AnswerStatus.ANSWERED,
        claims=tuple(Claim(text=text, sources=(source,)) for text in claims),
    )
    return result(answer=answer, text="\n".join(claims), **fields)


def repealed_result(claim: str, *, warned: bool = True) -> RagResult:
    warning_line = "\n⚠ [ГК, ст. 232]: ГК втратив чинність: це не чинна норма, а стара редакція."
    answer = Answer(
        status=AnswerStatus.ANSWERED, claims=(Claim(text=claim, sources=("S1",)),)
    )
    return result(
        answer=answer,
        text=f"{claim} [ГК, ст. 232]{warning_line if warned else ''}",
        sources={"S1": REPEALED_ID},
        cited=(REPEALED_ID,),
    )


def card(**fields: object) -> Scorecard:
    values: dict[str, object] = {
        "questions": (),
        "outcome": Ratio(20, 26),
        "source": Ratio(17, 20),
        "key_facts": Ratio(8, 10),
        "not_in_documents": Ratio(5, 5),
        "forbidden_facts": 0,
        "repealed_as_in_force": 0,
    }
    return Scorecard(**(values | fields))  # type: ignore[arg-type]


def gate_passed(scorecard: Scorecard, name: str) -> bool:
    return {gate.name: gate.passed for gate in check_thresholds(scorecard, THRESHOLDS)}[name]


def test_claims_text_holds_the_claims_without_citations_or_warnings() -> None:
    scored = result(
        answer=Answer(
            status=AnswerStatus.ANSWERED,
            claims=(Claim(text="Штраф 5%.", sources=("S1",)), Claim(text="Пеня.", sources=("S1",))),
        ),
        text="Штраф 5%. [Договір, п. 60]\nПеня. [Договір, п. 61]\n\n⚠ попередження 28.08.2025",
    )

    assert claims_text(scored) == "Штраф 5%. Пеня."


def test_claims_text_is_empty_when_there_is_no_answer() -> None:
    assert claims_text(result(outcome=Outcome.FAILED)) == ""


def test_a_number_in_a_citation_does_not_count_as_a_key_fact() -> None:
    scored = result(
        answer=Answer(
            status=AnswerStatus.ANSWERED, claims=(Claim(text="Можна відмовитись.", sources=("S1",)),)
        ),
        text="Можна відмовитись. [Договір, п. 60]",
    )

    assert missing_key_facts(question(key_facts=("60|шістдесят",)), scored) == ("60|шістдесят",)


def test_only_the_key_facts_the_claims_lack_are_missing() -> None:
    expected = question(key_facts=("30%", "подвійн"))

    assert missing_key_facts(expected, answered("Штраф 30% вартості.")) == ("подвійн",)
    assert missing_key_facts(expected, answered("Штраф 30%, подвійна ставка.")) == ()


def test_without_an_answer_every_key_fact_is_missing() -> None:
    expected = question(key_facts=("86%", "3 банківськ"))

    assert missing_key_facts(expected, result(outcome=Outcome.NOT_IN_DOCUMENTS)) == expected.key_facts


def test_a_forbidden_fact_is_reported_only_when_the_claims_hold_it() -> None:
    trap = question(forbidden_facts=("30%",))

    assert forbidden_facts_present(trap, answered("Штраф 30% вартості.")) == ("30%",)
    assert forbidden_facts_present(trap, answered("Штраф 5% вартості.")) == ()


def test_a_cited_repealed_chunk_needs_its_warning() -> None:
    chunks = {REPEALED_ID: repealed_chunk()}

    assert repealed_problems(repealed_result("ГК передбачав строк."), chunks) == ()
    problems = repealed_problems(repealed_result("ГК передбачав строк.", warned=False), chunks)
    assert len(problems) == 1
    assert REPEALED_ID in problems[0]


@pytest.mark.parametrize(
    ("claim", "is_problem"),
    [
        ("Кодекс, що вже не діє, передбачав строк.", False),
        ("Нечинний ГК передбачав строк.", False),
        ("Це не чинна норма.", False),
        ("ГК не є чинним.", False),
        ("Кодекс втратив чинність 28.08.2025.", False),
        ("Кодекс припинив чинність.", False),
        ("Чинність кодекс втратив.", False),
        ("Чинний ГК встановлює строк.", True),
        ("За чинним Господарським кодексом строк шість місяців.", True),
    ],
)
def test_a_claim_on_a_repealed_law_must_not_call_it_in_force(claim: str, is_problem: bool) -> None:
    problems = repealed_problems(repealed_result(claim), {REPEALED_ID: repealed_chunk()})

    assert bool(problems) is is_problem


def test_a_claim_on_a_law_in_force_may_say_it_is_in_force() -> None:
    law = chunk("law_civil_code#007", doc_type=DocType.LAW, in_force=True, short_title="ЦК")
    scored = answered(
        "Чинний ЦК визначає пеню.",
        sources={"S1": law.chunk_id},
        cited=(law.chunk_id,),
    )

    assert repealed_problems(scored, {law.chunk_id: law}) == ()


def test_a_cited_chunk_missing_from_the_corpus_stops_the_scoring() -> None:
    with pytest.raises(ValueError, match=REPEALED_ID):
        repealed_problems(repealed_result("ГК передбачав строк."), {})


def test_the_run_counts_each_metric_over_its_own_questions() -> None:
    chunks = {"own_003#030": chunk()}
    pairs = [
        (
            question(id="q01", should_cite=("own_003#030",), key_facts=("30%", "подвійн")),
            answered("Штраф 30% вартості.", cited=("own_003#030",)),
        ),
        (
            question(id="q02", should_cite=("own_003#030",), key_facts=("86%",)),
            result(outcome=Outcome.NOT_IN_DOCUMENTS),
        ),
        (
            question(id="t01", kind="off_corpus", expect=AnswerStatus.NOT_IN_DOCUMENTS),
            result(outcome=Outcome.NOT_IN_DOCUMENTS),
        ),
        (
            question(id="t02", kind="off_corpus", expect=AnswerStatus.NOT_IN_DOCUMENTS),
            answered("Придумана відповідь."),
        ),
        (
            question(id="s01", kind="similar_contract", forbidden_facts=("30%",)),
            answered("Штраф 30% вартості."),
        ),
    ]

    scorecard = score_run(pairs, chunks)

    assert scorecard.outcome == Ratio(3, 5)
    assert scorecard.source == Ratio(1, 2)
    assert scorecard.key_facts == Ratio(1, 3)
    assert scorecard.not_in_documents == Ratio(1, 2)
    assert scorecard.forbidden_facts == 1
    assert [score.id for score in scorecard.questions] == ["q01", "q02", "t01", "t02", "s01"]
    assert scorecard.questions[0].missing_key_facts == ("подвійн",)
    assert scorecard.questions[0].cites_evidence is True
    assert scorecard.questions[1].outcome_ok is False


def test_the_values_exactly_at_the_thresholds_pass() -> None:
    gates = check_thresholds(card(), THRESHOLDS)

    assert [gate.name for gate in gates] == [
        "source",
        "key_facts",
        "not_in_documents",
        "forbidden_facts",
        "repealed_as_in_force",
    ]
    assert all(gate.passed for gate in gates)


@pytest.mark.parametrize(
    ("name", "fields"),
    [
        ("source", {"source": Ratio(16, 20)}),
        ("key_facts", {"key_facts": Ratio(7, 10)}),
        ("not_in_documents", {"not_in_documents": Ratio(4, 5)}),
        ("forbidden_facts", {"forbidden_facts": 1}),
        ("repealed_as_in_force", {"repealed_as_in_force": 1}),
    ],
)
def test_one_value_past_its_threshold_fails_its_gate(name: str, fields: dict[str, object]) -> None:
    assert gate_passed(card(**fields), name) is False


def test_nothing_measured_is_not_a_pass() -> None:
    assert Ratio(0, 0).rate == 0.0
    assert gate_passed(card(source=Ratio(0, 0)), "source") is False


def test_the_repository_thresholds_are_the_plan_targets() -> None:
    assert load_thresholds(EVAL_CONFIG_PATH) == THRESHOLDS


VALID_THRESHOLDS = (
    "source_min: 0.85\nkey_facts_min: 0.8\nnot_in_documents_min: 1.0\n"
    "forbidden_facts_max: 0\nrepealed_as_in_force_max: 0\n"
)


@pytest.mark.parametrize(
    "text",
    [
        VALID_THRESHOLDS.replace("0.85", "1.5"),
        "source_min: 0.85\n",
        f"{VALID_THRESHOLDS}extra: 1\n",
    ],
)
def test_a_broken_thresholds_file_is_refused_with_its_name(tmp_path: Path, text: str) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="eval.yaml"):
        load_thresholds(path)
