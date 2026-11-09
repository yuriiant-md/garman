from pathlib import Path

import pytest

from src.rag.answer import AnswerStatus
from src.rag.answer_eval import (
    ANSWER_QUESTIONS_PATH,
    AnswerKind,
    AnswerQuestion,
    cites_expected,
    load_answer_questions,
    outcome_matches,
    summarize,
)
from src.rag.answerer import Outcome, RagResult
from src.rag.search_eval import load_questions

SEARCH_QUESTIONS = """\
version: 1
questions:
  - id: q01
    question: Яку пеню платить покупець, якщо затримає оплату?
    kind: plain
    filter: {contract_id: own_003, doc_types: [contract]}
    doc_id: own_003
    evidence: облікової ставки НБУ
    chunk_ids: [own_003#030]
    note: п. 7.2
  - id: q09
    question: Скільки процентів річних платить боржник?
    kind: exact
    filter: {doc_types: [law]}
    doc_id: law_civil_code
    evidence: три проценти річних
    chunk_ids: [law_civil_code#026]
"""
ANSWER_QUESTIONS = """\
version: 1
search_questions: [q01, q09]
questions:
  - id: t01
    kind: off_corpus
    question: Яку ціну дизпалива встановлює договір?
    contract_id: own_003
    expect: not_in_documents
"""


def write(tmp_path: Path, answers: str = ANSWER_QUESTIONS) -> tuple[Path, Path]:
    search_path = tmp_path / "search_questions.yaml"
    answer_path = tmp_path / "answer_questions.yaml"
    search_path.write_text(SEARCH_QUESTIONS, encoding="utf-8")
    answer_path.write_text(answers, encoding="utf-8")
    return answer_path, search_path


def question(**fields: object) -> AnswerQuestion:
    values: dict[str, object] = {
        "id": "q01",
        "kind": AnswerKind.SEARCH,
        "question": "Яку пеню платить покупець?",
        "expect": AnswerStatus.ANSWERED,
    }
    return AnswerQuestion.model_validate(values | fields)


def result(**fields: object) -> RagResult:
    values: dict[str, object] = {
        "question": "Яку пеню платить покупець?",
        "contract_id": "own_003",
        "outcome": Outcome.ANSWERED,
        "model_called": True,
        "sources": {"S1": "own_003#030"},
        "context_tokens": 500,
        "left_out": (),
        "below_threshold": (),
        "model": "claude-haiku-5-5",
        "prompt_version": "v1",
        "prompt_sha256": "abc",
        "attempts": 1,
        "latency_s": 2.0,
        "input_tokens": 3000,
        "output_tokens": 200,
    }
    return RagResult.model_validate(values | fields)


def test_search_questions_come_with_their_contract_and_their_evidence_chunks(
    tmp_path: Path,
) -> None:
    answer_path, search_path = write(tmp_path)

    questions = load_answer_questions(answer_path, search_path)

    assert [q.id for q in questions] == ["q01", "q09", "t01"]
    q01, q09, t01 = questions
    assert (q01.kind, q01.contract_id, q01.expect) == (
        AnswerKind.SEARCH,
        "own_003",
        AnswerStatus.ANSWERED,
    )
    assert q01.should_cite == ("own_003#030",)
    assert q01.note == "п. 7.2"
    assert q09.contract_id is None  # a law question: the doc_types filter of Е2.4 is not kept
    assert (t01.kind, t01.expect, t01.should_cite) == (
        AnswerKind.OFF_CORPUS,
        AnswerStatus.NOT_IN_DOCUMENTS,
        (),
    )


def test_an_unknown_search_question_is_refused(tmp_path: Path) -> None:
    answer_path, search_path = write(tmp_path, ANSWER_QUESTIONS.replace("q09]", "q99]"))

    with pytest.raises(ValueError, match="q99"):
        load_answer_questions(answer_path, search_path)


def test_an_id_used_twice_is_refused(tmp_path: Path) -> None:
    answer_path, search_path = write(tmp_path, ANSWER_QUESTIONS.replace("id: t01", "id: q01"))

    with pytest.raises(ValueError, match="q01"):
        load_answer_questions(answer_path, search_path)


def test_the_repository_questions_load_and_cover_every_kind() -> None:
    questions = load_answer_questions(ANSWER_QUESTIONS_PATH)
    search_ids = {q.id for q in load_questions(Path("data/rag/search_questions.yaml"))}

    assert len(questions) >= 20
    assert {q.kind for q in questions} == set(AnswerKind)
    assert search_ids <= {q.id for q in questions}


def test_the_outcome_matches_only_the_expected_one() -> None:
    assert outcome_matches(result(), question()) is True
    assert outcome_matches(result(outcome=Outcome.FAILED), question()) is False
    assert outcome_matches(
        result(outcome=Outcome.NOT_IN_DOCUMENTS), question(expect=AnswerStatus.NOT_IN_DOCUMENTS)
    )


def test_citing_any_evidence_chunk_is_enough_and_no_evidence_means_no_check() -> None:
    expected = question(should_cite=("court_910_13071_19#113", "court_910_13071_19#083"))

    assert cites_expected(result(cited=("court_910_13071_19#083",)), expected) is True
    assert cites_expected(result(cited=("own_003#031",)), expected) is False
    assert cites_expected(result(cited=()), question()) is None


def test_the_summary_counts_outcomes_tokens_cost_and_latency() -> None:
    pairs = [
        (question(should_cite=("own_003#030",)), result(cited=("own_003#030",), latency_s=1.0)),
        (question(id="q02", should_cite=("own_003#031",)), result(latency_s=3.0)),
        (
            question(id="t01", expect=AnswerStatus.NOT_IN_DOCUMENTS),
            result(
                outcome=Outcome.NOT_IN_DOCUMENTS,
                model_called=False,
                attempts=0,
                input_tokens=0,
                output_tokens=0,
                latency_s=0.5,
            ),
        ),
    ]

    summary = summarize(pairs, price_input=0.1, price_output=0.5)

    assert summary.questions == 3
    assert summary.outcomes == {Outcome.ANSWERED: 2, Outcome.NOT_IN_DOCUMENTS: 1}
    assert (summary.matched, summary.cite_checked, summary.cite_hits) == (3, 2, 1)
    assert (summary.model_calls, summary.input_tokens, summary.output_tokens) == (2, 6000, 400)
    assert summary.cost_usd == pytest.approx((6000 * 0.1 + 400 * 0.5) / 1_000_000)
    assert (summary.latency_p50_s, summary.latency_p95_s) == (1.0, pytest.approx(2.8))
