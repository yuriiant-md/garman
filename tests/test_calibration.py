from pathlib import Path

import pytest

from src.rag.answer import NOT_IN_DOCUMENTS_TEXT, Answer, AnswerStatus, Claim
from src.rag.answer_eval import AnswerQuestion
from src.rag.answer_scoring import Ratio
from src.rag.answerer import Outcome, RagResult
from src.rag.calibration import (
    AnswerMarks,
    Check,
    ClaimMark,
    JudgeVerdicts,
    MarkSheet,
    agreement,
    build_sheet,
    calibrate,
    calibration_gates,
    dump_sheet,
    load_sheet,
    unmarked,
)
from src.rag.judge import JudgeThresholds
from tests.test_answer_eval import question, result

RUN = "2026-11-10T10-08-19"
DOC_S1 = '<document id="S1">\nДокумент: Договір own_003\nТекст:\nп. 7.2 Пеня.\n</document>'
DOC_S2 = '<document id="S2">\nДокумент: ЦК\nТекст:\nСтаття 549.\n</document>'
THRESHOLDS = JudgeThresholds(correct_min=0.8, faithful_min=0.9, agreement_min=0.8)


def answered(*claims: tuple[str, tuple[str, ...]], **fields: object) -> RagResult:
    answer = Answer(
        status=AnswerStatus.ANSWERED,
        claims=tuple(Claim(text=text, sources=sources) for text, sources in claims),
    )
    values: dict[str, object] = {
        "text": "Пеня — подвійна облікова ставка НБУ.\n\n— Договір own_003, п. 7.2",
        "answer": answer,
        "sources": {"S1": "own_003#030", "S2": "law_cc#549", "S3": "law_cc#625"},
        "cited_documents": {"S1": DOC_S1, "S2": DOC_S2},
    }
    return result(**(values | fields))


def referenced(id_: str = "q01", **fields: object) -> AnswerQuestion:
    return question(id=id_, expected_answer="Подвійна облікова ставка НБУ (п. 7.2).", **fields)


def marks(
    id_: str = "q01",
    seen: bool = False,
    correct: bool | None = True,
    supported: tuple[bool | None, ...] = (True,),
    reference: str | None = "Еталон.",
) -> AnswerMarks:
    return AnswerMarks(
        id=id_,
        seen=seen,
        question="Питання?",
        reference=reference,
        answer="Відповідь.",
        correct=correct if reference is not None else None,
        claims=tuple(
            ClaimMark(n=n, text=f"Твердження {n}.", sources=("S1",), supported=flag)
            for n, flag in enumerate(supported, start=1)
        ),
        documents={"S1": DOC_S1},
    )


def sheet(*answers: AnswerMarks) -> MarkSheet:
    return MarkSheet(version=1, run=RUN, answers=answers)


# --- the sheet


def test_sheet_holds_the_answer_its_claims_and_only_the_cited_documents_with_empty_marks() -> None:
    # Arrange
    pair = (
        referenced(),
        answered(("Пеня — подвійна ставка.", ("S1",)), ("Це неустойка.", ("S2",))),
    )

    # Act
    built = build_sheet(RUN, [pair], seen=frozenset())

    # Assert
    (entry,) = built.answers
    assert entry.id == "q01" and entry.seen is False
    assert entry.reference == "Подвійна облікова ставка НБУ (п. 7.2)."
    assert entry.answer.startswith("Пеня — подвійна облікова ставка НБУ.")
    assert entry.correct is None
    assert [(c.n, c.text, c.sources, c.supported) for c in entry.claims] == [
        (1, "Пеня — подвійна ставка.", ("S1",), None),
        (2, "Це неустойка.", ("S2",), None),
    ]
    assert entry.documents == {"S1": DOC_S1, "S2": DOC_S2}


def test_sheet_leaves_out_what_the_judge_does_not_judge() -> None:
    # Arrange: unanswered (q03), an honest «немає» with no claims (t01)
    unanswered = (
        referenced("q03"),
        result(outcome=Outcome.NOT_IN_DOCUMENTS, text=NOT_IN_DOCUMENTS_TEXT),
    )
    honest = (
        question(id="t01", expect=AnswerStatus.NOT_IN_DOCUMENTS),
        result(outcome=Outcome.NOT_IN_DOCUMENTS, text=NOT_IN_DOCUMENTS_TEXT),
    )

    # Act
    built = build_sheet(RUN, [unanswered, honest], seen=frozenset())

    # Assert
    assert built.answers == ()


def test_claims_of_a_question_that_expects_no_answer_are_marked_without_a_reference() -> None:
    # Arrange
    pair = (question(id="t06", expect=AnswerStatus.NOT_IN_DOCUMENTS), answered(("Щось.", ("S1",))))

    # Act
    (entry,) = build_sheet(RUN, [pair], seen=frozenset()).answers

    # Assert
    assert entry.reference is None
    assert len(entry.claims) == 1


def test_sheet_marks_the_questions_already_seen_and_refuses_an_unknown_id() -> None:
    # Arrange
    pairs = [
        (referenced("q01"), answered(("А.", ("S1",)))),
        (referenced("q02"), answered(("Б.", ("S1",)))),
    ]

    # Act
    built = build_sheet(RUN, pairs, seen=frozenset({"q02"}))

    # Assert
    assert [(entry.id, entry.seen) for entry in built.answers] == [("q01", False), ("q02", True)]
    with pytest.raises(ValueError, match="q99"):
        build_sheet(RUN, pairs, seen=frozenset({"q02", "q99"}))


def test_unmarked_names_every_empty_mark_and_nothing_once_all_are_filled() -> None:
    # Arrange
    empty = marks("q01", correct=None, supported=(True, None))
    no_reference = marks("t06", reference=None, supported=(None,))

    # Act
    missing = unmarked(sheet(empty, no_reference, marks("q02")))

    # Assert
    assert missing == ["q01 correct", "q01 claim 2", "t06 claim 1"]
    assert unmarked(sheet(marks("q01"), marks("t06", reference=None))) == []


def test_sheet_survives_a_round_trip_and_keeps_long_texts_readable(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "judge_marks.yaml"
    original = sheet(marks("q01", correct=None, supported=(None,)))

    # Act
    dump_sheet(original, path)
    text = path.read_text(encoding="utf-8")

    # Assert
    assert load_sheet(path) == original
    assert "Договір own_003" in text  # Ukrainian as is, not \u escapes
    assert "S1: |" in text  # a document is a block, line by line as the model read it


def test_a_broken_sheet_is_a_value_error_that_names_the_file(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "judge_marks.yaml"
    path.write_text(
        "version: 1\nrun: x\nanswers:\n  - id: q01\n    correct: maybe\n", encoding="utf-8"
    )

    # Act / Assert
    with pytest.raises(ValueError, match="judge_marks.yaml"):
        load_sheet(path)


# --- agreement


def test_agreement_counts_correct_by_answers_and_faithful_by_claims_split_by_seen() -> None:
    # Arrange: the author agrees on q01 everywhere; on q02 (seen) the judge missed claim 2
    marked = sheet(
        marks("q01", correct=True, supported=(True, False)),
        marks("q02", seen=True, correct=False, supported=(True, False)),
    )
    judge = {
        "q01": JudgeVerdicts(correct=True, supported=(True, False)),
        "q02": JudgeVerdicts(correct=False, supported=(True, True)),
    }

    # Act
    items = calibrate(marked, judge)

    # Assert
    assert agreement(items, Check.CORRECT) == Ratio(2, 2)
    assert agreement(items, Check.FAITHFUL) == Ratio(3, 4)
    assert agreement(items, Check.FAITHFUL, seen=False) == Ratio(2, 2)
    assert agreement(items, Check.FAITHFUL, seen=True) == Ratio(1, 2)
    (miss,) = [item for item in items if not item.agrees]
    assert (miss.id, miss.check, miss.n, miss.judge, miss.author) == (
        "q02",
        Check.FAITHFUL,
        2,
        True,
        False,
    )


def test_a_judgement_with_no_verdict_is_a_disagreement() -> None:
    # Arrange
    marked = sheet(marks("q01", correct=True, supported=(True,)))
    judge = {"q01": JudgeVerdicts(correct=None, supported=(None,))}

    # Act
    items = calibrate(marked, judge)

    # Assert
    assert agreement(items, Check.CORRECT) == Ratio(0, 1)
    assert agreement(items, Check.FAITHFUL) == Ratio(0, 1)


def test_calibration_refuses_marks_and_verdicts_that_do_not_fit() -> None:
    # Arrange
    marked = sheet(marks("q01", supported=(True, True)))

    # Act / Assert
    with pytest.raises(ValueError, match="q01"):
        calibrate(marked, {})  # the judge has no record of q01
    with pytest.raises(ValueError, match="q01"):
        calibrate(marked, {"q01": JudgeVerdicts(correct=True, supported=(True,))})  # 2 claims vs 1
    with pytest.raises(ValueError, match="q01 correct"):
        calibrate(sheet(marks("q01", correct=None)), {"q01": JudgeVerdicts(True, (True,))})


def test_gates_need_the_agreement_on_both_checks() -> None:
    # Arrange: correct agrees 1 of 1, faithful 3 of 4 (75% < 80%)
    marked = sheet(marks("q01", correct=True, supported=(True, True, True, True)))
    judge = {"q01": JudgeVerdicts(correct=True, supported=(True, True, True, False))}

    # Act
    gates = calibration_gates(calibrate(marked, judge), THRESHOLDS)

    # Assert
    assert [(gate.name, gate.measured, gate.passed) for gate in gates] == [
        ("correct_agreement", 1.0, True),
        ("faithful_agreement", 0.75, False),
    ]
