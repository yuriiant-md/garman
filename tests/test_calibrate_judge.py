from pathlib import Path

import pytest

from scripts.calibrate_judge import CALIBRATION_NAME, main, verdicts_of
from scripts.judge_answers import JUDGEMENTS_NAME, JudgeRecord
from src.rag.calibration import (
    AnswerMarks,
    ClaimMark,
    JudgeVerdicts,
    MarkSheet,
    dump_sheet,
)
from src.rag.judge import CorrectJudgement, FaithfulJudgement, JudgeStatus
from tests.test_judge_answers import correct_judgement, faithful_judgement, record

RUN = "2026-11-10T10-08-19"


def entry(id_: str, seen: bool, correct: bool | None, *supported: bool | None) -> AnswerMarks:
    return AnswerMarks(
        id=id_,
        seen=seen,
        question=f"Питання {id_}?",
        reference="Пеня за п. 7.2.",
        answer="Відповідь.",
        correct=correct,
        claims=tuple(
            ClaimMark(n=n, text=f"Твердження {n}.", sources=("S1",), supported=flag)
            for n, flag in enumerate(supported, start=1)
        ),
    )


def marked_sheet(run: str = RUN, q09_claim_2: bool | None = False) -> MarkSheet:
    """q01 unseen: correct, 1 claim supported; q09 seen: correct, claim 1 yes, claim 2 no."""
    return MarkSheet(
        version=1,
        run=run,
        answers=(entry("q01", False, True, True), entry("q09", True, True, True, q09_claim_2)),
    )


def judged(id_: str, correct: CorrectJudgement, faithful: FaithfulJudgement) -> JudgeRecord:
    claims = len(faithful.verdict.verdicts) if faithful.verdict else 1
    return record(id_, claims=claims, correct=correct, faithful=faithful)


def agreeing_judge() -> list[JudgeRecord]:
    """The judge says what marked_sheet says."""
    return [
        judged("q01", correct_judgement(True), faithful_judgement(True)),
        judged("q09", correct_judgement(True), faithful_judgement(True, False)),
    ]


def save(tmp_path: Path, sheet: MarkSheet, records: list[JudgeRecord] | None) -> tuple[Path, Path]:
    """The sheet and a run folder named RUN under results/; records=None leaves no judge.jsonl."""
    marks = tmp_path / "judge_marks.yaml"
    dump_sheet(sheet, marks)
    run_dir = tmp_path / "results" / RUN
    run_dir.mkdir(parents=True)
    if records is not None:
        lines = "".join(f"{item.model_dump_json()}\n" for item in records)
        (run_dir / JUDGEMENTS_NAME).write_text(lines, encoding="utf-8")
    return marks, run_dir


def run_main(tmp_path: Path, marks: Path, *extra: str) -> int:
    return main([*extra, "--marks", str(marks), "--results-dir", str(tmp_path / "results")])


def report(run_dir: Path) -> str:
    return (run_dir / CALIBRATION_NAME).read_text(encoding="utf-8")


def test_verdicts_of_a_record_take_correct_and_every_claim_by_its_number() -> None:
    # Arrange
    item = judged("q09", correct_judgement(False), faithful_judgement(True, False))

    # Act
    verdicts = verdicts_of(item)

    # Assert
    assert verdicts == JudgeVerdicts(correct=False, supported=(True, False))


def test_a_judgement_with_no_verdict_gives_none_for_each_thing_it_should_have_judged() -> None:
    # Arrange
    item = record(
        "q09",
        claims=2,
        correct=correct_judgement(status=JudgeStatus.ERROR),
        faithful=faithful_judgement(True, True, status=JudgeStatus.FAILED),
    )

    # Act
    verdicts = verdicts_of(item)

    # Assert
    assert verdicts == JudgeVerdicts(correct=None, supported=(None, None))


def test_a_judge_that_agrees_with_every_mark_exits_0_and_counts_seen_and_unseen_apart(
    tmp_path: Path,
) -> None:
    # Arrange
    marks, run_dir = save(tmp_path, marked_sheet(), agreeing_judge())

    # Act
    code = run_main(tmp_path, marks)

    # Assert
    assert code == 0
    text = report(run_dir)
    assert f"# Калібрування судді: прогін {RUN}" in text
    assert "| correct | 2 з 2 (100%) | 1 з 1 (100%) | 1 з 1 (100%) |" in text
    assert "| faithful | 3 з 3 (100%) | 1 з 1 (100%) | 2 з 2 (100%) |" in text
    assert text.count("Немає.") == 3  # no disagreement of any kind


def test_a_judge_that_misses_an_error_is_listed_with_its_reason_and_fails_the_gate(
    tmp_path: Path,
) -> None:
    # Arrange: the author says claim 2 of q09 is not supported, the judge says it is
    judge = [
        judged("q01", correct_judgement(True), faithful_judgement(True)),
        judged("q09", correct_judgement(True), faithful_judgement(True, True)),
    ]
    marks, run_dir = save(tmp_path, marked_sheet(), judge)

    # Act
    code = run_main(tmp_path, marks)

    # Assert
    assert code == 1  # faithful agreement 2 of 3 = 67% < 80%
    missed = report(run_dir).split("## Суддя «так», автор «ні»")[1].split("##")[0]
    assert "- q09 (бачене), твердження 2: Пояснення." in missed


def test_a_too_strict_judge_is_listed_apart_from_missed_errors(tmp_path: Path) -> None:
    # Arrange: the author says q01 is correct, the judge says it is not
    judge = agreeing_judge()
    judge[0] = judged("q01", correct_judgement(False), faithful_judgement(True))
    marks, run_dir = save(tmp_path, marked_sheet(), judge)

    # Act
    code = run_main(tmp_path, marks)

    # Assert
    assert code == 1  # correct agreement 1 of 2 = 50% < 80%
    strict = report(run_dir).split("## Суддя «ні», автор «так»")[1].split("##")[0]
    assert "- q01, correct: Пояснення." in strict


def test_a_judgement_with_no_verdict_counts_as_a_disagreement_and_is_listed(
    tmp_path: Path,
) -> None:
    # Arrange
    judge = agreeing_judge()
    judge[0] = judged(
        "q01", correct_judgement(True), faithful_judgement(True, status=JudgeStatus.ERROR)
    )
    marks, run_dir = save(tmp_path, marked_sheet(), judge)

    # Act
    code = run_main(tmp_path, marks)

    # Assert
    assert code == 1  # faithful agreement 2 of 3
    text = report(run_dir)
    assert "| faithful | 2 з 3 (67%) | 0 з 1 (0%) | 2 з 2 (100%) |" in text
    assert "- q01, твердження 1" in text.split("## Без вердикту судді")[1]


def test_an_unmarked_sheet_exits_2_and_writes_no_report(tmp_path: Path) -> None:
    # Arrange
    marks, run_dir = save(tmp_path, marked_sheet(q09_claim_2=None), agreeing_judge())

    # Act / Assert
    assert run_main(tmp_path, marks) == 2
    assert not (run_dir / CALIBRATION_NAME).exists()


def test_a_named_run_other_than_the_sheets_exits_2(tmp_path: Path) -> None:
    # Arrange
    marks, _ = save(tmp_path, marked_sheet(), agreeing_judge())
    other = tmp_path / "results" / "2026-12-01T00-00-00"
    other.mkdir()

    # Act / Assert
    assert run_main(tmp_path, marks, str(other)) == 2
    assert not (other / CALIBRATION_NAME).exists()


@pytest.mark.parametrize("records", [None, agreeing_judge()[:1]])
def test_a_run_without_judgements_or_with_one_missing_exits_2(
    tmp_path: Path, records: list[JudgeRecord] | None
) -> None:
    # Arrange
    marks, run_dir = save(tmp_path, marked_sheet(), records)

    # Act / Assert
    assert run_main(tmp_path, marks) == 2
    assert not (run_dir / CALIBRATION_NAME).exists()
