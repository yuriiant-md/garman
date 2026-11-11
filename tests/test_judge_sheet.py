from pathlib import Path

import pytest

from scripts.ask_rag import ANSWERS_NAME
from scripts.judge_sheet import main
from src.rag.calibration import load_sheet, unmarked
from tests.test_answer_eval import write
from tests.test_judge_answers import RUN_NAME, Paths, save_run


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    answer_path, search_path = write(tmp_path)  # q01, q09 expect an answer, t01 does not
    return Paths(answer_path, search_path, tmp_path / "answers")


def run_main(paths: Paths, out: Path, *extra: str) -> int:
    return main([*paths.args(*extra), "--out", str(out)])


def test_the_sheet_of_the_newest_run_holds_every_judged_answer_with_empty_marks(
    paths: Paths, tmp_path: Path
) -> None:
    # Arrange
    save_run(paths)
    out = tmp_path / "judge_marks.yaml"

    # Act
    code = run_main(paths, out, "--seen", "q09")

    # Assert
    assert code == 0
    sheet = load_sheet(out)
    assert sheet.run == RUN_NAME
    assert [(entry.id, entry.seen) for entry in sheet.answers] == [("q01", False), ("q09", True)]
    assert unmarked(sheet) == [
        "q01 correct", "q01 claim 1", "q09 correct", "q09 claim 1", "q09 claim 2",
    ]
    assert out.read_text(encoding="utf-8").startswith("# Manual marks")  # how to mark, on top


def test_a_sheet_that_exists_is_not_overwritten_without_force(paths: Paths, tmp_path: Path) -> None:
    # Arrange
    save_run(paths)
    out = tmp_path / "judge_marks.yaml"
    out.write_text("my marks\n", encoding="utf-8")

    # Act
    refused = run_main(paths, out)
    kept = out.read_text(encoding="utf-8")
    forced = run_main(paths, out, "--force")

    # Assert
    assert refused == 2 and kept == "my marks\n"
    assert forced == 0 and load_sheet(out).run == RUN_NAME


@pytest.mark.parametrize("seen", ["q99", "q01,,q09"])
def test_an_unknown_or_empty_seen_id_exits_2_and_writes_nothing(
    paths: Paths, tmp_path: Path, seen: str
) -> None:
    # Arrange
    save_run(paths)
    out = tmp_path / "judge_marks.yaml"

    # Act
    code = run_main(paths, out, "--seen", seen)

    # Assert
    assert code == 2
    assert not out.exists()


def test_a_run_saved_before_the_answers_kept_their_documents_exits_2(
    paths: Paths, tmp_path: Path
) -> None:
    # Arrange
    save_run(paths, documents=False)
    out = tmp_path / "judge_marks.yaml"

    # Act / Assert
    assert run_main(paths, out) == 2
    assert not out.exists()


def test_a_named_run_is_used_instead_of_the_newest(paths: Paths, tmp_path: Path) -> None:
    # Arrange
    run_dir = save_run(paths)
    newer = paths.results / "2026-12-01T00-00-00"
    newer.mkdir()
    (newer / ANSWERS_NAME).write_text("", encoding="utf-8")
    out = tmp_path / "judge_marks.yaml"

    # Act
    code = main([str(run_dir), *paths.args(), "--out", str(out)])

    # Assert
    assert code == 0
    assert load_sheet(out).run == RUN_NAME
