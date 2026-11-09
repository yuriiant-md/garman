from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.ask_rag import ANSWERS_NAME, AnswerRecord
from scripts.score_answers import SCORES_NAME, main
from src.rag.answer import NOT_IN_DOCUMENTS_TEXT, AnswerStatus
from src.rag.answer_eval import AnswerQuestion, load_answer_questions
from src.rag.answerer import Outcome, RagResult
from src.rag.sources import DocType
from tests.test_answer_eval import result, write
from tests.test_answer_scoring import VALID_THRESHOLDS, answered, chunk

GOOD_CLAIMS = {
    "q01": "Пеня в розмірі подвійної облікової ставки НБУ.",
    "q09": "Три проценти річних.",
}
RUN_NAME = "2026-11-09T10-00-00"


@dataclass(frozen=True)
class Paths:
    tmp: Path
    questions: Path
    search_questions: Path
    chunks: Path
    thresholds: Path
    results: Path

    def args(self, *run: str) -> list[str]:
        return [
            *run,
            "--questions", str(self.questions),
            "--search-questions", str(self.search_questions),
            "--chunks", str(self.chunks),
            "--thresholds", str(self.thresholds),
            "--results-dir", str(self.results),
        ]

    def loaded(self) -> tuple[AnswerQuestion, ...]:
        return load_answer_questions(self.questions, self.search_questions)


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    answer_path, search_path = write(tmp_path)
    chunks_path = tmp_path / "chunks.jsonl"
    corpus = (chunk(), chunk("law_civil_code#026", doc_type=DocType.LAW, in_force=True))
    chunks_path.write_text(
        "".join(f"{item.model_dump_json()}\n" for item in corpus), encoding="utf-8"
    )
    thresholds_path = tmp_path / "eval.yaml"
    thresholds_path.write_text(VALID_THRESHOLDS, encoding="utf-8")
    return Paths(
        tmp_path, answer_path, search_path, chunks_path, thresholds_path, tmp_path / "answers"
    )


def good_result(question: AnswerQuestion) -> RagResult:
    if question.expect is AnswerStatus.NOT_IN_DOCUMENTS:
        return result(
            outcome=Outcome.NOT_IN_DOCUMENTS,
            model_called=False,
            sources={},
            text=NOT_IN_DOCUMENTS_TEXT,
        )
    chunk_id = question.should_cite[0]
    return answered(GOOD_CLAIMS[question.id], sources={"S1": chunk_id}, cited=(chunk_id,))


def save_run(paths: Paths, records: list[AnswerRecord], name: str = RUN_NAME) -> Path:
    run_dir = paths.results / name
    run_dir.mkdir(parents=True)
    lines = "".join(f"{record.model_dump_json()}\n" for record in records)
    (run_dir / ANSWERS_NAME).write_text(lines, encoding="utf-8")
    return run_dir


def good_records(paths: Paths) -> list[AnswerRecord]:
    return [AnswerRecord(question=q, result=good_result(q)) for q in paths.loaded()]


def test_a_run_over_every_threshold_exits_0_and_writes_its_scores(paths: Paths) -> None:
    run_dir = save_run(paths, good_records(paths))

    assert main(paths.args(str(run_dir))) == 0

    scores = (run_dir / SCORES_NAME).read_text(encoding="utf-8")
    assert "q01" in scores
    assert "✗" not in scores


def test_a_run_below_a_threshold_exits_1_and_still_writes_its_scores(paths: Paths) -> None:
    records = good_records(paths)
    weak = [
        record.model_copy(update={"result": answered("Щось інше.", cited=("own_003#030",))})
        if record.question.id == "q01"
        else record
        for record in records
    ]
    run_dir = save_run(paths, weak)

    assert main(paths.args(str(run_dir))) == 1

    scores = (run_dir / SCORES_NAME).read_text(encoding="utf-8")
    assert "✗" in scores
    assert "подвійн" in scores


def test_a_fact_with_alternatives_does_not_break_the_table(paths: Paths) -> None:
    records = [
        record.model_copy(update={"result": answered("Щось інше.", cited=("own_003#030",))})
        if record.question.id == "q01"
        else record
        for record in good_records(paths)
    ]
    run_dir = save_run(paths, records)

    main(paths.args(str(run_dir)))

    scores = (run_dir / SCORES_NAME).read_text(encoding="utf-8")
    assert "облікової ставки / облікову ставку" in scores
    assert "облікової ставки|облікову" not in scores


def test_without_a_named_run_the_newest_one_is_scored(paths: Paths) -> None:
    records = good_records(paths)
    older = save_run(paths, records[:1], "2026-11-08T10-00-00")
    newest = save_run(paths, records, RUN_NAME)

    assert main(paths.args()) == 0

    assert (newest / SCORES_NAME).exists()
    assert not (older / SCORES_NAME).exists()


def test_a_run_without_some_questions_is_refused_by_name(
    paths: Paths, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = save_run(paths, good_records(paths)[:2])

    assert main(paths.args(str(run_dir))) == 2

    assert "t01" in capsys.readouterr().err
    assert not (run_dir / SCORES_NAME).exists()


def test_a_run_that_asked_another_text_is_refused_by_name(
    paths: Paths, capsys: pytest.CaptureFixture[str]
) -> None:
    records = good_records(paths)
    changed = records[0].question.model_copy(update={"question": "Інше питання?"})
    run_dir = save_run(paths, [records[0].model_copy(update={"question": changed}), *records[1:]])

    assert main(paths.args(str(run_dir))) == 2

    assert "q01" in capsys.readouterr().err


def test_a_run_citing_a_chunk_that_is_gone_is_refused(
    paths: Paths, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [
        record.model_copy(
            update={
                "result": answered(
                    GOOD_CLAIMS["q01"], sources={"S1": "gone#001"}, cited=("gone#001",)
                )
            }
        )
        if record.question.id == "q01"
        else record
        for record in good_records(paths)
    ]
    run_dir = save_run(paths, records)

    assert main(paths.args(str(run_dir))) == 2

    assert "gone#001" in capsys.readouterr().err


def test_no_run_at_all_exits_2(paths: Paths, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(paths.args()) == 2

    assert "no run" in capsys.readouterr().err


def test_a_folder_without_answers_exits_2(paths: Paths) -> None:
    empty = paths.results / RUN_NAME
    empty.mkdir(parents=True)

    assert main(paths.args(str(empty))) == 2


def test_broken_thresholds_exit_2(paths: Paths) -> None:
    paths.thresholds.write_text("source_min: 2\n", encoding="utf-8")
    run_dir = save_run(paths, good_records(paths))

    assert main(paths.args(str(run_dir))) == 2
