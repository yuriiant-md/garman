from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from scripts.ask_rag import (
    ANSWERS_NAME,
    REPORT_NAME,
    AnswerRecord,
    main,
    parse_args,
    prompt_path,
    render_report,
    run_questions,
    select_questions,
    write_run,
)
from src.rag.answer import NOT_IN_DOCUMENTS_TEXT, AnswerStatus
from src.rag.answer_config import ANSWER_CONFIG_PATH
from src.rag.answer_eval import summarize
from src.rag.answerer import Outcome, RagResult
from tests.test_answer_eval import question, result

QUESTIONS = (
    question(id="q01", should_cite=("own_003#030",)),
    question(id="t01", expect=AnswerStatus.NOT_IN_DOCUMENTS, kind="off_corpus"),
)
STARTED = datetime(2026, 11, 7, 12, 30, 5)


@dataclass
class FakeAnswerer:
    asked: list[tuple[str, str | None]] = field(default_factory=list)

    def ask(self, question: str, contract_id: str | None = None) -> RagResult:
        self.asked.append((question, contract_id))
        return result(question=question, contract_id=contract_id, cited=("own_003#030",))


def records() -> list[AnswerRecord]:
    return [
        AnswerRecord(
            question=QUESTIONS[0],
            result=result(
                cited=("own_003#030",), text="Покупець сплачує пеню. [own_003, п. 7.2]"
            ),
        ),
        AnswerRecord(
            question=QUESTIONS[1],
            result=result(
                outcome=Outcome.NOT_IN_DOCUMENTS,
                model_called=False,
                attempts=0,
                input_tokens=0,
                output_tokens=0,
                sources={},
                text=NOT_IN_DOCUMENTS_TEXT,
                below_threshold=("own_003#056",),
            ),
        ),
    ]


def test_ids_pick_questions_in_the_order_of_the_set_and_an_unknown_id_is_refused() -> None:
    assert select_questions(QUESTIONS, None) == QUESTIONS
    assert [q.id for q in select_questions(QUESTIONS, "t01,q01")] == ["q01", "t01"]
    with pytest.raises(ValueError, match="q99"):
        select_questions(QUESTIONS, "q01,q99")


def test_ids_that_name_no_question_are_refused() -> None:
    with pytest.raises(ValueError, match="no questions"):
        select_questions(QUESTIONS, " , ")


def test_every_question_is_asked_with_its_contract() -> None:
    answerer = FakeAnswerer()

    done = run_questions(answerer, QUESTIONS)

    assert answerer.asked == [(q.question, q.contract_id) for q in QUESTIONS]
    assert [record.question.id for record in done] == ["q01", "t01"]


def test_the_report_shows_the_summary_each_question_and_each_answer() -> None:
    done = records()
    summary = summarize([(r.question, r.result) for r in done], price_input=0.1, price_output=0.5)

    report = render_report(done, summary, ["Прогін: 2026-11-07 12:30"])

    assert report.startswith("Прогін: 2026-11-07 12:30")
    assert "| 2 | 2 з 2 | 1 з 1 | 1 |" in report
    assert "| q01 | search | answered | answered | ✓ | ✓ |" in report
    assert "| t01 | off_corpus | not_in_documents | not_in_documents | ✓ | — |" in report
    assert "Покупець сплачує пеню. [own_003, п. 7.2]" in report
    assert "модель не питали, фрагментів нижче порогу: 1" in report


def test_a_run_gets_a_new_folder_and_never_overwrites_one(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path, STARTED, "звіт", records())

    assert run_dir == tmp_path / "2026-11-07T12-30-05"
    assert (run_dir / REPORT_NAME).read_text(encoding="utf-8") == "звіт"
    lines = (run_dir / ANSWERS_NAME).read_text(encoding="utf-8").splitlines()
    assert [AnswerRecord.model_validate_json(line) for line in lines] == records()
    with pytest.raises(FileExistsError):
        write_run(tmp_path, STARTED, "звіт", records())


def test_the_prompt_of_the_config_is_used_unless_another_is_named() -> None:
    assert prompt_path(parse_args([]), Path("prompts/answer_v1.md")) == Path("prompts/answer_v1.md")
    named = parse_args(["--prompt", "prompts/answer_v2.md"])
    assert prompt_path(named, Path("prompts/answer_v1.md")) == Path("prompts/answer_v2.md")


def test_without_an_api_key_it_stops_before_loading_anything(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert main([]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def cost_report(*results: RagResult) -> str:
    done = [AnswerRecord(question=QUESTIONS[0], result=item) for item in results]
    summary = summarize([(r.question, r.result) for r in done], price_input=1.0, price_output=0.0)
    return render_report(done, summary, ["Прогін"])


def test_the_cost_per_1000_leaves_out_questions_that_got_no_answer() -> None:
    answered = result(input_tokens=1000, output_tokens=0)
    errored = result(outcome=Outcome.ERROR, input_tokens=0, output_tokens=0, error="timeout")

    assert "| 0.0010 | 1.00 |" in cost_report(answered, errored)


def test_the_cost_per_1000_is_a_dash_when_no_question_got_an_answer() -> None:
    errored = result(outcome=Outcome.ERROR, input_tokens=0, output_tokens=0, error="timeout")

    assert "| 0.0000 | — |" in cost_report(errored)


def config_with(tmp_path: Path, **changes: object) -> Path:
    values = yaml.safe_load(ANSWER_CONFIG_PATH.read_text(encoding="utf-8")) | changes
    path = tmp_path / "answer.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def broken_yaml(tmp_path: Path) -> list[str]:
    path = tmp_path / "broken.yaml"
    path.write_text("model: [", encoding="utf-8")
    return ["--config", str(path)]


def missing_prompt(tmp_path: Path) -> list[str]:
    return ["--prompt", str(tmp_path / "missing.md")]


def unknown_model(tmp_path: Path) -> list[str]:
    return ["--config", str(config_with(tmp_path, model="no-such-model"))]


def no_ids(_: Path) -> list[str]:
    return ["--ids", ","]


@pytest.mark.parametrize("bad_args", [broken_yaml, missing_prompt, unknown_model, no_ids])
def test_a_bad_setup_stops_with_code_2_before_qdrant_and_the_models(
    bad_args: Callable[[Path], list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(
        "scripts.ask_rag.reachable_client", lambda _: pytest.fail("Qdrant was reached")
    )

    assert main(bad_args(tmp_path)) == 2
    assert "cannot start" in capsys.readouterr().err
