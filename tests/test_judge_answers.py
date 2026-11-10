from dataclasses import dataclass, field
from pathlib import Path

import pytest

from scripts.ask_rag import ANSWERS_NAME, AnswerRecord
from scripts.judge_answers import (
    JUDGE_MD_NAME,
    JUDGEMENTS_NAME,
    JudgeRecord,
    judge_card,
    judge_gates,
    judge_pair,
    main,
    run_cost,
)
from src.extraction.metrics import cost_usd
from src.rag.answer import NOT_IN_DOCUMENTS_TEXT, Answer, AnswerStatus, Claim
from src.rag.answer_eval import AnswerQuestion, load_answer_questions
from src.rag.answerer import Outcome, RagResult
from src.rag.judge import (
    ClaimVerdict,
    CorrectJudgement,
    CorrectVerdict,
    FaithfulJudgement,
    FaithfulVerdict,
    JudgeConfig,
    JudgeStatus,
    JudgeThresholds,
    load_judge_config,
)
from tests.test_answer_eval import result, write

RUN_NAME = "2026-11-10T10-00-00"
DOCUMENT = '<document id="S1">\nДокумент: Договір own_003\nТекст:\nп. 7.2 Пеня.\n</document>'
LOGGED: dict[str, object] = {
    "model": "claude-sonnet-5-5", "prompt_version": "v1", "prompt_sha256": "x", "effort": "medium",
    "attempts": 1, "input_tokens": 1000, "output_tokens": 200, "latency_s": 3.0,
}


def correct_judgement(correct: bool = True, status: JudgeStatus = JudgeStatus.OK) -> CorrectJudgement:
    verdict = CorrectVerdict(
        conclusion_matches=correct, citation_fits=True, numbers_ok=True, reason="Пояснення."
    )
    return CorrectJudgement(
        status=status, verdict=verdict if status is JudgeStatus.OK else None, **LOGGED  # type: ignore[arg-type]
    )


def faithful_judgement(*flags: bool, status: JudgeStatus = JudgeStatus.OK) -> FaithfulJudgement:
    verdict = FaithfulVerdict(
        verdicts=tuple(
            ClaimVerdict(n=n, supported=flag, reason="Пояснення.")
            for n, flag in enumerate(flags, start=1)
        )
    )
    return FaithfulJudgement(
        status=status, verdict=verdict if status is JudgeStatus.OK else None, **LOGGED  # type: ignore[arg-type]
    )


@dataclass
class FakeJudge:
    """Judges every answer correct and every claim supported unless told otherwise."""

    correct_ok: bool = True
    supported_ok: bool = True
    correct_calls: list[tuple[str, str, str]] = field(default_factory=list)
    faithful_calls: list[RagResult] = field(default_factory=list)

    def correct(self, question: str, reference: str, answer: str) -> CorrectJudgement:
        self.correct_calls.append((question, reference, answer))
        return correct_judgement(self.correct_ok)

    def faithful(self, result_: RagResult) -> FaithfulJudgement:
        self.faithful_calls.append(result_)
        claims = len(result_.answer.claims) if result_.answer else 0
        return faithful_judgement(*([self.supported_ok] * claims))


def answered(*claims: str, documents: bool = True) -> RagResult:
    answer = Answer(
        status=AnswerStatus.ANSWERED,
        claims=tuple(Claim(text=text, sources=("S1",)) for text in claims),
    )
    return result(
        answer=answer,
        text="\n".join(f"{claim} [own_003, п. 7.2]" for claim in claims),
        cited=("own_003#030",),
        cited_documents={"S1": DOCUMENT} if documents else {},
    )


def nothing_found() -> RagResult:
    return result(
        outcome=Outcome.NOT_IN_DOCUMENTS, model_called=False, sources={}, text=NOT_IN_DOCUMENTS_TEXT
    )


def record(id_: str, *, expects_answer: bool = True, answered_: bool = True, claims: int = 1,
           correct: CorrectJudgement | None = None,
           faithful: FaithfulJudgement | None = None) -> JudgeRecord:
    return JudgeRecord(
        id=id_, expects_answer=expects_answer, answered=answered_, claims=claims,
        correct=correct, faithful=faithful,
    )


@dataclass(frozen=True)
class Paths:
    questions: Path
    search_questions: Path
    results: Path

    def args(self, *extra: str) -> list[str]:
        return [
            *extra,
            "--questions", str(self.questions),
            "--search-questions", str(self.search_questions),
            "--results-dir", str(self.results),
        ]

    def loaded(self) -> tuple[AnswerQuestion, ...]:
        return load_answer_questions(self.questions, self.search_questions)


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    answer_path, search_path = write(tmp_path)  # q01, q09 expect an answer, t01 does not
    return Paths(answer_path, search_path, tmp_path / "answers")


def save_run(paths: Paths, documents: bool = True) -> Path:
    results = {
        "q01": answered("Пеня в розмірі подвійної облікової ставки НБУ.", documents=documents),
        "q09": answered("Три проценти річних.", "Плюс інфляційні.", documents=documents),
        "t01": nothing_found(),
    }
    records = [AnswerRecord(question=q, result=results[q.id]) for q in paths.loaded()]
    run_dir = paths.results / RUN_NAME
    run_dir.mkdir(parents=True)
    lines = "".join(f"{item.model_dump_json()}\n" for item in records)
    (run_dir / ANSWERS_NAME).write_text(lines, encoding="utf-8")
    return run_dir


def run_main(paths: Paths, judge: FakeJudge, *extra: str) -> int:
    return main(paths.args(*extra), make_judge=lambda *_: judge)


# --- which checks a question gets


def test_an_answered_question_gets_both_checks_against_its_reference(paths: Paths) -> None:
    q01 = paths.loaded()[0]
    judge = FakeJudge()
    answer = answered("Пеня.", "Ще одна.")

    item = judge_pair(judge, q01, answer)

    assert judge.correct_calls == [(q01.question, q01.expected_answer, answer.text)]
    assert judge.faithful_calls == [answer]
    assert (item.id, item.expects_answer, item.answered, item.claims) == ("q01", True, True, 2)
    assert item.correct is not None and item.faithful is not None


def test_a_question_left_unanswered_is_not_sent_to_the_judge(paths: Paths) -> None:
    judge = FakeJudge()

    item = judge_pair(judge, paths.loaded()[0], nothing_found())

    assert (judge.correct_calls, judge.faithful_calls) == ([], [])
    assert (item.expects_answer, item.answered, item.correct, item.faithful) == (
        True, False, None, None,
    )


def test_a_question_that_expects_no_answer_gets_no_correct_check(paths: Paths) -> None:
    t01 = paths.loaded()[2]
    judge = FakeJudge()

    item = judge_pair(judge, t01, answered("Вигадане твердження."))

    assert judge.correct_calls == []
    assert len(judge.faithful_calls) == 1  # what it did say is still checked
    assert (item.expects_answer, item.correct) == (False, None)


# --- the numbers


def test_correct_counts_every_question_that_expects_an_answer() -> None:
    card = judge_card([
        record("q01", correct=correct_judgement(True)),
        record("q02", correct=correct_judgement(False)),
        record("q03", answered_=False, claims=0),  # not answered: wrong, and no call
        record("q04", correct=correct_judgement(status=JudgeStatus.FAILED)),  # no verdict: wrong
        record("t01", expects_answer=False, answered_=False, claims=0),
    ])

    assert (card.correct.hit, card.correct.total) == (1, 4)
    assert card.not_judged == 1


def test_faithful_counts_supported_claims_of_all_answers() -> None:
    card = judge_card([
        record("q01", claims=2, faithful=faithful_judgement(True, False)),
        record("q02", claims=3, faithful=faithful_judgement(True, True, True)),
        record("q03", claims=2, faithful=faithful_judgement(status=JudgeStatus.ERROR)),
    ])

    assert (card.faithful.hit, card.faithful.total) == (4, 7)  # unjudged claims are not supported
    assert card.not_judged == 1


def test_the_gates_use_the_judge_thresholds() -> None:
    card = judge_card([
        record("q01", correct=correct_judgement(True), faithful=faithful_judgement(True)),
        record("q02", correct=correct_judgement(False), faithful=faithful_judgement(True)),
    ])
    thresholds = JudgeThresholds(correct_min=0.8, faithful_min=0.9, agreement_min=0.8)

    gates = {gate.name: gate.passed for gate in judge_gates(card, thresholds)}

    assert gates == {"correct": False, "faithful": True}


def test_the_cost_is_every_judgement_priced_by_the_judge_model() -> None:
    config: JudgeConfig = load_judge_config()
    records = [
        record("q01", correct=correct_judgement(), faithful=faithful_judgement(True)),
        record("q03", answered_=False, claims=0),
    ]

    expected = cost_usd(
        2000, 400, config.model.price_input_per_mtok, config.model.price_output_per_mtok
    )
    assert run_cost(records, config.model) == pytest.approx(expected)


# --- the script


def test_a_run_over_both_thresholds_exits_0_and_writes_its_judgements(paths: Paths) -> None:
    run_dir = save_run(paths)
    judge = FakeJudge()

    assert run_main(paths, judge, str(run_dir)) == 0

    lines = (run_dir / JUDGEMENTS_NAME).read_text(encoding="utf-8").splitlines()
    assert [JudgeRecord.model_validate_json(line).id for line in lines] == ["q01", "q09", "t01"]
    report = (run_dir / JUDGE_MD_NAME).read_text(encoding="utf-8")
    assert "claude-sonnet-5-5" in report and "q09" in report
    assert len(judge.correct_calls) == 2 and len(judge.faithful_calls) == 2


def test_a_run_below_a_threshold_exits_1(paths: Paths) -> None:
    run_dir = save_run(paths)

    assert run_main(paths, FakeJudge(supported_ok=False), str(run_dir)) == 1
    assert (run_dir / JUDGE_MD_NAME).exists()


def test_without_a_named_run_the_newest_one_is_judged(paths: Paths) -> None:
    run_dir = save_run(paths)

    assert run_main(paths, FakeJudge()) == 0
    assert (run_dir / JUDGEMENTS_NAME).exists()


def test_a_limited_run_judges_the_first_questions_only_and_checks_no_threshold(
    paths: Paths,
) -> None:
    run_dir = save_run(paths)
    judge = FakeJudge(correct_ok=False)  # below the threshold, yet a sample says nothing of it

    assert run_main(paths, judge, str(run_dir), "--limit", "1") == 0

    assert len(judge.correct_calls) == 1
    report = (run_dir / JUDGE_MD_NAME).read_text(encoding="utf-8")
    assert "--limit 1" in report


def test_a_run_saved_before_the_answers_kept_their_documents_exits_2_before_any_call(
    paths: Paths, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = save_run(paths, documents=False)
    judge = FakeJudge()

    assert run_main(paths, judge, str(run_dir)) == 2

    assert (judge.correct_calls, judge.faithful_calls) == ([], [])
    assert "q01" in capsys.readouterr().err


def test_no_run_at_all_exits_2(paths: Paths) -> None:
    assert run_main(paths, FakeJudge()) == 2
