"""Judge a saved run of the stage 2 answers with the LLM judge: paid, needs ANTHROPIC_API_KEY.

Reads answers.jsonl of a run of scripts/ask_rag.py, asks the judge of configs/judge.yaml two
checks of every answer (src/rag/judge.py) and writes judge.jsonl and judge.md next to it.
- correct: of the questions that expect an answer, those whose answer reaches the reference's
  conclusion with the right citation and numbers. One left unanswered is wrong and costs nothing;
  the questions that expect «немає» are left to scripts/score_answers.py, which checks them free.
- faithful: of all claims of all answers, those the documents they cite support.

    uv run --env-file .env python -m scripts.judge_answers              # the newest run
    uv run --env-file .env python -m scripts.judge_answers --limit 2    # a check on 2 questions

Exit code: 0 — both thresholds are met, or --limit judged a sample, which says nothing of them;
1 — some is not; 2 — the run cannot be judged (checked before any paid call: a run saved before
the answers kept their documents, or one that does not cover the current questions).
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from scripts.ask_rag import (
    ANSWERS_NAME,
    MARKS,
    RESULTS_DIR,
    SETUP_ERRORS,
    AnswerRecord,
    _table,
)
from scripts.score_answers import latest_run, pair_with_questions
from src.extraction.metrics import cost_usd
from src.extraction.prompts import Prompt, load_prompt
from src.rag.answer import AnswerStatus
from src.rag.answer_eval import (
    ANSWER_QUESTIONS_PATH,
    SEARCH_QUESTIONS_PATH,
    AnswerQuestion,
    load_answer_questions,
)
from src.rag.answer_scoring import Gate, Ratio
from src.rag.answerer import Outcome, RagResult
from src.rag.documents import read_jsonl
from src.rag.judge import (
    JUDGE_CONFIG_PATH,
    CorrectJudgement,
    FaithfulJudgement,
    Judge,
    JudgeConfig,
    Judgement,
    JudgeModel,
    JudgeStatus,
    JudgeThresholds,
    faithful_turn,
    load_judge_config,
)

JUDGEMENTS_NAME = "judge.jsonl"
JUDGE_MD_NAME = "judge.md"


class Judging(Protocol):
    def correct(self, question: str, reference: str, answer: str) -> CorrectJudgement: ...
    def faithful(self, result: RagResult) -> FaithfulJudgement: ...


type MakeJudge = Callable[[JudgeConfig, Prompt, Prompt], Judging]


class JudgeRecord(BaseModel):
    """One line of judge.jsonl: what one answer got from the judge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    expects_answer: bool  # the question expects an answer and has a reference for it
    answered: bool
    claims: int
    correct: CorrectJudgement | None = None  # None: expects no answer, or none was given
    faithful: FaithfulJudgement | None = None  # None: the answer has no claims


@dataclass(frozen=True)
class JudgeCard:
    correct: Ratio
    faithful: Ratio
    not_judged: int  # judgements with no verdict: failed or error, each counted as a miss


def judge_pair(judge: Judging, question: AnswerQuestion, result: RagResult) -> JudgeRecord:
    expects = question.expect is AnswerStatus.ANSWERED and question.expected_answer is not None
    text = result.text if result.outcome is Outcome.ANSWERED else None
    claims = len(result.answer.claims) if result.answer else 0
    correct = (
        judge.correct(question.question, question.expected_answer, text)
        if expects and text is not None and question.expected_answer is not None
        else None
    )
    return JudgeRecord(
        id=question.id,
        expects_answer=expects,
        answered=text is not None,
        claims=claims,
        correct=correct,
        faithful=judge.faithful(result) if claims else None,
    )


def _judgements(record: JudgeRecord) -> list[Judgement]:
    return [j for j in (record.correct, record.faithful) if j is not None]


def _is_correct(record: JudgeRecord) -> bool:
    verdict = record.correct.verdict if record.correct else None
    return verdict is not None and verdict.correct


def _supported(record: JudgeRecord) -> int:
    verdict = record.faithful.verdict if record.faithful else None
    return sum(item.supported for item in verdict.verdicts) if verdict else 0


def judge_card(records: Sequence[JudgeRecord]) -> JudgeCard:
    return JudgeCard(
        correct=Ratio(
            sum(map(_is_correct, records)), sum(record.expects_answer for record in records)
        ),
        faithful=Ratio(sum(map(_supported, records)), sum(record.claims for record in records)),
        not_judged=sum(
            judgement.status is not JudgeStatus.OK
            for record in records
            for judgement in _judgements(record)
        ),
    )


def judge_gates(card: JudgeCard, thresholds: JudgeThresholds) -> tuple[Gate, ...]:
    measured = (
        ("correct", card.correct, thresholds.correct_min),
        ("faithful", card.faithful, thresholds.faithful_min),
    )
    return tuple(Gate(name, ratio.rate, limit, ratio.rate >= limit) for name, ratio, limit in measured)


def run_cost(records: Sequence[JudgeRecord], model: JudgeModel) -> float:
    judgements = [judgement for record in records for judgement in _judgements(record)]
    return cost_usd(
        sum(j.input_tokens for j in judgements),
        sum(j.output_tokens for j in judgements),
        model.price_input_per_mtok,
        model.price_output_per_mtok,
    )


@dataclass(frozen=True)
class Judged:
    run_dir: Path
    config: JudgeConfig
    prompts: tuple[Prompt, Prompt]  # correct, faithful
    limit: int | None
    records: tuple[JudgeRecord, ...]

    @property
    def card(self) -> JudgeCard:
        return judge_card(self.records)

    @property
    def gates(self) -> tuple[Gate, ...]:
        """None checked on a sample: its numbers say nothing of the whole run."""
        return () if self.limit else judge_gates(self.card, self.config.thresholds)


def _ratio(ratio: Ratio) -> str:
    return f"{ratio.hit} з {ratio.total} ({ratio.rate:.0%})"


def _correct_cells(record: JudgeRecord) -> list[str]:
    if not record.expects_answer:
        return ["—"] * 4
    if record.correct is None:
        return ["✗ не відповів", "—", "—", "—"]
    verdict = record.correct.verdict
    if verdict is None:
        return [record.correct.status.value, "—", "—", "—"]
    checks = (verdict.conclusion_matches, verdict.citation_fits, verdict.numbers_ok)
    return [MARKS[verdict.correct], *(MARKS[check] for check in checks)]


def _question_row(record: JudgeRecord) -> list[str]:
    faithful = record.faithful
    if faithful is None:
        supported = "—"
    elif faithful.verdict is None:
        supported = faithful.status.value
    else:
        supported = f"{_supported(record)} / {record.claims}"
    reason = record.correct.verdict.reason if record.correct and record.correct.verdict else "—"
    return [record.id, *_correct_cells(record), supported, reason.replace("|", "/")]


def _findings(records: Sequence[JudgeRecord]) -> list[str]:
    unsupported = [
        f"- {record.id}, твердження {item.n}: {item.reason}"
        for record in records
        if record.faithful and record.faithful.verdict
        for item in record.faithful.verdict.verdicts
        if not item.supported
    ]
    not_judged = [
        f"- {record.id}: {judgement.status.value}: {judgement.error}"
        for record in records
        for judgement in _judgements(record)
        if judgement.status is not JudgeStatus.OK
    ]
    return [
        "## Твердження без опори в документах", "", *(unsupported or ["Немає."]), "",
        "## Не оцінено", "", *(not_judged or ["Немає."]), "",
    ]


def render_judge(judged: Judged) -> str:
    card, model = judged.card, judged.config.model
    correct_prompt, faithful_prompt = judged.prompts
    records = judged.records
    tokens_in = sum(j.input_tokens for r in records for j in _judgements(r))
    tokens_out = sum(j.output_tokens for r in records for j in _judgements(r))
    gate_rows = [
        [gate.name, f"{gate.measured:.0%}", f"≥ {gate.limit:.0%}", MARKS[gate.passed]]
        for gate in judged.gates
    ]
    sample = (
        [f"Вибірка `--limit {judged.limit}`: перші {judged.limit} питань, пороги не перевіряються.", ""]
        if judged.limit else []
    )
    return "\n".join([
        f"# Суддя: прогін {judged.run_dir.name}",
        "",
        (
            f"Суддя: {model.id} · effort {model.effort} · промпти: correct "
            f"{correct_prompt.version} ({correct_prompt.sha256[:8]}), faithful "
            f"{faithful_prompt.version} ({faithful_prompt.sha256[:8]})"
        ),
        "",
        *sample,
        (
            f"correct: {_ratio(card.correct)} · faithful: {_ratio(card.faithful)} · "
            f"не оцінено: {card.not_judged}"
        ),
        "",
        f"Токени in / out: {tokens_in} / {tokens_out} · вартість: ${run_cost(records, model):.4f}",
        "",
        *(_table(["Метрика", "Виміряно", "Вимога", "Пройдено"], gate_rows) + [""] if gate_rows else []),
        "## По питаннях",
        "",
        *_table(
            ["id", "correct", "висновок", "посилання", "числа", "faithful", "чому (correct)"],
            [_question_row(record) for record in records],
        ),
        "",
        *_findings(records),
    ])


def load_pairs(args: argparse.Namespace, run_dir: Path) -> list[tuple[AnswerQuestion, RagResult]]:
    """The questions to judge with their answers; refused before any paid call if one cannot be."""
    questions = load_answer_questions(args.questions, args.search_questions)
    records = read_jsonl(run_dir / ANSWERS_NAME, AnswerRecord)
    pairs = pair_with_questions(questions, records)[: args.limit]
    problems = []
    for question, result in pairs:
        if result.answer and result.answer.claims:
            try:
                faithful_turn(result)
            except ValueError as error:
                problems.append(f"{question.id}: {error}")
    if problems:
        raise ValueError("; ".join(problems))
    return pairs


def _real_judge(config: JudgeConfig, correct: Prompt, faithful: Prompt) -> Judging:
    return Judge(config.model, correct, faithful)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", type=Path, help="a run's folder; the newest by default")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--questions", type=Path, default=ANSWER_QUESTIONS_PATH)
    parser.add_argument("--search-questions", type=Path, default=SEARCH_QUESTIONS_PATH)
    parser.add_argument("--config", type=Path, default=JUDGE_CONFIG_PATH)
    parser.add_argument("--limit", type=int, help="judge only the first N questions")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main(argv: list[str] | None = None, make_judge: MakeJudge = _real_judge) -> int:
    args = parse_args(argv)
    try:
        config = load_judge_config(args.config)
        prompts = (load_prompt(config.prompts.correct), load_prompt(config.prompts.faithful))
        run_dir: Path = args.run or latest_run(args.results_dir)
        pairs = load_pairs(args, run_dir)
    except SETUP_ERRORS as error:
        print(f"cannot judge: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    judge = make_judge(config, *prompts)
    records = []
    for n, (question, result) in enumerate(pairs, start=1):
        records.append(judge_pair(judge, question, result))
        row = _question_row(records[-1])
        print(f"{n}/{len(pairs)} {question.id}: correct {row[1]} · faithful {row[5]}")
    judged = Judged(run_dir, config, prompts, args.limit, tuple(records))
    lines = "".join(f"{record.model_dump_json()}\n" for record in records)
    (run_dir / JUDGEMENTS_NAME).write_text(lines, encoding="utf-8")
    (run_dir / JUDGE_MD_NAME).write_text(render_judge(judged), encoding="utf-8")
    card = judged.card
    print(f"correct {_ratio(card.correct)} · faithful {_ratio(card.faithful)} · "
          f"not judged {card.not_judged} · ${run_cost(records, config.model):.4f}")
    for gate in judged.gates:
        print(f"{MARKS[gate.passed]} {gate.name}: {gate.measured:.2f} (limit {gate.limit:g})")
    if judged.limit:
        print(f"a sample of {judged.limit}: thresholds not checked")
    print(f"judgements: {run_dir / JUDGE_MD_NAME}")
    return 0 if all(gate.passed for gate in judged.gates) else 1


if __name__ == "__main__":
    sys.exit(main())
