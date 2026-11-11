"""Score a saved run of the stage 2 answers by rules: no model, no search, no key, free.

Reads answers.jsonl of a run of scripts/ask_rag.py, scores it against the current question set
and the thresholds of configs/eval.yaml, and writes scores.md next to it.
    uv run python -m scripts.score_answers                 # the newest run
    uv run python -m scripts.score_answers results/stage2/answers/2026-11-09T08-46-28

Exit code: 0 — every threshold is met; 1 — some is not; 2 — the run cannot be scored (a run that
does not cover the current questions is refused: its numbers would say nothing about them).
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from scripts.ask_rag import (
    ANSWERS_NAME,
    MARKS,
    RESULTS_DIR,
    SETUP_ERRORS,
    AnswerRecord,
    _table,
)
from scripts.chunk_rag import CHUNKS_PATH
from src.rag.answer_eval import (
    ANSWER_QUESTIONS_PATH,
    FACT_ALTERNATIVE_SEPARATOR,
    SEARCH_QUESTIONS_PATH,
    AnswerQuestion,
    load_answer_questions,
)
from src.rag.answer_scoring import (
    EVAL_CONFIG_PATH,
    Gate,
    QuestionScore,
    Ratio,
    Scorecard,
    check_thresholds,
    load_thresholds,
    score_run,
)
from src.rag.answerer import RagResult
from src.rag.chunking import Chunk
from src.rag.documents import read_jsonl

SCORES_NAME = "scores.md"
FACT_ALTERNATIVE_SHOWN = " / "  # «|» would split a cell of a markdown table
GATE_TITLES = {
    "source": "Правильне джерело",
    "key_facts": "Ключові факти",
    "not_in_documents": "Чесне «у документах немає»",
    "forbidden_facts": "Заборонені факти",
    "repealed_as_in_force": "ГК як чинний",
}


@dataclass(frozen=True)
class Scored:
    run_dir: Path
    scorecard: Scorecard
    gates: tuple[Gate, ...]


def latest_run(results_dir: Path) -> Path:
    """The newest run: the folders are named by their start time."""
    runs = sorted(path.parent for path in results_dir.glob(f"*/{ANSWERS_NAME}"))
    if not runs:
        raise ValueError(f"no run with {ANSWERS_NAME} in {results_dir}")
    return runs[-1]


def pair_with_questions(
    questions: Sequence[AnswerQuestion], records: Sequence[AnswerRecord]
) -> list[tuple[AnswerQuestion, RagResult]]:
    """The current questions with their answers; a run that did not ask them all is refused."""
    by_id = {record.question.id: record for record in records}
    if missing := [question.id for question in questions if question.id not in by_id]:
        raise ValueError(f"the run did not ask: {', '.join(missing)}; make a new run")
    if changed := [
        question.id
        for question in questions
        if (by_id[question.id].question.question, by_id[question.id].question.contract_id)
        != (question.question, question.contract_id)
    ]:
        raise ValueError(f"the run asked another text for: {', '.join(changed)}; make a new run")
    return [(question, by_id[question.id].result) for question in questions]


def score(args: argparse.Namespace) -> Scored:
    thresholds = load_thresholds(args.thresholds)
    questions = load_answer_questions(args.questions, args.search_questions)
    run_dir: Path = args.run or latest_run(args.results_dir)
    records = read_jsonl(run_dir / ANSWERS_NAME, AnswerRecord)
    chunks = {chunk.chunk_id: chunk for chunk in read_jsonl(args.chunks, Chunk)}
    scorecard = score_run(pair_with_questions(questions, records), chunks)
    return Scored(run_dir, scorecard, check_thresholds(scorecard, thresholds))


def _ratio(ratio: Ratio) -> str:
    return f"{ratio.hit} з {ratio.total} ({ratio.rate:.0%})"


def _ratios(scorecard: Scorecard) -> dict[str, Ratio]:
    return {
        "source": scorecard.source,
        "key_facts": scorecard.key_facts,
        "not_in_documents": scorecard.not_in_documents,
    }


def _gate_row(gate: Gate, scorecard: Scorecard) -> list[str]:
    ratio = _ratios(scorecard).get(gate.name)
    measured = _ratio(ratio) if ratio else str(int(gate.measured))
    required = f"≥ {gate.limit:.0%}" if ratio else f"≤ {int(gate.limit)}"
    return [GATE_TITLES[gate.name], measured, required, MARKS[gate.passed]]


def _shown_facts(facts: Sequence[str]) -> str:
    shown = [fact.replace(FACT_ALTERNATIVE_SEPARATOR, FACT_ALTERNATIVE_SHOWN) for fact in facts]
    return "; ".join(shown) or "—"


def _question_row(score_: QuestionScore) -> list[str]:
    found = score_.key_facts_total - len(score_.missing_key_facts)
    return [
        score_.id,
        score_.kind,
        MARKS[score_.outcome_ok],
        MARKS[score_.cites_evidence],
        f"{found} / {score_.key_facts_total}" if score_.key_facts_total else "—",
        _shown_facts(score_.missing_key_facts),
        _shown_facts(score_.forbidden_found),
    ]


def render_scores(scored: Scored) -> str:
    scorecard = scored.scorecard
    gate_rows = [_gate_row(gate, scorecard) for gate in scored.gates]
    question_header = ["id", "тип", "outcome", "evidence", "факти", "бракує", "заборонені"]
    problems = [
        f"- {score_.id}: {problem}"
        for score_ in scorecard.questions
        for problem in score_.repealed_problems
    ]
    return "\n".join(
        [
            f"# Оцінка прогону {scored.run_dir.name}",
            "",
            *_table(["Метрика", "Виміряно", "Вимога", "Пройдено"], gate_rows),
            "",
            f"Outcome як очікували: {_ratio(scorecard.outcome)} (без порога).",
            "",
            "## По питаннях",
            "",
            *_table(question_header, [_question_row(item) for item in scorecard.questions]),
            "",
            "## ГК як чинний: порушення",
            "",
            *(problems or ["Немає."]),
            "",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", type=Path, help="a run's folder; the newest by default")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--questions", type=Path, default=ANSWER_QUESTIONS_PATH)
    parser.add_argument("--search-questions", type=Path, default=SEARCH_QUESTIONS_PATH)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--thresholds", type=Path, default=EVAL_CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        scored = score(args)
        (scored.run_dir / SCORES_NAME).write_text(render_scores(scored), encoding="utf-8")
    except SETUP_ERRORS as error:
        print(f"cannot score: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    for gate in scored.gates:
        print(f"{MARKS[gate.passed]} {gate.name}: {gate.measured:.2f} (limit {gate.limit:g})")
    print(f"scores: {scored.run_dir / SCORES_NAME}")
    return 0 if all(gate.passed for gate in scored.gates) else 1


if __name__ == "__main__":
    sys.exit(main())
