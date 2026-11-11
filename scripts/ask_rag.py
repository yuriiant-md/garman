"""Ask the stage 2 answerer the questions of data/rag/answer_questions.yaml and write a report.

Each question is searched (the question's contract, the laws and the decisions apart), and
Claude answers from what was found; with nothing over the threshold the answer is «немає» and
the model is not asked (src/rag/answerer.py). The answers and the report go to a new folder
results/stage2/answers/<time>/: answers.jsonl, one question and its RagResult per line, and
report.md with the summary, every question and every answer to read.

Paid: Claude Haiku 5.5, about $0.01 for 20 questions. Qdrant must be running with a fresh index:
    caffeinate -i uv run --env-file .env python -m scripts.ask_rag
    caffeinate -i uv run --env-file .env python -m scripts.ask_rag --ids q01,t05
    caffeinate -i uv run --env-file .env python -m scripts.ask_rag --prompt prompts/answer_v2.md
"""

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, ConfigDict

from scripts.chunk_rag import CHUNKS_PATH
from scripts.index_rag import QDRANT_URL, STRUCTURAL_COLLECTION, reachable_client
from src.extraction.anthropic_extractor import AnthropicConfig
from src.extraction.prompts import Prompt, load_prompt
from src.extraction.run import (
    ANTHROPIC,
    load_anthropic_config,
    load_prices,
    model_provider,
)
from src.rag.answer_config import ANSWER_CONFIG_PATH, AnswerConfig, load_answer_config
from src.rag.answer_eval import (
    ANSWER_QUESTIONS_PATH,
    SEARCH_QUESTIONS_PATH,
    AnswerQuestion,
    Summary,
    cites_expected,
    load_answer_questions,
    outcome_matches,
    summarize,
)
from src.rag.answerer import Answerer, Outcome, RagResult
from src.rag.chunking import Chunk
from src.rag.documents import read_jsonl
from src.rag.embedding import bge_m3
from src.rag.index import StaleIndexError, index_stamp
from src.rag.search import RERANKER_NAME, Searcher, bge_reranker

API_KEY = "ANTHROPIC_API_KEY"
RESULTS_DIR = Path("results/stage2/answers")
RUN_STAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"  # as stage 1: no colons in folder names
REPORT_NAME = "report.md"
ANSWERS_NAME = "answers.jsonl"
COST_BATCH = 1000  # cost is shown per 1000 questions, as stage 1 shows it per 1000 documents
MARKS = {True: "✓", False: "✗", None: "—"}
# what a missing or wrong config, prompt or question file raises while the run is set up
SETUP_ERRORS = (OSError, ValueError, KeyError, TypeError, yaml.YAMLError)


class AnswerRecord(BaseModel):
    """One line of answers.jsonl: what was asked and expected, and what came back."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: AnswerQuestion
    result: RagResult


class Asks(Protocol):
    def ask(self, question: str, contract_id: str | None = None) -> RagResult: ...


@dataclass(frozen=True)
class RunSetup:
    """Everything the run reads from files, known before Qdrant and the models are touched."""

    config: AnswerConfig
    prompt: Prompt
    model: AnthropicConfig
    price_input: float
    price_output: float
    questions: tuple[AnswerQuestion, ...]


def select_questions(
    questions: Sequence[AnswerQuestion], ids: str | None
) -> tuple[AnswerQuestion, ...]:
    """The questions named in a comma-separated list, in the order of the set; all without one."""
    if ids is None:
        return tuple(questions)
    wanted = {qid.strip() for qid in ids.split(",") if qid.strip()}
    if not wanted:
        raise ValueError(f"--ids {ids!r} names no questions")
    if unknown := sorted(wanted - {question.id for question in questions}):
        raise ValueError(f"no such questions: {', '.join(unknown)}")
    return tuple(question for question in questions if question.id in wanted)


def run_questions(answerer: Asks, questions: Sequence[AnswerQuestion]) -> list[AnswerRecord]:
    done = []
    for number, question in enumerate(questions, start=1):
        result = answerer.ask(question.question, question.contract_id)
        print(f"{number}/{len(questions)} {question.id}: {result.outcome}", flush=True)
        done.append(AnswerRecord(question=question, result=result))
    return done


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [header, ["---"] * len(header), *rows]
    return ["| " + " | ".join(line) + " |" for line in lines]


def _cost_per_batch(summary: Summary) -> str:
    """Over answered questions only, as cost_per_1000_docs: an `error` result used no tokens."""
    answered = summary.questions - summary.outcomes.get(Outcome.ERROR, 0)
    return f"{summary.cost_usd / answered * COST_BATCH:.2f}" if answered else "—"


def _summary_lines(summary: Summary) -> list[str]:
    row = [
        str(summary.questions),
        f"{summary.matched} з {summary.questions}",
        f"{summary.cite_hits} з {summary.cite_checked}",
        str(summary.model_calls),
        f"{summary.input_tokens} / {summary.output_tokens}",
        f"{summary.cost_usd:.4f}",
        _cost_per_batch(summary),
        f"{summary.latency_p50_s:.1f} / {summary.latency_p95_s:.1f}",
    ]
    header = [
        "Питань", "Outcome як очікували", "Цитує evidence", "Викликів моделі",
        "Токени in / out", "Вартість, $", f"На {COST_BATCH} питань, $", "Latency p50 / p95, с",
    ]
    counts = " · ".join(f"{outcome}: {summary.outcomes.get(outcome, 0)}" for outcome in Outcome)
    return [*_table(header, [row]), "", counts]


def _question_row(record: AnswerRecord) -> list[str]:
    question, result = record.question, record.result
    return [
        question.id,
        question.kind,
        question.expect,
        result.outcome,
        MARKS[outcome_matches(result, question)],
        MARKS[cites_expected(result, question)],
        ", ".join(result.cited) or "—",
        str(result.context_tokens),
        str(result.attempts),
        f"{result.latency_s:.1f}",
    ]


def _answer_lines(record: AnswerRecord) -> list[str]:
    question, result = record.question, record.result
    lines = [f"### {question.id} — {question.question}", ""]
    if question.note:
        lines += [f"Очікували: {question.note}", ""]
    lines += [result.text or f"`{result.outcome}`: {result.error}", ""]
    if not result.model_called:
        lines.append(f"_модель не питали, фрагментів нижче порогу: {len(result.below_threshold)}_")
    else:
        context = ", ".join(f"{label} {chunk_id}" for label, chunk_id in result.sources.items())
        lines.append(f"_контекст: {context}_")
    return [*lines, ""]


def render_report(
    records: Sequence[AnswerRecord], summary: Summary, header: Sequence[str]
) -> str:
    question_header = [
        "id", "тип", "очікували", "вийшло", "outcome", "evidence", "процитовано",
        "контекст, ток.", "спроб", "с",
    ]
    lines = [
        *header,
        "",
        "## Підсумок",
        "",
        *_summary_lines(summary),
        "",
        "## По питаннях",
        "",
        *_table(question_header, [_question_row(record) for record in records]),
        "",
        "## Відповіді",
        "",
    ]
    for record in records:
        lines += _answer_lines(record)
    return "\n".join(lines)


def write_run(
    results_dir: Path, started: datetime, report: str, records: Sequence[AnswerRecord]
) -> Path:
    """A new folder for this run; an existing one is an error, never overwritten."""
    run_dir = results_dir / started.strftime(RUN_STAMP_FORMAT)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / REPORT_NAME).write_text(report, encoding="utf-8")
    lines = "".join(f"{record.model_dump_json()}\n" for record in records)
    (run_dir / ANSWERS_NAME).write_text(lines, encoding="utf-8")
    return run_dir


def report_header(
    config: AnswerConfig, prompt: Prompt, model: str, started: datetime, count: int
) -> list[str]:
    context = config.context
    return [
        (
            f"Прогін: {started:%Y-%m-%d %H:%M} · питань: {count} · модель: {model} · "
            f"промпт: {prompt.version} ({prompt.sha256[:8]}) · reranker: {RERANKER_NAME}"
        ),
        "",
        (
            f"Контекст: договір {context.contract_k} · закони {context.law_k} · постанови "
            f"{context.decision_k} · до {context.max_tokens} токенів · поріг {context.min_score}."
        ),
        "",
        (
            "outcome ✓ — вийшло те, що очікували. evidence ✓ — відповідь цитує фрагмент із "
            "доказом (`should_cite`); «—» — для питання такого фрагмента немає."
        ),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ids", help="comma-separated question ids, e.g. q01,t05; all by default")
    parser.add_argument("--config", type=Path, default=ANSWER_CONFIG_PATH)
    parser.add_argument("--prompt", type=Path, help="another prompt than the config's one")
    parser.add_argument("--questions", type=Path, default=ANSWER_QUESTIONS_PATH)
    parser.add_argument("--search-questions", type=Path, default=SEARCH_QUESTIONS_PATH)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", QDRANT_URL))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args(argv)


def prompt_path(args: argparse.Namespace, configured: Path) -> Path:
    """The prompt named with --prompt, to compare versions; else the config's."""
    return args.prompt or configured


def load_setup(args: argparse.Namespace) -> RunSetup:
    config = load_answer_config(args.config)
    if model_provider(config.model) != ANTHROPIC:
        raise ValueError(f"{args.config}: model {config.model} is not an anthropic model")
    price_input, price_output = load_prices(section=config.model)
    return RunSetup(
        config=config,
        prompt=load_prompt(prompt_path(args, config.prompt)),
        model=load_anthropic_config(section=config.model),
        price_input=price_input,
        price_output=price_output,
        questions=select_questions(
            load_answer_questions(args.questions, args.search_questions), args.ids
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.environ.get(API_KEY):
        print(
            f"{API_KEY} is not set: put it in .env and run with "
            "`uv run --env-file .env python -m scripts.ask_rag`.",
            file=sys.stderr,
        )
        return 2
    try:
        setup = load_setup(args)
    except SETUP_ERRORS as error:
        print(f"cannot start: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    client = reachable_client(args.qdrant_url)
    if client is None:
        return 1
    chunks = read_jsonl(args.chunks, Chunk)
    print("loading bge-m3 and the reranker …", flush=True)
    embedder = bge_m3()
    try:
        searcher = Searcher(
            client, STRUCTURAL_COLLECTION, embedder, index_stamp(chunks, embedder), bge_reranker()
        )
    except StaleIndexError as error:
        print(error, file=sys.stderr)
        return 1
    started = datetime.now()
    answerer = Answerer(searcher, setup.config.context, setup.model, setup.prompt)
    records = run_questions(answerer, setup.questions)
    summary = summarize(
        [(record.question, record.result) for record in records],
        price_input=setup.price_input,
        price_output=setup.price_output,
    )
    header = report_header(
        setup.config, setup.prompt, setup.model.model, started, len(setup.questions)
    )
    run_dir = write_run(args.results_dir, started, render_report(records, summary, header), records)
    print(f"{summary.matched} of {summary.questions} as expected · ${summary.cost_usd:.4f}")
    print(f"report: {run_dir / REPORT_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
