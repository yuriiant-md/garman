"""Measure stage 2 search on the questions with known answers: recall@k and MRR per search mode.

Every question of data/rag/search_questions.yaml is searched in three modes (vector, hybrid,
hybrid + rerank) in both collections: `chunks` (structural) and `chunks_fixed` (fixed-size
windows). A question about a contract searches only that contract; it is searched once more with
laws and decisions allowed, to show what court decisions quoting other contracts cost. The report
and every rank go to a new folder results/stage2/search/<time>/, so no run overwrites another.

Qdrant must be running with a fresh index: docker compose up -d qdrant, then scripts.index_rag.
The first run downloads the 0.57 GB reranker. About two minutes, most of it the reranker:
caffeinate -i uv run python -m scripts.eval_search
"""

import argparse
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient

from scripts.chunk_rag import (
    CHUNKS_PATH,
    DOCUMENTS_PATH,
    MANIFEST_PATH,
    SOURCES_PATH,
    join_corpus,
)
from scripts.index_rag import (
    FIXED_COLLECTION,
    QDRANT_URL,
    STRUCTURAL_COLLECTION,
    fixed_chunks,
    reachable_client,
)
from src.rag.chunking import Chunk
from src.rag.documents import print_problems, read_documents, read_jsonl, read_manifest
from src.rag.embedding import Embedder, bge_m3
from src.rag.index import StaleIndexError, index_stamp
from src.rag.search import (
    RERANKER_NAME,
    Reranker,
    Searcher,
    SearchFilter,
    SearchMode,
    bge_reranker,
)
from src.rag.search_eval import (
    QuestionKind,
    SearchQuestion,
    check_questions,
    first_relevant_rank,
    load_questions,
    mean_reciprocal_rank,
    recall_at,
    with_laws_and_decisions,
)
from src.rag.sources import load_sources

QUESTIONS_PATH = Path("data/rag/search_questions.yaml")
RESULTS_DIR = Path("results/stage2/search")
RUN_STAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"  # as stage 1: no colons in folder names
REPORT_NAME = "search_metrics.md"
RANKS_NAME = "ranks.jsonl"
K_VALUES = (1, 3, 5, 10)
SEARCH_K = max(K_VALUES)  # hits read per search: a right chunk below 10 counts as missed
REPORT_K = 5  # the k of the comparison tables
AS_WRITTEN = "as_written"  # the question's own filter
WIDENED = "widened"  # a contract question with laws and decisions allowed too
NOT_FOUND = "—"


@dataclass(frozen=True)
class Task:
    collection: str
    mode: SearchMode
    variant: str
    question: SearchQuestion
    search_filter: SearchFilter


class Outcome(BaseModel):
    """One search of one question: where the first right chunk stood."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection: str
    mode: SearchMode
    variant: str
    question_id: str
    kind: QuestionKind
    rank: int | None  # None: not among the first SEARCH_K hits
    seconds: float
    top: tuple[str, ...]  # chunk ids of the first hits, to see what came instead


def plan_tasks(questions: Sequence[SearchQuestion]) -> tuple[Task, ...]:
    as_written = [
        Task(collection, mode, AS_WRITTEN, question, question.filter)
        for collection in (STRUCTURAL_COLLECTION, FIXED_COLLECTION)
        for mode in SearchMode
        for question in questions
    ]
    widened = [
        Task(STRUCTURAL_COLLECTION, mode, WIDENED, question, wider)
        for mode in SearchMode
        for question in questions
        if (wider := with_laws_and_decisions(question.filter)) != question.filter
    ]
    return tuple(as_written + widened)


def run_tasks(
    tasks: Sequence[Task],
    searchers: Mapping[str, Searcher],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[Outcome, ...]:
    outcomes = []
    width = len(str(len(tasks)))
    for number, task in enumerate(tasks, start=1):
        started = clock()
        hits = searchers[task.collection].search(
            task.question.question, task.mode, k=SEARCH_K, search_filter=task.search_filter
        )
        item = Outcome(
            collection=task.collection,
            mode=task.mode,
            variant=task.variant,
            question_id=task.question.id,
            kind=task.question.kind,
            rank=first_relevant_rank(hits, task.question),
            seconds=clock() - started,
            top=tuple(hit.chunk.chunk_id for hit in hits[:REPORT_K]),
        )
        print(
            f"[{number:>{width}}/{len(tasks)}] {task.collection:<12} {task.mode:<13} "
            f"{task.variant:<10} {task.question.id}: rank {_rank_text(item.rank)} "
            f"({item.seconds:.2f} s)",
            flush=True,
        )
        outcomes.append(item)
    return tuple(outcomes)


def _rank_text(rank: int | None) -> str:
    return NOT_FOUND if rank is None else str(rank)


@dataclass(frozen=True)
class Selection:
    """Which outcomes a table row is about; by default every question as written, structural."""

    collection: str = STRUCTURAL_COLLECTION
    variant: str = AS_WRITTEN
    question_ids: tuple[str, ...] | None = None  # None: every question searched so


ALL = Selection()


@dataclass(frozen=True)
class Results:
    outcomes: Sequence[Outcome]

    def chosen(self, mode: SearchMode, selection: Selection = ALL) -> list[Outcome]:
        found = {
            item.question_id: item
            for item in self.outcomes
            if (item.collection, item.mode, item.variant)
            == (selection.collection, mode, selection.variant)
        }
        wanted = tuple(found) if selection.question_ids is None else selection.question_ids
        return [found[question_id] for question_id in wanted]

    def ranks(self, mode: SearchMode, selection: Selection = ALL) -> list[int | None]:
        return [item.rank for item in self.chosen(mode, selection)]

    def mean_seconds(self, mode: SearchMode) -> float:
        times = [item.seconds for item in self.chosen(mode)]
        return sum(times) / len(times)


def _number(value: float) -> str:
    return f"{value:.2f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [f"| {' | '.join(header)} |", f"|{' --- |' * len(header)}"]
    return lines + [f"| {' | '.join(row)} |" for row in rows]


def _modes_table(results: Results) -> list[str]:
    header = ["Режим", *(f"R@{k}" for k in K_VALUES), "MRR", "Середній час, с"]
    rows = []
    for mode in SearchMode:
        ranks = results.ranks(mode)
        recalls = [_number(recall_at(ranks, k)) for k in K_VALUES]
        mrr = _number(mean_reciprocal_rank(ranks))
        rows.append([mode.value, *recalls, mrr, _number(results.mean_seconds(mode))])
    return _table(header, rows)


def _kinds_table(results: Results, questions: Sequence[SearchQuestion]) -> list[str]:
    rows = []
    for kind in QuestionKind:
        ids = tuple(question.id for question in questions if question.kind == kind)
        if ids:
            only = Selection(question_ids=ids)
            ranks = [results.ranks(mode, only) for mode in SearchMode]
            recalls = [_number(recall_at(mode_ranks, REPORT_K)) for mode_ranks in ranks]
            rows.append([kind.value, str(len(ids)), *recalls])
    return _table(["Тип", "Питань", *(mode.value for mode in SearchMode)], rows)


def _pair_table(
    results: Results, labels: tuple[str, str], first: Selection, second: Selection
) -> list[str]:
    """R@5 and MRR of two configurations side by side, a row per mode."""
    header = [
        "Режим",
        f"R@{REPORT_K} {labels[0]}",
        f"R@{REPORT_K} {labels[1]}",
        f"MRR {labels[0]}",
        f"MRR {labels[1]}",
    ]
    rows = []
    for mode in SearchMode:
        left, right = results.ranks(mode, first), results.ranks(mode, second)
        rows.append(
            [
                mode.value,
                _number(recall_at(left, REPORT_K)),
                _number(recall_at(right, REPORT_K)),
                _number(mean_reciprocal_rank(left)),
                _number(mean_reciprocal_rank(right)),
            ]
        )
    return _table(header, rows)


def _contracts_table(results: Results, contract_ids: tuple[str, ...]) -> list[str]:
    """Contract questions searched in the contract alone and with laws and decisions."""
    if not contract_ids:
        return ["Питань про договір немає."]
    return _pair_table(
        results,
        ("лише договір", "+ закони й рішення"),
        Selection(question_ids=contract_ids),
        Selection(variant=WIDENED, question_ids=contract_ids),
    )


def _ranks_table(results: Results, questions: Sequence[SearchQuestion]) -> list[str]:
    header = [
        "Питання",
        "Тип",
        *(mode.value for mode in SearchMode),
        *(f"вікна: {mode.value}" for mode in SearchMode),
    ]
    rows = []
    for question in questions:
        this = Selection(question_ids=(question.id,))
        this_fixed = Selection(collection=FIXED_COLLECTION, question_ids=(question.id,))
        structural = [results.ranks(mode, this)[0] for mode in SearchMode]
        fixed = [results.ranks(mode, this_fixed)[0] for mode in SearchMode]
        rows.append([question.id, question.kind.value, *map(_rank_text, structural + fixed)])
    return _table(header, rows)


def _missed(results: Results, questions: Sequence[SearchQuestion]) -> list[str]:
    missed = [
        question
        for question in questions
        if all(
            rank is None or rank > REPORT_K
            for mode in SearchMode
            for rank in results.ranks(mode, Selection(question_ids=(question.id,)))
        )
    ]
    if not missed:
        return ["Немає: кожне питання хоч один режим знайшов у перших 5."]
    return [
        f"- **{question.id}** ({question.kind.value}): {question.question}"
        + (f" — {question.note}" if question.note else "")
        for question in missed
    ]


def render_report(
    questions: Sequence[SearchQuestion], outcomes: Sequence[Outcome], header: Sequence[str]
) -> str:
    results = Results(outcomes)
    contract_ids = tuple(
        question.id
        for question in questions
        if with_laws_and_decisions(question.filter) != question.filter
    )
    sections = [
        ["# Якість пошуку, етап 2 (Е2.4)", "", *header],
        [
            "## 1. Режими пошуку (структурні чанки, фільтри як у питаннях)",
            "",
            *_modes_table(results),
        ],
        [f"## 2. Recall@{REPORT_K} за типом питання", "", *_kinds_table(results, questions)],
        [
            "## 3. Структурні чанки проти вікон фіксованого розміру",
            "",
            *_pair_table(
                results, ("структурні", "вікна"), ALL, Selection(collection=FIXED_COLLECTION)
            ),
        ],
        [
            (
                f"## 4. Питання про договір ({len(contract_ids)}): лише договір проти "
                "договір + закони й рішення"
            ),
            "",
            *_contracts_table(results, contract_ids),
        ],
        [
            "## 5. Місце першого правильного чанка",
            "",
            f"«{NOT_FOUND}» означає: правильного чанка немає в перших {SEARCH_K}.",
            "",
            *_ranks_table(results, questions),
        ],
        [f"## 6. Не знайдено в перших {REPORT_K} жодним режимом", "", *_missed(results, questions)],
    ]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"


def write_run(
    results_dir: Path, started: datetime, report: str, outcomes: Sequence[Outcome]
) -> Path:
    """A new folder for this run; an existing one is an error, never overwritten."""
    run_dir = results_dir / started.strftime(RUN_STAMP_FORMAT)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / REPORT_NAME).write_text(report, encoding="utf-8")
    lines = "".join(f"{item.model_dump_json()}\n" for item in outcomes)
    (run_dir / RANKS_NAME).write_text(lines, encoding="utf-8")
    return run_dir


def run(
    client: QdrantClient,
    questions: Sequence[SearchQuestion],
    structural: Sequence[Chunk],
    fixed: Sequence[Chunk],
    embedder: Embedder,
    reranker: Reranker,
    *,
    reranker_name: str,
    results_dir: Path,
    started: datetime,
) -> Path:
    """Search every planned task, then write the report; a stale index raises StaleIndexError."""
    searchers = {
        collection: Searcher(client, collection, embedder, index_stamp(chunks, embedder), reranker)
        for collection, chunks in ((STRUCTURAL_COLLECTION, structural), (FIXED_COLLECTION, fixed))
    }
    outcomes = run_tasks(plan_tasks(questions), searchers)
    header = [
        (
            f"Прогін: {started:%Y-%m-%d %H:%M} · питань: {len(questions)} · "
            f"embedder: {embedder.model_name} · reranker: {reranker_name}"
        ),
        "",
        (
            f"Колекції: {STRUCTURAL_COLLECTION} — {len(structural)} структурних чанків; "
            f"{FIXED_COLLECTION} — {len(fixed)} вікон фіксованого розміру."
        ),
        "",
        (
            "Правильний чанк — із doc_id питання і дослівно містить його evidence "
            f"({QUESTIONS_PATH}). R@k — частка питань із правильним чанком у перших k; "
            "MRR — середнє 1/місце першого правильного, промах = 0."
        ),
    ]
    return write_run(results_dir, started, render_report(questions, outcomes, header), outcomes)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", QDRANT_URL))
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--documents", type=Path, default=DOCUMENTS_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--sources", type=Path, default=SOURCES_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    questions = load_questions(args.questions)
    structural = read_jsonl(args.chunks, Chunk)
    if problems := check_questions(questions, structural):
        return print_problems(problems)
    joined, problems = join_corpus(
        read_documents(args.documents), load_sources(args.sources), read_manifest(args.manifest)
    )
    if problems:
        return print_problems(problems)
    client = reachable_client(args.qdrant_url)
    if client is None:
        return 1
    print("loading bge-m3 and the reranker (the first run downloads 0.57 GB) …", flush=True)
    embedder, reranker = bge_m3(), bge_reranker()
    try:
        run_dir = run(
            client,
            questions,
            structural,
            fixed_chunks(joined),
            embedder,
            reranker,
            reranker_name=RERANKER_NAME,
            results_dir=args.results_dir,
            started=datetime.now(),
        )
    except StaleIndexError as error:
        print(error)
        return 1
    print(f"report: {run_dir / REPORT_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
