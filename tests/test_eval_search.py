import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from scripts import eval_search
from scripts.eval_search import AS_WRITTEN, WIDENED, Outcome, plan_tasks, render_report
from scripts.index_rag import FIXED_COLLECTION, STRUCTURAL_COLLECTION
from src.rag.chunking import Chunk
from src.rag.embedding import Progress, Vector
from src.rag.index import build_index, index_stamp
from src.rag.search import Searcher, SearchFilter, SearchMode
from src.rag.search_eval import QuestionKind, SearchQuestion
from src.rag.sources import DocType

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
STARTED = datetime(2026, 11, 6, 14, 3, 0)


class PenaltyEmbedder:
    model_name = "fake-embedder"
    dimension = 2

    def embed(
        self, texts: Sequence[str], on_progress: Progress | None = None
    ) -> tuple[Vector, ...]:
        return tuple((1.0, 0.0) if "пен" in text.lower() else (0.0, 1.0) for text in texts)


class SilentReranker:
    """Scores every text the same, so the hybrid order stays."""

    def rerank(self, query: str, texts: Sequence[str]) -> tuple[float, ...]:
        return (0.0,) * len(texts)


def chunk(chunk_id: str, text: str, doc_type: DocType = DocType.CONTRACT) -> Chunk:
    doc_id = chunk_id.split("#")[0]
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_type=doc_type,
        title=doc_id,
        short_title=doc_id,
        text_sha256="0" * 64,
        text=text,
    )


CHUNKS = (
    chunk("own_003#001", "6.1. Покупець сплачує 86% вартості Товару."),
    chunk("own_003#002", "7.2. Покупець сплачує пеню за кожний день прострочення."),
    chunk("cc#625", "Боржник сплачує три проценти річних.", DocType.LAW),
    chunk("vs#001", "Суд стягнув пеню з покупця.", DocType.COURT_DECISION),
)
CONTRACT_ONLY = SearchFilter(contract_id="own_003", doc_types=frozenset({DocType.CONTRACT}))
CONTRACT_QUESTION = SearchQuestion(
    id="q01",
    question="Яку пеню платить покупець?",
    kind=QuestionKind.EXACT,
    filter=CONTRACT_ONLY,
    doc_id="own_003",
    evidence="сплачує пеню",
    chunk_ids=("own_003#002",),
)
LAW_QUESTION = SearchQuestion(
    id="q02",
    question="Скільки процентів річних?",
    kind=QuestionKind.PLAIN,
    filter=SearchFilter(doc_types=frozenset({DocType.LAW})),
    doc_id="cc",
    evidence="три проценти річних",
    chunk_ids=("cc#625",),
)
QUESTIONS = (CONTRACT_QUESTION, LAW_QUESTION)


@pytest.fixture
def client() -> QdrantClient:
    client = QdrantClient(":memory:")
    for collection in (STRUCTURAL_COLLECTION, FIXED_COLLECTION):
        build_index(client, collection, CHUNKS, PenaltyEmbedder())
    return client


def searchers(client: QdrantClient) -> dict[str, Searcher]:
    embedder = PenaltyEmbedder()
    stamp = index_stamp(CHUNKS, embedder)
    return {
        collection: Searcher(client, collection, embedder, stamp, SilentReranker())
        for collection in (STRUCTURAL_COLLECTION, FIXED_COLLECTION)
    }


type Key = tuple[str, SearchMode, str, str]  # collection, mode, variant, question id


def outcomes_with(
    ranks: dict[Key, int | None], questions: Sequence[SearchQuestion] = QUESTIONS
) -> tuple[Outcome, ...]:
    """An outcome for every planned search: rank 1 unless `ranks` says otherwise."""
    return tuple(
        Outcome(
            collection=task.collection,
            mode=task.mode,
            variant=task.variant,
            question_id=task.question.id,
            kind=task.question.kind,
            rank=ranks.get((task.collection, task.mode, task.variant, task.question.id), 1),
            seconds=0.5,
            top=(),
        )
        for task in plan_tasks(questions)
    )


def test_every_question_runs_in_both_collections_and_contracts_also_widened() -> None:
    tasks = plan_tasks(QUESTIONS)

    as_written = [task for task in tasks if task.variant == AS_WRITTEN]
    widened = [task for task in tasks if task.variant == WIDENED]
    assert len(as_written) == 2 * len(SearchMode) * len(QUESTIONS)
    assert len(widened) == len(SearchMode)
    assert {task.question.id for task in widened} == {"q01"}
    assert {task.collection for task in widened} == {STRUCTURAL_COLLECTION}
    assert all(task.search_filter == SearchFilter(contract_id="own_003") for task in widened)


def test_run_tasks_finds_the_rank_of_the_right_chunk_and_prints_progress(
    client: QdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = plan_tasks(QUESTIONS)

    outcomes = eval_search.run_tasks(tasks, searchers(client))

    assert len(outcomes) == len(tasks)
    as_written = [item for item in outcomes if item.variant == AS_WRITTEN]
    assert all(item.rank == 1 for item in as_written)
    widened = [item for item in outcomes if item.variant == WIDENED]
    assert all(item.rank in (1, 2) for item in widened)  # the decision ties with the contract
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("[ 1/15] chunks")
    assert lines[-1].startswith("[15/15] chunks")


def test_the_report_tables_hold_recall_mrr_and_the_ranks() -> None:
    structural = STRUCTURAL_COLLECTION
    outcomes = outcomes_with(
        {
            (structural, SearchMode.VECTOR, AS_WRITTEN, "q02"): None,
            (structural, SearchMode.HYBRID, AS_WRITTEN, "q01"): 4,
            (structural, SearchMode.HYBRID, AS_WRITTEN, "q02"): 7,
            (structural, SearchMode.HYBRID_RERANK, AS_WRITTEN, "q02"): None,
            (structural, SearchMode.VECTOR, WIDENED, "q01"): 3,
        }
    )

    report = render_report(QUESTIONS, outcomes, ["Прогін: тест"])

    assert "Прогін: тест" in report
    # vector: q01 at rank 1, q02 missed; MRR (1 + 0) / 2
    assert "| vector | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |" in report
    # hybrid: ranks 4 and 7; MRR (1/4 + 1/7) / 2 = 0.196
    assert "| hybrid | 0.00 | 0.00 | 0.50 | 1.00 | 0.20 | 0.50 |" in report
    assert "| q02 | plain | — | 7 | — | 1 | 1 | 1 |" in report
    # the contract question: rank 1 in the contract alone, 3 with laws and decisions
    assert "| vector | 1.00 | 1.00 | 1.00 | 0.33 |" in report
    missed = report.split("## 6.")[1]
    assert "q02" in missed and "q01" not in missed


def test_an_empty_question_list_selects_no_outcome_not_all() -> None:
    results = eval_search.Results(outcomes_with({}))

    none = results.chosen(SearchMode.VECTOR, eval_search.Selection(question_ids=()))

    assert none == []


def test_a_report_without_contract_questions_says_so_instead_of_failing() -> None:
    only_law = (LAW_QUESTION,)

    report = render_report(only_law, outcomes_with({}, only_law), ["Прогін: тест"])

    contracts = report.split("## 4.")[1].split("## 5.")[0]
    assert "Питань про договір немає" in contracts
    assert "| vector |" not in contracts


def test_write_run_saves_the_report_and_every_rank_in_a_new_folder(tmp_path: Path) -> None:
    outcomes = outcomes_with({})

    run_dir = eval_search.write_run(tmp_path, STARTED, "# звіт\n", outcomes)

    assert run_dir == tmp_path / "2026-11-06T14-03-00"
    assert (run_dir / "search_metrics.md").read_text(encoding="utf-8") == "# звіт\n"
    lines = (run_dir / "ranks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(outcomes)
    assert json.loads(lines[0])["question_id"] == "q01"


def test_write_run_never_overwrites_an_earlier_run(tmp_path: Path) -> None:
    eval_search.write_run(tmp_path, STARTED, "# перший\n", outcomes_with({}))

    with pytest.raises(FileExistsError):
        eval_search.write_run(tmp_path, STARTED, "# другий\n", outcomes_with({}))


def test_run_searches_both_collections_and_writes_the_report(
    client: QdrantClient, tmp_path: Path
) -> None:
    run_dir = eval_search.run(
        client,
        QUESTIONS,
        CHUNKS,
        CHUNKS,
        PenaltyEmbedder(),
        SilentReranker(),
        reranker_name="fake-reranker",
        results_dir=tmp_path,
        started=STARTED,
    )

    report = (run_dir / "search_metrics.md").read_text(encoding="utf-8")
    assert "fake-embedder" in report and "fake-reranker" in report
    assert "| vector | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |" in report


def test_main_stops_before_qdrant_when_a_question_is_broken(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    questions = tmp_path / "questions.yaml"
    questions.write_text(
        "version: 1\nquestions:\n"
        "  - {id: q01, question: 'Що?', kind: plain, doc_id: own_003,"
        " evidence: такого тексту немає, chunk_ids: [own_003#030]}\n",
        encoding="utf-8",
    )

    code = eval_search.main(["--questions", str(questions), "--qdrant-url", "http://127.0.0.1:1"])

    assert code == 1
    assert "q01: evidence not found in own_003" in capsys.readouterr().out


def test_main_says_how_to_start_qdrant_when_it_is_not_running(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert eval_search.main(["--qdrant-url", "http://127.0.0.1:1"]) == 1
    assert "docker compose up -d qdrant" in capsys.readouterr().out
