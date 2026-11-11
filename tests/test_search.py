from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient

from src.rag.chunking import Chunk
from src.rag.embedding import Progress, Vector
from src.rag.index import StaleIndexError, build_index, index_stamp, searchable_text
from src.rag.search import (
    PREFETCH_LIMIT,
    RERANK_CANDIDATES,
    CrossEncoderReranker,
    Hit,
    Searcher,
    SearchFilter,
    SearchMode,
)
from src.rag.sources import DocType

COLLECTION = "chunks_test"
# :memory: Qdrant filters without payload indexes and says so; the server uses them
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")


class PenaltyEmbedder:
    """A model that knows one topic: (1, 0) for a text about a penalty, (0, 1) for any other."""

    model_name = "fake-embedder"
    dimension = 2

    def embed(
        self, texts: Sequence[str], on_progress: Progress | None = None
    ) -> tuple[Vector, ...]:
        return tuple((1.0, 0.0) if "пен" in text.lower() else (0.0, 1.0) for text in texts)


@dataclass
class KeywordReranker:
    """Scores 1.0 a text with the keyword and 0.0 any other; remembers what it was asked."""

    keyword: str
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def rerank(self, query: str, texts: Sequence[str]) -> tuple[float, ...]:
        self.calls.append((query, tuple(texts)))
        return tuple(1.0 if self.keyword in text else 0.0 for text in texts)


def chunk(chunk_id: str, text: str, **fields: object) -> Chunk:
    doc_id = chunk_id.split("#")[0]
    values: dict[str, object] = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_type": DocType.CONTRACT,
        "title": f"Документ {doc_id}",
        "short_title": doc_id,
        "text_sha256": "0" * 64,
        "text": text,
    }
    return Chunk.model_validate(values | fields)


LAW = {"doc_type": DocType.LAW}
DECISION = {"doc_type": DocType.COURT_DECISION}
CHUNKS = (
    chunk("own_003#001", "6.1. Покупець сплачує 86% вартості Товару до відвантаження."),
    chunk("own_003#002", "7.2. Покупець сплачує пеню за кожний день прострочення."),
    chunk("own_003#003", "8.1. Сторони звільняються від відповідальності у разі форс-мажору."),
    chunk("own_005#001", "7.1. Продавець сплачує пеню у розмірі подвійної ставки НБУ."),
    chunk("cc#625", "Боржник сплачує три проценти річних.", in_force=True, **LAW),
    chunk("old_law#001", "Пеня за правилами, що втратили силу.", in_force=False, **LAW),
    chunk("vs_2019#001", "Суд зменшив пеню.", version_date=date(2019, 5, 1), **DECISION),
    chunk("vs_2023#001", "Суд стягнув пеню.", version_date=date(2023, 1, 10), **DECISION),
)


@pytest.fixture
def client() -> QdrantClient:
    client = QdrantClient(":memory:")
    build_index(client, COLLECTION, CHUNKS, PenaltyEmbedder())
    return client


def searcher(client: QdrantClient, reranker: KeywordReranker | None = None) -> Searcher:
    embedder = PenaltyEmbedder()
    return Searcher(client, COLLECTION, embedder, index_stamp(CHUNKS, embedder), reranker)


def ids(hits: Sequence[Hit]) -> list[str]:
    return [hit.chunk.chunk_id for hit in hits]


def test_vector_search_returns_k_chunks_from_payload_best_first(client: QdrantClient) -> None:
    hits = searcher(client).search("пеня за прострочення", SearchMode.VECTOR, k=3)

    assert len(hits) == 3
    assert all("пен" in hit.chunk.text.lower() for hit in hits)
    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
    by_id = {item.chunk_id: item for item in CHUNKS}
    assert all(hit.chunk == by_id[hit.chunk.chunk_id] for hit in hits)


def test_hybrid_finds_a_chunk_by_an_exact_word_the_vector_misses(client: QdrantClient) -> None:
    # the vector hears only "пеню"; the rare "86" is known to BM25 alone
    found = searcher(client)

    vector = found.search("пеню 86", SearchMode.VECTOR, k=3)
    hybrid = found.search("пеню 86", SearchMode.HYBRID, k=3)

    assert "own_003#001" not in ids(vector)
    assert "own_003#001" in ids(hybrid)


def test_hybrid_works_when_bm25_knows_no_word_of_the_query(client: QdrantClient) -> None:
    hits = searcher(client).search("абракадабра", SearchMode.HYBRID, k=2)

    assert len(hits) == 2


def test_rerank_reorders_the_hybrid_candidates_by_the_reranker(client: QdrantClient) -> None:
    reranker = KeywordReranker("форс-мажору")

    [hit] = searcher(client, reranker).search("пеню", SearchMode.HYBRID_RERANK, k=1)

    assert hit.chunk.chunk_id == "own_003#003"
    assert hit.score == 1.0
    [(query, texts)] = reranker.calls
    assert query == "пеню"
    assert searchable_text(CHUNKS[2]) in texts


def test_rerank_reads_at_most_the_candidate_limit() -> None:
    many = tuple(chunk(f"own_009#{number:03d}", f"{number}. Пункт.") for number in range(30))
    client = QdrantClient(":memory:")
    embedder = PenaltyEmbedder()
    build_index(client, COLLECTION, many, embedder)
    reranker = KeywordReranker("пеню")
    found = Searcher(client, COLLECTION, embedder, index_stamp(many, embedder), reranker)

    hits = found.search("пункт", SearchMode.HYBRID_RERANK, k=5)

    assert len(hits) == 5
    [(_, texts)] = reranker.calls
    assert len(texts) == RERANK_CANDIDATES


def test_cross_encoder_reranker_gives_one_float_per_text_in_order() -> None:
    class LengthModel:
        def rerank(self, query: str, documents: Iterable[str]) -> Iterable[float]:
            return (len(document) for document in documents)  # ints, as numpy may give

    scores = CrossEncoderReranker(LengthModel()).rerank("пеня", ["абв", "а"])

    assert scores == (3.0, 1.0)
    assert all(type(score) is float for score in scores)


def test_rerank_without_a_reranker_is_an_error(client: QdrantClient) -> None:
    with pytest.raises(ValueError, match="reranker"):
        searcher(client).search("пеню", SearchMode.HYBRID_RERANK)


def test_contract_filter_keeps_this_contract_laws_and_decisions(client: QdrantClient) -> None:
    only = SearchFilter(contract_id="own_003")

    hits = searcher(client).search("пеню", SearchMode.VECTOR, k=10, search_filter=only)

    assert sorted(ids(hits)) == [
        "cc#625",
        "old_law#001",
        "own_003#001",
        "own_003#002",
        "own_003#003",
        "vs_2019#001",
        "vs_2023#001",
    ]


def test_contract_with_contract_type_keeps_only_this_contract(client: QdrantClient) -> None:
    # the filter of every contract question in data/rag/search_questions.yaml
    this_only = SearchFilter(contract_id="own_003", doc_types=frozenset({DocType.CONTRACT}))

    hits = searcher(client).search("пеню", SearchMode.HYBRID, k=10, search_filter=this_only)

    assert sorted(ids(hits)) == ["own_003#001", "own_003#002", "own_003#003"]


def test_hybrid_filters_inside_each_list_not_only_after_the_fusion() -> None:
    # 60 contract chunks closer to the query than the one law fill both lists of 50;
    # filtered only after the fusion, the law would be lost
    near = tuple(chunk(f"own_009#{number:03d}", "Пеню сплачує покупець.") for number in range(60))
    law = chunk("cc#625", "Боржник сплачує три проценти річних.", **LAW)
    corpus = (*near, law)
    client = QdrantClient(":memory:")
    embedder = PenaltyEmbedder()
    build_index(client, COLLECTION, corpus, embedder)
    found = Searcher(client, COLLECTION, embedder, index_stamp(corpus, embedder))
    laws = SearchFilter(doc_types=frozenset({DocType.LAW}))

    hits = found.search("пеню", SearchMode.HYBRID, k=5, search_filter=laws)

    assert len(near) > PREFETCH_LIMIT
    assert ids(hits) == ["cc#625"]


def test_doc_types_filter_keeps_only_those_types(client: QdrantClient) -> None:
    laws = SearchFilter(doc_types=frozenset({DocType.LAW}))

    hits = searcher(client).search("пеню", SearchMode.HYBRID, k=10, search_filter=laws)

    assert sorted(ids(hits)) == ["cc#625", "old_law#001"]


def test_skip_repealed_drops_laws_not_in_force_and_keeps_the_rest(client: QdrantClient) -> None:
    in_force = SearchFilter(skip_repealed=True)

    hits = searcher(client).search("пеню", SearchMode.VECTOR, k=10, search_filter=in_force)

    assert "old_law#001" not in ids(hits)
    assert len(hits) == len(CHUNKS) - 1


def test_date_range_drops_dated_chunks_outside_it_and_keeps_undated(client: QdrantClient) -> None:
    recent = SearchFilter(version_from=date(2020, 1, 1), version_to=date(2024, 12, 31))

    hits = searcher(client).search("пеню", SearchMode.VECTOR, k=10, search_filter=recent)

    assert "vs_2019#001" not in ids(hits)
    assert "vs_2023#001" in ids(hits)
    assert len(hits) == len(CHUNKS) - 1


def test_an_empty_filter_means_no_qdrant_filter() -> None:
    assert SearchFilter().to_qdrant() is None


def test_a_date_range_that_ends_before_it_starts_is_rejected() -> None:
    with pytest.raises(ValidationError, match="version_from"):
        SearchFilter(version_from=date(2024, 1, 1), version_to=date(2020, 1, 1))


def test_a_searcher_refuses_an_index_of_other_chunks(client: QdrantClient) -> None:
    embedder = PenaltyEmbedder()
    other = index_stamp(CHUNKS[:2], embedder)

    with pytest.raises(StaleIndexError, match="other chunks"):
        Searcher(client, COLLECTION, embedder, other)


def test_a_searcher_refuses_a_missing_collection() -> None:
    embedder = PenaltyEmbedder()

    with pytest.raises(StaleIndexError, match="no collection"):
        Searcher(QdrantClient(":memory:"), COLLECTION, embedder, index_stamp(CHUNKS, embedder))


@pytest.mark.parametrize(("query", "k"), [("пеню", 0), ("   ", 5)])
def test_a_blank_query_or_k_below_one_is_an_error(
    client: QdrantClient, query: str, k: int
) -> None:
    with pytest.raises(ValueError):
        searcher(client).search(query, SearchMode.VECTOR, k=k)
