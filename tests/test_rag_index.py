from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient, models

from src.rag.bm25 import query_weights, tokenize
from src.rag.chunking import Chunk
from src.rag.embedding import Progress, Vector
from src.rag.index import (
    BM25,
    DENSE,
    StaleIndexError,
    build_index,
    check_fresh,
    index_stamp,
    point_id,
    searchable_text,
    stale_reason,
)
from src.rag.sources import DocType

COLLECTION = "chunks_test"
# :memory: Qdrant filters without payload indexes and says so; the server uses them
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")


class FakeEmbedder:
    """A unit vector per text: (1, 0) if it speaks of a penalty, (0, 1) otherwise."""

    model_name = "fake-embedder"
    dimension = 2

    def embed(
        self, texts: Sequence[str], on_progress: Progress | None = None
    ) -> tuple[Vector, ...]:
        if on_progress:
            on_progress(len(texts), len(texts))
        return tuple((1.0, 0.0) if "пен" in text.lower() else (0.0, 1.0) for text in texts)


def chunk(number: int, text: str, **fields: object) -> Chunk:
    values: dict[str, object] = {
        "chunk_id": f"own_003#{number:03d}",
        "doc_id": "own_003",
        "doc_type": DocType.CONTRACT,
        "title": "Договір own_003",
        "short_title": "Договір own_003",
        "text_sha256": "0" * 64,
        "text": text,
    }
    return Chunk.model_validate(values | fields)


CHUNKS = (
    chunk(1, "6.1. Покупець сплачує 86% вартості Товару.", section="6. Оплата", ref="п. 6.1"),
    chunk(2, "7.2. Покупець сплачує пеню за кожний день прострочення.", ref="п. 7.2"),
    chunk(3, "8.1. Сторони звільняються від відповідальності у разі форс-мажору.", ref="п. 8.1"),
)


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


def test_searchable_text_is_title_section_heading_and_text() -> None:
    law = chunk(
        1,
        "1. Боржник не звільняється від відповідальності.",
        title="Цивільний кодекс України",
        heading="Стаття 625. Відповідальність за порушення грошового зобов'язання",
        ref="ст. 625",
    )

    assert searchable_text(law) == (
        "Цивільний кодекс України\n"
        "Стаття 625. Відповідальність за порушення грошового зобов'язання\n"
        "1. Боржник не звільняється від відповідальності."
    )


def test_each_chunk_becomes_one_point_with_the_chunk_as_payload(client: QdrantClient) -> None:
    build_index(client, COLLECTION, CHUNKS, FakeEmbedder())

    [record] = client.retrieve(COLLECTION, [point_id("own_003#002")])

    assert client.count(COLLECTION).count == len(CHUNKS)
    assert record.payload is not None
    assert Chunk.model_validate(record.payload) == CHUNKS[1]


def test_the_dense_vector_is_the_embedding_of_the_searchable_text(client: QdrantClient) -> None:
    build_index(client, COLLECTION, CHUNKS, FakeEmbedder())

    [record] = client.retrieve(COLLECTION, [point_id("own_003#002")], with_vectors=True)

    assert isinstance(record.vector, dict)
    assert record.vector[DENSE] == [1.0, 0.0]


def test_bm25_finds_the_chunk_with_the_rare_query_word_first(client: QdrantClient) -> None:
    build_index(client, COLLECTION, CHUNKS, FakeEmbedder())
    weights = query_weights(tokenize("пеню покупець"))  # «покупець» is in two chunks, «пеню» in one

    found = client.query_points(
        COLLECTION,
        query=models.SparseVector(indices=list(weights.indices), values=list(weights.values)),
        using=BM25,
        limit=3,
    ).points

    assert [point.id for point in found] == [point_id("own_003#002"), point_id("own_003#001")]


def test_a_rebuilt_index_holds_only_the_new_chunks(client: QdrantClient) -> None:
    build_index(client, COLLECTION, CHUNKS, FakeEmbedder())

    build_index(client, COLLECTION, CHUNKS[:1], FakeEmbedder())

    assert client.count(COLLECTION).count == 1


def test_an_empty_corpus_is_not_indexed(client: QdrantClient) -> None:
    with pytest.raises(ValueError, match="no chunks"):
        build_index(client, COLLECTION, (), FakeEmbedder())


def test_an_index_built_from_these_chunks_is_fresh(client: QdrantClient) -> None:
    build_index(client, COLLECTION, CHUNKS, FakeEmbedder())

    check_fresh(client, COLLECTION, index_stamp(CHUNKS, FakeEmbedder()))  # raises nothing


def test_changed_chunks_make_the_index_stale(client: QdrantClient) -> None:
    build_index(client, COLLECTION, CHUNKS, FakeEmbedder())
    changed = (*CHUNKS[:2], chunk(3, "8.1. Інший текст пункту."))

    with pytest.raises(StaleIndexError, match="other chunks"):
        check_fresh(client, COLLECTION, index_stamp(changed, FakeEmbedder()))


def test_another_model_makes_the_index_stale(client: QdrantClient) -> None:
    build_index(client, COLLECTION, CHUNKS, FakeEmbedder())

    class OtherEmbedder(FakeEmbedder):
        model_name = "other-embedder"

    with pytest.raises(StaleIndexError, match="fake-embedder"):
        check_fresh(client, COLLECTION, index_stamp(CHUNKS, OtherEmbedder()))


def test_a_missing_collection_is_stale(client: QdrantClient) -> None:
    with pytest.raises(StaleIndexError, match="no collection"):
        check_fresh(client, COLLECTION, index_stamp(CHUNKS, FakeEmbedder()))


def test_a_collection_without_a_stamp_is_stale(client: QdrantClient) -> None:
    # what a build that stopped half-way leaves behind: the stamp is written last
    client.create_collection(COLLECTION, vectors_config={})

    with pytest.raises(StaleIndexError, match="unfinished"):
        check_fresh(client, COLLECTION, index_stamp(CHUNKS, FakeEmbedder()))


def test_build_index_passes_the_embedding_progress_on(client: QdrantClient) -> None:
    reported: list[tuple[int, int]] = []

    build_index(
        client,
        COLLECTION,
        CHUNKS,
        FakeEmbedder(),
        on_progress=lambda done, total: reported.append((done, total)),
    )

    assert reported == [(3, 3)]


def test_a_fresh_index_has_no_stale_reason(client: QdrantClient) -> None:
    build_index(client, COLLECTION, CHUNKS, FakeEmbedder())

    assert stale_reason(client, COLLECTION, index_stamp(CHUNKS, FakeEmbedder())) is None
    reason = stale_reason(client, COLLECTION, index_stamp(CHUNKS[:2], FakeEmbedder()))
    assert reason is not None
    assert "other chunks (3 then, 2 now)" in reason
