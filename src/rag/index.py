"""The stage 2 search index: each chunk is a Qdrant point with two vectors and the chunk as payload.

`dense` is the bge-m3 embedding of the chunk, compared by cosine. `bm25` holds the BM25 term
weights of its words (src/rag/bm25.py); Qdrant multiplies them by IDF when it answers a query.
The payload is the whole chunk: search filters on doc_id, doc_type, in_force and version_date, and
an answer cites the chunk's text and ref. After the last point, the collection's metadata gets a
stamp of what the index was built from, so search can refuse an index built from other chunks or
by another model, and one whose build stopped half-way.
"""

import uuid
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient, models

from src.rag.bm25 import TermWeights, average_length, document_weights, tokenize
from src.rag.chunking import Chunk
from src.rag.documents import sha256_hex
from src.rag.embedding import Embedder, Progress, Vector

DENSE = "dense"
BM25 = "bm25"
STAMP_KEY = "index_stamp"
UPSERT_BATCH = 256  # points per request; 256 chunks with vectors are about 3 MB
# A point id must be a number or a UUID; uuid5 gives a chunk_id the same UUID in every run.
POINT_NAMESPACE = uuid.UUID("5d0f3c1e-8a52-4f0b-9a5e-2f7c4b1d9e60")
FILTER_FIELDS = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "in_force": models.PayloadSchemaType.BOOL,
    "version_date": models.PayloadSchemaType.DATETIME,
}
REBUILD = "rebuild it: uv run python -m scripts.index_rag"


class IndexStamp(BaseModel):
    """What an index was built from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunks_sha256: str
    chunks: int
    model: str
    dimension: int


class StaleIndexError(RuntimeError):
    """The index does not match the chunks or the model the search uses."""


def searchable_text(chunk: Chunk) -> str:
    """What is embedded and counted for BM25: the text with the names of where it stands.

    The ref is left out: a clause number already starts the text ("7.2. …"), an article number
    starts the heading ("Стаття 625. …").
    """
    parts = (chunk.title, chunk.section, chunk.heading, chunk.text)
    return "\n".join(part for part in parts if part)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def index_stamp(chunks: Sequence[Chunk], embedder: Embedder) -> IndexStamp:
    lines = "".join(f"{chunk.model_dump_json()}\n" for chunk in chunks)
    return IndexStamp(
        chunks_sha256=sha256_hex(lines.encode("utf-8")),
        chunks=len(chunks),
        model=embedder.model_name,
        dimension=embedder.dimension,
    )


def build_index(
    client: QdrantClient,
    collection: str,
    chunks: Sequence[Chunk],
    embedder: Embedder,
    *,
    batch_size: int = UPSERT_BATCH,
    on_progress: Progress | None = None,
) -> IndexStamp:
    """Replace the collection with one point per chunk; the stamp is written last."""
    if not chunks:
        raise ValueError(f"{collection}: no chunks to index")
    texts = [searchable_text(chunk) for chunk in chunks]
    tokens = [tokenize(text) for text in texts]
    average = average_length(tokens)
    # embedded before the old collection is dropped: a failure here leaves the old index whole
    vectors = embedder.embed(texts, on_progress)
    points = [
        _point(chunk, vector, document_weights(words, average=average))
        for chunk, vector, words in zip(chunks, vectors, tokens, strict=True)
    ]
    _create_collection(client, collection, embedder.dimension)
    for start in range(0, len(points), batch_size):
        client.upsert(collection, points[start : start + batch_size], wait=True)
    stamp = index_stamp(chunks, embedder)
    client.update_collection(collection, metadata={STAMP_KEY: stamp.model_dump()})
    return stamp


def read_stamp(client: QdrantClient, collection: str) -> IndexStamp | None:
    metadata = client.get_collection(collection).config.metadata or {}
    stamp = metadata.get(STAMP_KEY)
    return None if stamp is None else IndexStamp.model_validate(stamp)


def stale_reason(client: QdrantClient, collection: str, expected: IndexStamp) -> str | None:
    """Why the collection is not an index of these chunks by this model; None when it is."""
    if not client.collection_exists(collection):
        return f"no collection {collection!r} in Qdrant"
    stamp = read_stamp(client, collection)
    if stamp is None:
        return f"{collection}: the index is unfinished, it has no stamp"
    if (stamp.model, stamp.dimension) != (expected.model, expected.dimension):
        return (
            f"{collection} was built by {stamp.model} ({stamp.dimension} numbers), "
            f"search uses {expected.model} ({expected.dimension})"
        )
    if stamp.chunks_sha256 != expected.chunks_sha256:
        return (
            f"{collection} was built from other chunks ({stamp.chunks} then, "
            f"{expected.chunks} now)"
        )
    return None


def check_fresh(client: QdrantClient, collection: str, expected: IndexStamp) -> None:
    """Raise StaleIndexError unless the collection was built from these chunks by this model."""
    if reason := stale_reason(client, collection, expected):
        raise StaleIndexError(f"{reason}; {REBUILD}")


def _create_collection(client: QdrantClient, collection: str, dimension: int) -> None:
    if client.collection_exists(collection):
        client.delete_collection(collection)
    dense = models.VectorParams(size=dimension, distance=models.Distance.COSINE)
    client.create_collection(
        collection,
        vectors_config={DENSE: dense},
        sparse_vectors_config={BM25: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field, schema in FILTER_FIELDS.items():
        client.create_payload_index(collection, field, field_schema=schema)


def _point(chunk: Chunk, vector: Vector, weights: TermWeights) -> models.PointStruct:
    return models.PointStruct(
        id=point_id(chunk.chunk_id),
        vector={
            DENSE: list(vector),
            BM25: models.SparseVector(indices=list(weights.indices), values=list(weights.values)),
        },
        payload=chunk.model_dump(mode="json"),
    )
