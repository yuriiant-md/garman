"""Search over the stage 2 index: by vector, by vector and BM25 fused, and fused then reranked.

`vector` compares the query's bge-m3 embedding with each chunk's by cosine. `hybrid` takes the best
chunks by vector and the best by BM25 and fuses the two lists by RRF (reciprocal rank fusion): a
chunk scores by its places in the lists, not by the lists' scores, which are on different scales.
`hybrid_rerank` gives the hybrid's top chunks to a cross-encoder, which reads the query and a chunk
together and judges the pair better, but is too slow to read the whole index.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Protocol, Self, assert_never

from pydantic import BaseModel, ConfigDict, model_validator
from qdrant_client import QdrantClient, models

from src.rag.bm25 import query_weights, tokenize
from src.rag.chunking import Chunk
from src.rag.embedding import CACHE_DIR, Embedder
from src.rag.index import BM25, DENSE, IndexStamp, check_fresh, searchable_text
from src.rag.sources import DocType

DEFAULT_K = 5
PREFETCH_LIMIT = 50  # chunks each of vector and BM25 hands to the fusion
RERANK_CANDIDATES = 20  # hybrid chunks the cross-encoder reads
RERANKER_NAME = "BAAI/bge-reranker-v2-m3-int8"  # multilingual, int8 ONNX for CPU
# what a question about one contract may also cite: the law and the court practice
REFERENCE_TYPES = (DocType.LAW, DocType.COURT_DECISION)


class SearchMode(StrEnum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class SearchFilter(BaseModel):
    """Which chunks a search may return; an unset field does not filter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_id: str | None = None  # this contract, laws and decisions; no other contract
    doc_types: frozenset[DocType] = frozenset()
    skip_repealed: bool = False  # drops laws not in force; other documents have no in_force
    # only decisions have a date: undated chunks pass the range, dated ones must fall in it
    version_from: date | None = None
    version_to: date | None = None

    @model_validator(mode="after")
    def _range_is_ordered(self) -> Self:
        if self.version_from and self.version_to and self.version_from > self.version_to:
            raise ValueError(
                f"version_from {self.version_from} is after version_to {self.version_to}"
            )
        return self

    def to_qdrant(self) -> models.Filter | None:
        must: list[models.Condition] = []
        if self.contract_id is not None:
            this_contract = _match("doc_id", self.contract_id)
            references = _match_any("doc_type", REFERENCE_TYPES)
            must.append(models.Filter(should=[this_contract, references]))
        if self.doc_types:
            must.append(_match_any("doc_type", sorted(self.doc_types)))
        if self.version_from or self.version_to:
            in_range = models.DatetimeRange(gte=self.version_from, lte=self.version_to)
            undated = models.IsEmptyCondition(is_empty=models.PayloadField(key="version_date"))
            dated_in_range = models.FieldCondition(key="version_date", range=in_range)
            must.append(models.Filter(should=[undated, dated_in_range]))
        must_not: list[models.Condition] = []
        if self.skip_repealed:
            must_not.append(_match("in_force", False))
        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine, RRF or reranker score, by the mode: compare only within one search
    rank: int  # 1 is the best


class Reranker(Protocol):
    def rerank(self, query: str, texts: Sequence[str]) -> tuple[float, ...]: ...


class CrossEncoderModel(Protocol):
    """The part of fastembed's TextCrossEncoder this module uses."""

    def rerank(self, query: str, documents: Iterable[str]) -> Iterable[float]: ...


@dataclass(frozen=True)
class CrossEncoderReranker:
    model: CrossEncoderModel

    def rerank(self, query: str, texts: Sequence[str]) -> tuple[float, ...]:
        """One score per text, in the order of `texts`; higher means a better answer."""
        return tuple(float(score) for score in self.model.rerank(query, texts))


def bge_reranker(cache_dir: Path = CACHE_DIR) -> CrossEncoderReranker:
    """bge-reranker-v2-m3 on CPU; the first call downloads the model into cache_dir."""
    # imported here: fastembed loads onnxruntime, which only a real model needs
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return CrossEncoderReranker(TextCrossEncoder(RERANKER_NAME, cache_dir=str(cache_dir)))


@dataclass(frozen=True)
class Searcher:
    """Searches one collection; refuses at once an index not built from the expected chunks."""

    client: QdrantClient
    collection: str
    embedder: Embedder
    expected: IndexStamp
    reranker: Reranker | None = None

    def __post_init__(self) -> None:
        check_fresh(self.client, self.collection, self.expected)

    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        k: int = DEFAULT_K,
        search_filter: SearchFilter | None = None,
    ) -> tuple[Hit, ...]:
        """The k best chunks for the query, best first."""
        if not query.strip():
            raise ValueError("the query is empty")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        qdrant_filter = search_filter.to_qdrant() if search_filter else None
        match mode:
            case SearchMode.VECTOR:
                return self._vector(query, k, qdrant_filter)
            case SearchMode.HYBRID:
                return self._hybrid(query, k, qdrant_filter)
            case SearchMode.HYBRID_RERANK:
                return self._reranked(query, k, qdrant_filter)
            case _:
                assert_never(mode)

    def _vector(self, query: str, k: int, qdrant_filter: models.Filter | None) -> tuple[Hit, ...]:
        response = self.client.query_points(
            self.collection,
            query=self._dense(query),
            using=DENSE,
            query_filter=qdrant_filter,
            limit=k,
            with_payload=True,
        )
        return _hits(response.points)

    def _hybrid(self, query: str, k: int, qdrant_filter: models.Filter | None) -> tuple[Hit, ...]:
        # the filter goes into each list: a list cut to 50 before filtering could lose them all
        prefetch = [
            models.Prefetch(
                query=self._dense(query), using=DENSE, filter=qdrant_filter, limit=PREFETCH_LIMIT
            )
        ]
        weights = query_weights(tokenize(query))
        if weights.indices:  # a query of punctuation alone has no words for BM25
            sparse = models.SparseVector(indices=list(weights.indices), values=list(weights.values))
            prefetch.append(
                models.Prefetch(
                    query=sparse, using=BM25, filter=qdrant_filter, limit=PREFETCH_LIMIT
                )
            )
        response = self.client.query_points(
            self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=qdrant_filter,
            limit=k,
            with_payload=True,
        )
        return _hits(response.points)

    def _reranked(
        self, query: str, k: int, qdrant_filter: models.Filter | None
    ) -> tuple[Hit, ...]:
        if self.reranker is None:
            raise ValueError("hybrid_rerank needs a reranker: pass bge_reranker() to Searcher")
        candidates = self._hybrid(query, max(k, RERANK_CANDIDATES), qdrant_filter)
        scores = self.reranker.rerank(query, [searchable_text(hit.chunk) for hit in candidates])
        # sorted() is stable: chunks the reranker scores equal keep their hybrid order
        pairs = zip(scores, candidates, strict=True)
        ranked = sorted(pairs, key=lambda pair: pair[0], reverse=True)
        return tuple(
            Hit(chunk=hit.chunk, score=score, rank=rank)
            for rank, (score, hit) in enumerate(ranked[:k], start=1)
        )

    def _dense(self, query: str) -> list[float]:
        [vector] = self.embedder.embed([query])
        return list(vector)


def _hits(points: Sequence[models.ScoredPoint]) -> tuple[Hit, ...]:
    return tuple(
        Hit(chunk=Chunk.model_validate(point.payload), score=point.score, rank=rank)
        for rank, point in enumerate(points, start=1)
    )


def _match(key: str, value: str | bool) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _match_any(key: str, values: Sequence[str]) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchAny(any=list(values)))
