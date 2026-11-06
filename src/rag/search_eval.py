"""Search quality on questions with a known answer: which hit is right, recall@k and MRR.

A hit is right when it comes from the question's document and holds the question's evidence word
for word. The rule needs no chunk ids, so it scores structural chunks and fixed-size windows alike.
recall@k is the share of questions with a right hit among the first k; MRR is the mean of 1/rank
of the first right hit, a miss counting 0: it rewards a right hit at 1 more than one at 5.
"""

from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.rag.bm25 import unify_apostrophes
from src.rag.chunking import Chunk
from src.rag.search import Hit, SearchFilter
from src.rag.sources import DocType


class QuestionKind(StrEnum):
    EXACT = "exact"  # shares rare words or numbers with the answer: BM25 should help
    PARAPHRASE = "paraphrase"  # says it in other words: the vector should help
    PLAIN = "plain"


class SearchQuestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str = Field(min_length=1)
    kind: QuestionKind
    filter: SearchFilter = SearchFilter()
    doc_id: str
    evidence: str = Field(min_length=1)
    chunk_ids: tuple[str, ...] = Field(min_length=1)  # the structural chunks with the evidence
    note: str | None = None


class QuestionSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1]
    questions: tuple[SearchQuestion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Self:
        counts = Counter(question.id for question in self.questions)
        if repeated := sorted(question_id for question_id, count in counts.items() if count > 1):
            raise ValueError(f"question ids repeat: {', '.join(repeated)}")
        return self


def load_questions(path: Path) -> tuple[SearchQuestion, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return QuestionSet.model_validate(raw).questions


def _loose(text: str) -> str:
    """One space for any run of spaces and line breaks, one apostrophe for all of its spellings."""
    return " ".join(unify_apostrophes(text).split())


def is_relevant(chunk: Chunk, question: SearchQuestion) -> bool:
    return chunk.doc_id == question.doc_id and _loose(question.evidence) in _loose(chunk.text)


def first_relevant_rank(hits: Sequence[Hit], question: SearchQuestion) -> int | None:
    """The rank of the first right hit, 1 being the best; None when no hit is right."""
    return next((hit.rank for hit in hits if is_relevant(hit.chunk, question)), None)


def recall_at(ranks: Sequence[int | None], k: int) -> float:
    _require_questions(ranks)
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    _require_questions(ranks)
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)


def _require_questions(ranks: Sequence[int | None]) -> None:
    if not ranks:
        raise ValueError("no questions to measure")


def check_questions(
    questions: Sequence[SearchQuestion], chunks: Sequence[Chunk]
) -> tuple[str, ...]:
    """Questions whose evidence no chunk holds, or other chunks hold than chunk_ids list."""
    problems = []
    for question in questions:
        holding = [chunk.chunk_id for chunk in chunks if is_relevant(chunk, question)]
        if not holding:
            problems.append(f"{question.id}: evidence not found in {question.doc_id}")
        elif sorted(holding) != sorted(question.chunk_ids):
            problems.append(
                f"{question.id}: evidence is in {', '.join(holding)}, "
                f"chunk_ids list {', '.join(question.chunk_ids)}"
            )
    return tuple(problems)


def with_laws_and_decisions(search_filter: SearchFilter) -> SearchFilter:
    """A contract question's filter widened to laws and decisions; any other filter as it is."""
    if search_filter.contract_id is None or search_filter.doc_types != {DocType.CONTRACT}:
        return search_filter
    return search_filter.model_copy(update={"doc_types": frozenset()})
