"""The context of a stage 2 answer: the found chunks, labelled S1, S2, … and wrapped as data.

A question about a contract searches three lists apart: that contract, the laws and the court
decisions. Searched together, the long decisions push the laws out: «скільки процентів річних…»
found no law in a shared top 6 but ЦК ст. 625 first among laws alone. No other farmer's contract
is ever searched. The lists then take turns, so each gets room in the token budget.

Chunks the reranker scores below `min_score` are dropped; when none is left the context is empty
and the answer is «немає» without asking the model. Each chunk goes into the prompt as
<document id="S1">…</document>, with its source and, for the repealed Commercial Code or practice
resting on it, the warning; a chunk's own text cannot open or close such a block.
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from types import MappingProxyType
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.rag.answer import citation, warning
from src.rag.chunking import Chunk
from src.rag.search import Hit, SearchFilter, SearchMode
from src.rag.sources import DocType

# Haiku 5.5 on the stage 1 texts: slope of usage.input_tokens over the text's characters,
# 48 documents, prompt v2 (v1 gives 2.37); Ukrainian takes more tokens than English
CHARS_PER_TOKEN = 2.36
MODE = SearchMode.HYBRID_RERANK  # the reranker's score is what min_score is measured on
BLOCK_SEPARATOR = "\n\n"
# "<document", "</document", "< / DOCUMENT" …: what could close a block or open a fake one
TAG_START = re.compile(r"<(?=\s*/?\s*document)", re.IGNORECASE)
LAWS = SearchFilter(doc_types=frozenset({DocType.LAW}))
DECISIONS = SearchFilter(doc_types=frozenset({DocType.COURT_DECISION}))
# so the model can say whose words a claim repeats: the contract's, a law's or the court's
DOC_TYPE_NAMES = {
    DocType.CONTRACT: "договір",
    DocType.LAW: "закон",
    DocType.COURT_DECISION: "постанова Верховного Суду",
}


class Search(Protocol):
    """The part of Searcher this module uses; tests pass a fake."""

    def search(
        self,
        query: str,
        mode: SearchMode = ...,
        k: int = ...,
        search_filter: SearchFilter | None = None,
    ) -> tuple[Hit, ...]: ...


class ContextSettings(BaseModel):
    """How many chunks each search brings and how many tokens the context may take."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_k: int = Field(ge=1)
    law_k: int = Field(ge=1)
    decision_k: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    min_score: float = Field(allow_inf_nan=False)  # reranker score; below it a chunk is noise


@dataclass(frozen=True)
class Context:
    sources: Mapping[str, Chunk]  # "S1" -> chunk, in the order of the blocks
    text: str  # the <document> blocks for the prompt; "" when nothing was found
    tokens: int  # estimated
    left_out: tuple[str, ...]  # chunk_ids that passed min_score but did not fit max_tokens
    below_threshold: tuple[str, ...]  # chunk_ids under min_score


def estimate_tokens(text: str) -> int:
    """Tokens the text takes, from its length; no API call."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def document_block(label: str, chunk: Chunk) -> str:
    """The chunk as the model sees it: its source, its warning if any, then the text."""
    caution = warning(chunk)
    lines = [
        f"Документ: {chunk.title}",
        f"Тип: {DOC_TYPE_NAMES[chunk.doc_type]}",
        f"Джерело: {citation(chunk)}",
        *([f"Увага: {caution}"] if caution else []),
        "Текст:",
        *([chunk.heading] if chunk.heading else []),
        chunk.text,
    ]
    body = "\n".join(escaped(line) for line in lines)
    return f'<document id="{label}">\n{body}\n</document>'


def retrieve(
    search: Search, question: str, contract_id: str | None, settings: ContextSettings
) -> tuple[Hit, ...]:
    """The hits of each search, taking turns: contract, law, decision, contract, …"""
    searches = [(LAWS, settings.law_k), (DECISIONS, settings.decision_k)]
    if contract_id is not None:
        this_contract = SearchFilter(
            contract_id=contract_id, doc_types=frozenset({DocType.CONTRACT})
        )
        searches.insert(0, (this_contract, settings.contract_k))
    lists = [search.search(question, MODE, k, search_filter) for search_filter, k in searches]
    return tuple(hit for turn in zip_longest(*lists) for hit in turn if hit is not None)


def pack(hits: Sequence[Hit], settings: ContextSettings) -> Context:
    """Whole chunks in the order given, over min_score, while they fit max_tokens.

    A chunk that does not fit is skipped and takes no label, so a smaller one after it may fit.
    """
    passed = [hit.chunk for hit in hits if hit.score >= settings.min_score]
    below = tuple(hit.chunk.chunk_id for hit in hits if hit.score < settings.min_score)
    chosen: list[Chunk] = []
    blocks: list[str] = []
    left_out: list[str] = []
    for chunk in passed:
        block = document_block(f"S{len(chosen) + 1}", chunk)
        if estimate_tokens(BLOCK_SEPARATOR.join([*blocks, block])) > settings.max_tokens:
            left_out.append(chunk.chunk_id)
            continue
        chosen.append(chunk)
        blocks.append(block)
    text = BLOCK_SEPARATOR.join(blocks)
    return Context(
        sources=MappingProxyType({f"S{n}": chunk for n, chunk in enumerate(chosen, start=1)}),
        text=text,
        tokens=estimate_tokens(text),
        left_out=tuple(left_out),
        below_threshold=below,
    )


def build_context(
    search: Search, question: str, contract_id: str | None, settings: ContextSettings
) -> Context:
    """Search for the question and pack what was found into the answer's context."""
    return pack(retrieve(search, question, contract_id, settings), settings)


def escaped(line: str) -> str:
    """The text with anything that could close a document block or open a fake one defused."""
    return TAG_START.sub("&lt;", line)
