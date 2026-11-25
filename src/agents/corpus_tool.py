"""The agent's tool search_corpus: the stage 2 search over the laws and the court decisions.

It searches as a stage 2 answer does (context.retrieve and pack): the laws and the decisions apart
and taking turns, reranked, chunks under min_score dropped, whole chunks within the token budget.
The settings are the stage 2 ones (configs/answer.yaml), so both stages find the same chunks.

Contracts are never searched. The index holds every farmer's contracts, and a contract read here
would bypass the 403 of the MCP server contracts; the agent reads a contract only through that
server. The filter is set in code, not by the model, and a contract chunk the search returns
anyway is dropped before the answer.

Each hit carries its citation and warning made by Python (answer.citation, answer.warning), so the
model cannot lose the warning that the Commercial Code has been repealed. The citation is the
stage 2 one with the date of the text quoted, «[ЦК, ст. 625; текст станом на 04.11.2026]»: a law
is the version downloaded that day, a decision is dated. The model copies it as it is, and the
task checks (tasks.py) fail an answer citing anything no search returned.
"""

from datetime import date
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from src.agents.tools import Tool, ToolArgs, load_tool
from src.rag.answer import citation, warning
from src.rag.chunking import Chunk
from src.rag.context import ContextSettings, Search, pack, retrieve
from src.rag.search import Hit
from src.rag.sources import DocType

REPO = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO / "tools" / "search_corpus.yaml"
MIN_QUERY_CHARS = 2  # «ЦК», «ПДВ»; one letter is not a question
MAX_QUERY_CHARS = 500  # a query, not a pasted contract


class SearchCorpusArgs(ToolArgs):
    query: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=MIN_QUERY_CHARS, max_length=MAX_QUERY_CHARS
        ),
    ]


class CorpusHit(BaseModel):
    """One found chunk of a law or a decision, as the model reads it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    doc_type: DocType
    title: str
    citation: str  # by Python (dated_citation): the only way the model may cite
    source_url: str | None = None
    version_date: date | None = None  # of a decision
    retrieved_on: date | None = None  # a law: the day its text was downloaded
    in_force: bool | None = None
    warning: str | None = None  # repealed law, or practice under the repealed Commercial Code
    pages: tuple[int, ...] = ()
    text: str

    @field_validator("doc_type")
    @classmethod
    def _not_a_contract(cls, doc_type: DocType) -> DocType:
        if doc_type is DocType.CONTRACT:
            raise ValueError("search_corpus never returns a contract")
        return doc_type


class CorpusHits(BaseModel):
    """The output of search_corpus: best first; empty when the corpus says nothing of it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hits: tuple[CorpusHit, ...]


def search_corpus(search: Search, settings: ContextSettings, query: str) -> CorpusHits:
    """The chunks of laws and decisions a stage 2 answer to the query would get."""
    found = retrieve(search, query, None, settings)
    references = tuple(hit for hit in found if not _is_contract(hit))
    context = pack(references, settings)
    return CorpusHits(hits=tuple(corpus_hit(chunk) for chunk in context.sources.values()))


def corpus_hit(chunk: Chunk) -> CorpusHit:
    """The chunk as the model reads it, with the citation and warning Python makes."""
    return CorpusHit(
        chunk_id=chunk.chunk_id,
        doc_type=chunk.doc_type,
        title=chunk.title,
        citation=dated_citation(chunk),
        source_url=chunk.source_url,
        version_date=chunk.version_date,
        retrieved_on=chunk.retrieved_on,
        in_force=chunk.in_force,
        warning=warning(chunk),
        pages=chunk.pages,
        text=chunk.text,
    )


def dated_citation(chunk: Chunk) -> str:
    """The stage 2 citation with the date of its text: a law's download day, a decision's date.
    A chunk with no date keeps the stage 2 citation; stage 2 answers are not changed."""
    if chunk.doc_type is DocType.LAW and chunk.retrieved_on is not None:
        dated = f"текст станом на {chunk.retrieved_on:%d.%m.%Y}"
    elif chunk.doc_type is DocType.COURT_DECISION and chunk.version_date is not None:
        dated = f"від {chunk.version_date:%d.%m.%Y}"
    else:
        return citation(chunk)
    return f"{citation(chunk).removesuffix(']')}; {dated}]"  # citation() ends in "]"


def search_corpus_tool(
    search: Search, settings: ContextSettings, path: Path = TOOL_PATH
) -> Tool[SearchCorpusArgs]:
    """search_corpus for the agent; settings are the `context` of configs/answer.yaml."""

    def run(args: SearchCorpusArgs) -> str:
        # compact, no nulls, Ukrainian as is: fewer tokens for the model
        return search_corpus(search, settings, args.query).model_dump_json(exclude_none=True)

    return load_tool(path, SearchCorpusArgs, run, output=CorpusHits)


def _is_contract(hit: Hit) -> bool:
    return hit.chunk.doc_type is DocType.CONTRACT
