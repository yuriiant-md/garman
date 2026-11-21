import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient

from src.agents.corpus_tool import (
    CorpusHit,
    CorpusHits,
    corpus_hit,
    search_corpus,
    search_corpus_tool,
)
from src.agents.graph import Agent
from src.agents.limits import Limits
from src.agents.tools import InvalidArguments
from src.rag.answer import citation, warning
from src.rag.chunking import Chunk
from src.rag.context import BLOCK_SEPARATOR, ContextSettings, document_block, estimate_tokens
from src.rag.embedding import Progress, Vector
from src.rag.index import build_index, index_stamp
from src.rag.search import Hit, Searcher, SearchFilter, SearchMode
from src.rag.sources import DocType
from tests.agent_fixtures import Clock, FakeModel, ask, tool_results, turn

SETTINGS = ContextSettings(contract_k=4, law_k=3, decision_k=3, max_tokens=4000, min_score=-1.7)
LIMITS = Limits(
    max_steps=5, max_tokens=10_000, max_cost_usd=1.0, tool_timeout_s=1, request_timeout_s=60
)


def chunk(chunk_id: str, doc_type: DocType, **fields: object) -> Chunk:
    doc_id = chunk_id.split("#")[0]
    values: dict[str, object] = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_type": doc_type,
        "title": f"Документ {doc_id}",
        "short_title": doc_id,
        "text_sha256": "0" * 64,
        "text": f"Текст {chunk_id} про пеню.",
    }
    return Chunk.model_validate(values | fields)


def hits(*chunks: Chunk, score: float = 1.0) -> tuple[Hit, ...]:
    return tuple(Hit(chunk=c, score=score, rank=rank) for rank, c in enumerate(chunks, start=1))


CIVIL_CODE = chunk(
    "law_civil_code#026",
    DocType.LAW,
    title="Цивільний кодекс України",
    short_title="ЦК",
    ref="ст. 625",
    in_force=True,
    retrieved_on=date(2025, 10, 1),
    source_url="https://zakon.rada.gov.ua/laws/show/435-15",
)
COMMERCIAL_CODE = chunk(
    "law_commercial_code#012", DocType.LAW, short_title="ГК", ref="ст. 231", in_force=False
)
DECISION_1 = chunk(
    "court_922_1467_19#038",
    DocType.COURT_DECISION,
    version_date=date(2023, 1, 10),
    cites_commercial_code=True,
    pages=(4, 5),
)
DECISION_2 = chunk("court_905_541_16#097", DocType.COURT_DECISION)
CLAUSE = chunk("own_003#030", DocType.CONTRACT, ref="п. 7.2")


@dataclass
class FakeSearch:
    """Answers by the filter's document type and records every call."""

    by_type: dict[DocType, tuple[Hit, ...]]
    calls: list[tuple[str, SearchMode, int, SearchFilter | None]] = field(default_factory=list)

    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        k: int = 5,
        search_filter: SearchFilter | None = None,
    ) -> tuple[Hit, ...]:
        self.calls.append((query, mode, k, search_filter))
        assert search_filter is not None
        [doc_type] = search_filter.doc_types
        return self.by_type.get(doc_type, ())[:k]


def fake_search(score: float = 1.0) -> FakeSearch:
    return FakeSearch(
        {
            DocType.CONTRACT: hits(CLAUSE, score=score),
            DocType.LAW: hits(CIVIL_CODE, COMMERCIAL_CODE, score=score),
            DocType.COURT_DECISION: hits(DECISION_1, DECISION_2, score=score),
        }
    )


# --- what is searched


def test_searches_the_laws_and_the_decisions_apart_reranked_with_their_k() -> None:
    search = fake_search()

    search_corpus(search, SETTINGS, "проценти річних")

    asked = [(query, mode, k, f.doc_types if f else None) for query, mode, k, f in search.calls]
    assert asked == [
        ("проценти річних", SearchMode.HYBRID_RERANK, 3, frozenset({DocType.LAW})),
        ("проценти річних", SearchMode.HYBRID_RERANK, 3, frozenset({DocType.COURT_DECISION})),
    ]


def test_never_searches_a_contract() -> None:
    search = fake_search()

    search_corpus(search, SETTINGS, "пеня")

    for _, _, _, search_filter in search.calls:
        assert search_filter is not None
        assert search_filter.contract_id is None
        assert DocType.CONTRACT not in search_filter.doc_types


def test_drops_a_contract_chunk_even_when_the_search_returns_one() -> None:
    search = FakeSearch({DocType.LAW: hits(CLAUSE, CIVIL_CODE)})

    found = search_corpus(search, SETTINGS, "пеня")

    assert [hit.chunk_id for hit in found.hits] == [CIVIL_CODE.chunk_id]


def test_laws_and_decisions_take_turns() -> None:
    found = search_corpus(fake_search(), SETTINGS, "пеня")

    assert [hit.chunk_id for hit in found.hits] == [
        CIVIL_CODE.chunk_id,
        DECISION_1.chunk_id,
        COMMERCIAL_CODE.chunk_id,
        DECISION_2.chunk_id,
    ]


def test_nothing_above_min_score_gives_no_hits() -> None:
    found = search_corpus(fake_search(score=-5.0), SETTINGS, "погода в Одесі")

    assert found.hits == ()


def test_the_token_budget_of_the_settings_limits_the_hits() -> None:
    two_blocks = BLOCK_SEPARATOR.join(
        [document_block("S1", CIVIL_CODE), document_block("S2", DECISION_1)]
    )
    tight = SETTINGS.model_copy(update={"max_tokens": estimate_tokens(two_blocks)})

    found = search_corpus(fake_search(), tight, "пеня")

    assert [hit.chunk_id for hit in found.hits] == [CIVIL_CODE.chunk_id, DECISION_1.chunk_id]


# --- what a hit says


def test_a_hit_carries_the_stage_2_citation_source_and_date() -> None:
    [hit] = search_corpus(FakeSearch({DocType.LAW: hits(CIVIL_CODE)}), SETTINGS, "пеня").hits

    assert hit.citation == citation(CIVIL_CODE) == "[ЦК, ст. 625]"
    assert hit.title == "Цивільний кодекс України"
    assert hit.doc_type is DocType.LAW
    assert hit.source_url == CIVIL_CODE.source_url
    assert hit.retrieved_on == date(2025, 10, 1)
    assert hit.in_force is True
    assert hit.warning is None
    assert hit.text == CIVIL_CODE.text


def test_a_repealed_law_and_practice_under_it_carry_the_stage_2_warning() -> None:
    search = FakeSearch(
        {DocType.LAW: hits(COMMERCIAL_CODE), DocType.COURT_DECISION: hits(DECISION_1)}
    )

    law, decision = search_corpus(search, SETTINGS, "пеня").hits

    assert law.warning == warning(COMMERCIAL_CODE)
    assert law.warning is not None and "втратив чинність" in law.warning
    assert decision.warning == warning(DECISION_1)
    assert decision.version_date == date(2023, 1, 10)
    assert decision.pages == (4, 5)


# --- the tool


def test_the_tool_answers_with_json_its_output_schema_accepts() -> None:
    tool = search_corpus_tool(fake_search(), SETTINGS)

    result = tool.run({"query": "пеня"})

    parsed = CorpusHits.model_validate_json(result)
    assert len(parsed.hits) == 4
    assert "Цивільний кодекс" in result  # Ukrainian as is, not \u escapes
    repealed = json.loads(result)["hits"][2]
    assert repealed["chunk_id"] == COMMERCIAL_CODE.chunk_id
    assert repealed["in_force"] is False  # no nulls in the output, but False is kept


def test_the_output_schema_refuses_a_contract_hit() -> None:
    contract_hit = corpus_hit(CIVIL_CODE).model_dump() | {"doc_type": DocType.CONTRACT}

    with pytest.raises(ValidationError, match="never returns a contract"):
        CorpusHit.model_validate(contract_hit)


def test_no_hits_come_back_as_an_empty_list_not_an_error() -> None:
    tool = search_corpus_tool(fake_search(score=-5.0), SETTINGS)

    assert CorpusHits.model_validate_json(tool.run({"query": "погода"})).hits == ()


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"query": "  "},
        {"query": " ц "},
        {"query": "п" * 501},
        {"query": 625},
        {"query": "пеня", "doc_type": "contract"},
    ],
)
def test_bad_arguments_are_refused_before_any_search(arguments: dict[str, object]) -> None:
    search = fake_search()
    tool = search_corpus_tool(search, SETTINGS)

    with pytest.raises(InvalidArguments, match="search_corpus"):
        tool.run(arguments)  # type: ignore[arg-type]
    assert search.calls == []


@pytest.mark.parametrize("query", ["ЦК", "п" * 500])
def test_a_query_of_2_to_500_characters_is_searched(query: str) -> None:
    search = fake_search()

    search_corpus_tool(search, SETTINGS).run({"query": query})

    assert {asked for asked, *_ in search.calls} == {query}


def test_the_query_is_searched_without_surrounding_spaces() -> None:
    search = fake_search()

    search_corpus_tool(search, SETTINGS).run({"query": "  пеня  "})

    assert {query for query, *_ in search.calls} == {"пеня"}


def test_the_tool_is_described_by_its_yaml_as_a_safe_v1_with_one_argument() -> None:
    tool = search_corpus_tool(fake_search(), SETTINGS)

    assert (tool.name, tool.version) == ("search_corpus", "v1")
    assert tool.description.side_effect is False
    schema = tool.spec["input_schema"]
    assert list(schema["properties"]) == ["query"]
    assert schema["required"] == ["query"]
    text = tool.description.description
    for must in ("Коли викликати", "Коли не викликати", "get_clause", "warning"):
        assert must in text


# --- with the agent


def test_the_agent_gets_the_hits_and_records_the_tool_version() -> None:
    model = FakeModel(turn("", ask(query="проценти річних")), turn("ЦК ст. 625 [ЦК, ст. 625]."))
    agent = Agent(
        model=model,
        tools=(search_corpus_tool(fake_search(), SETTINGS),),
        system="Ти помічник фермера.",
        limits=LIMITS,
        prices=(1.0, 5.0),
        model_name="haiku-5.5",
        clock=Clock(),
    )

    trace = agent.run("Скільки процентів річних за прострочення?")

    [result] = tool_results(model, 1)
    assert "[ЦК, ст. 625]" in str(result["content"])
    [call] = trace.steps[0].tool_calls
    assert (call.name, call.tool_version, call.error) == ("search_corpus", "v1", None)


# --- over a real (in-memory) Qdrant index


class TopicEmbedder:
    """(1, 0) for a text about a penalty, (0, 1) for any other."""

    model_name = "fake-embedder"
    dimension = 2

    def embed(
        self, texts: Sequence[str], on_progress: Progress | None = None
    ) -> tuple[Vector, ...]:
        return tuple((1.0, 0.0) if "пен" in text.lower() else (0.0, 1.0) for text in texts)


class FlatReranker:
    def rerank(self, query: str, texts: Sequence[str]) -> tuple[float, ...]:
        return tuple(1.0 for _ in texts)


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
def test_over_a_qdrant_index_a_contract_about_the_query_is_not_found() -> None:
    chunks = (CLAUSE, CIVIL_CODE, DECISION_1)
    client = QdrantClient(":memory:")
    embedder = TopicEmbedder()
    build_index(client, "chunks_test", chunks, embedder)
    searcher = Searcher(
        client, "chunks_test", embedder, index_stamp(chunks, embedder), FlatReranker()
    )

    found = search_corpus(searcher, SETTINGS, "пеня за прострочення")

    assert [hit.chunk_id for hit in found.hits] == [CIVIL_CODE.chunk_id, DECISION_1.chunk_id]
