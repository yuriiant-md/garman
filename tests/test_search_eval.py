from pathlib import Path

import pytest
from pydantic import ValidationError

from src.rag.chunking import Chunk
from src.rag.search import Hit, SearchFilter
from src.rag.search_eval import (
    QuestionKind,
    SearchQuestion,
    check_questions,
    first_relevant_rank,
    is_relevant,
    load_questions,
    mean_reciprocal_rank,
    recall_at,
    with_laws_and_decisions,
)
from src.rag.sources import DocType

QUESTIONS_PATH = Path("data/rag/search_questions.yaml")


def chunk(chunk_id: str, text: str) -> Chunk:
    doc_id = chunk_id.split("#")[0]
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_type=DocType.CONTRACT,
        title=doc_id,
        short_title=doc_id,
        text_sha256="0" * 64,
        text=text,
    )


def question(**fields: object) -> SearchQuestion:
    values: dict[str, object] = {
        "id": "q01",
        "question": "Яку пеню платить покупець?",
        "kind": QuestionKind.PLAIN,
        "doc_id": "own_003",
        "evidence": "пеню в розмірі подвійної облікової ставки НБУ",
        "chunk_ids": ["own_003#030"],
    }
    return SearchQuestion.model_validate(values | fields)


def hits(*chunks: Chunk) -> tuple[Hit, ...]:
    return tuple(Hit(chunk=item, score=1.0, rank=rank) for rank, item in enumerate(chunks, 1))


RIGHT = chunk("own_003#030", "7.2. Сплачує пеню в розмірі подвійної\nоблікової ставки НБУ.")
WRONG_DOC = chunk("own_008#013", "Покупець сплачує пеню в розмірі подвійної облікової ставки НБУ.")
WRONG_TEXT = chunk("own_003#031", "7.3. Постачальник сплачує штраф 30%.")


def test_the_project_questions_load_and_match_the_filter_fields() -> None:
    questions = load_questions(QUESTIONS_PATH)

    assert len(questions) == 15
    assert len({item.id for item in questions}) == 15
    contract_questions = [item for item in questions if item.filter.contract_id]
    assert contract_questions
    assert all(item.filter.doc_types == {DocType.CONTRACT} for item in contract_questions)


def test_a_chunk_is_relevant_with_the_evidence_from_the_right_document() -> None:
    assert is_relevant(RIGHT, question())
    assert not is_relevant(WRONG_DOC, question())
    assert not is_relevant(WRONG_TEXT, question())


def test_spaces_line_breaks_and_apostrophes_are_compared_loosely() -> None:
    text = chunk("law_civil_code#071", "Покупець  зобов’язаний\nоплатити товар")
    asked = question(doc_id="law_civil_code", evidence="Покупець зобов'язаний оплатити товар")

    assert is_relevant(text, asked)


def test_the_first_relevant_rank_counts_from_one_and_is_none_when_missing() -> None:
    assert first_relevant_rank(hits(WRONG_DOC, WRONG_TEXT, RIGHT, RIGHT), question()) == 3
    assert first_relevant_rank(hits(WRONG_DOC, WRONG_TEXT), question()) is None


def test_recall_at_k_is_the_share_of_questions_answered_in_the_top_k() -> None:
    ranks = (1, 3, None, 7)

    assert recall_at(ranks, 1) == 0.25
    assert recall_at(ranks, 3) == 0.5
    assert recall_at(ranks, 10) == 0.75


def test_mrr_averages_one_over_the_rank_and_counts_a_miss_as_zero() -> None:
    assert mean_reciprocal_rank((1, 2, None, 4)) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def test_metrics_of_no_questions_are_an_error() -> None:
    with pytest.raises(ValueError):
        recall_at((), 5)
    with pytest.raises(ValueError):
        mean_reciprocal_rank(())


def test_check_finds_evidence_missing_or_in_other_chunks_than_listed() -> None:
    chunks = (RIGHT, WRONG_TEXT)
    listed_wrong = question(id="q02", chunk_ids=["own_003#031"])
    not_there = question(id="q03", evidence="такого тексту немає")

    problems = check_questions((question(), listed_wrong, not_there), chunks)

    assert len(problems) == 2
    assert problems[0].startswith("q02:") and "own_003#030" in problems[0]
    assert problems[1].startswith("q03:") and "not found" in problems[1]


def test_the_project_questions_pass_the_check_on_the_project_chunks() -> None:
    lines = Path("data/rag/chunks.jsonl").read_text(encoding="utf-8").splitlines()
    chunks = tuple(Chunk.model_validate_json(line) for line in lines)

    assert check_questions(load_questions(QUESTIONS_PATH), chunks) == ()


def test_with_laws_and_decisions_widens_only_a_contract_question() -> None:
    contract_only = SearchFilter(contract_id="own_003", doc_types=frozenset({DocType.CONTRACT}))
    laws = SearchFilter(doc_types=frozenset({DocType.LAW}))

    assert with_laws_and_decisions(contract_only) == SearchFilter(contract_id="own_003")
    assert with_laws_and_decisions(laws) == laws


def test_repeated_question_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "questions.yaml"
    entry = (
        "  - {id: q01, question: 'Що?', kind: plain, doc_id: own_003, evidence: пеню,"
        " chunk_ids: [own_003#030]}\n"
    )
    path.write_text("version: 1\nquestions:\n" + entry + entry, encoding="utf-8")

    with pytest.raises(ValidationError, match="q01"):
        load_questions(path)


def test_an_unknown_field_in_a_question_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "questions.yaml"
    path.write_text(
        "version: 1\nquestions:\n"
        "  - {id: q01, question: 'Що?', kind: plain, doc_id: own_003, evidence: пеню,"
        " chunk_ids: [own_003#030], filter: {doc_id: own_003}}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="doc_id"):
        load_questions(path)
