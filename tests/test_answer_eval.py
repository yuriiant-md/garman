from pathlib import Path

import pytest

from src.rag.answer import AnswerStatus
from src.rag.answer_eval import (
    ANSWER_QUESTIONS_PATH,
    AnswerKind,
    AnswerQuestion,
    cites_expected,
    fact_found,
    load_answer_questions,
    outcome_matches,
    summarize,
)
from src.rag.answerer import Outcome, RagResult
from src.rag.chunking import Chunk
from src.rag.search_eval import load_questions

CHUNKS_PATH = Path("data/rag/chunks.jsonl")
MIN_OFF_CORPUS = 5

SEARCH_QUESTIONS = """\
version: 1
questions:
  - id: q01
    question: Яку пеню платить покупець, якщо затримає оплату?
    kind: plain
    filter: {contract_id: own_003, doc_types: [contract]}
    doc_id: own_003
    evidence: облікової ставки НБУ
    chunk_ids: [own_003#030]
    note: п. 7.2
  - id: q09
    question: Скільки процентів річних платить боржник?
    kind: exact
    filter: {doc_types: [law]}
    doc_id: law_civil_code
    evidence: три проценти річних
    chunk_ids: [law_civil_code#026]
"""
ANSWER_QUESTIONS = """\
version: 2
search_questions:
  - id: q01
    expected_answer: Пеня в розмірі подвійної облікової ставки НБУ за кожен день.
    key_facts: ["подвійн", "облікової ставки|облікову ставку"]
  - id: q09
    expected_answer: Три проценти річних.
    key_facts: ["три проценти|3%"]
    forbidden_facts: ["5%"]
questions:
  - id: t01
    kind: off_corpus
    question: Яку ціну дизпалива встановлює договір?
    contract_id: own_003
    expect: not_in_documents
    expected_answer: У документах цього немає.
"""


def write(tmp_path: Path, answers: str = ANSWER_QUESTIONS) -> tuple[Path, Path]:
    search_path = tmp_path / "search_questions.yaml"
    answer_path = tmp_path / "answer_questions.yaml"
    search_path.write_text(SEARCH_QUESTIONS, encoding="utf-8")
    answer_path.write_text(answers, encoding="utf-8")
    return answer_path, search_path


def question(**fields: object) -> AnswerQuestion:
    values: dict[str, object] = {
        "id": "q01",
        "kind": AnswerKind.SEARCH,
        "question": "Яку пеню платить покупець?",
        "expect": AnswerStatus.ANSWERED,
    }
    return AnswerQuestion.model_validate(values | fields)


def result(**fields: object) -> RagResult:
    values: dict[str, object] = {
        "question": "Яку пеню платить покупець?",
        "contract_id": "own_003",
        "outcome": Outcome.ANSWERED,
        "model_called": True,
        "sources": {"S1": "own_003#030"},
        "context_tokens": 500,
        "left_out": (),
        "below_threshold": (),
        "model": "claude-haiku-5-5",
        "prompt_version": "v1",
        "prompt_sha256": "abc",
        "attempts": 1,
        "latency_s": 2.0,
        "input_tokens": 3000,
        "output_tokens": 200,
    }
    return RagResult.model_validate(values | fields)


def test_search_questions_come_with_their_contract_and_their_evidence_chunks(
    tmp_path: Path,
) -> None:
    answer_path, search_path = write(tmp_path)

    questions = load_answer_questions(answer_path, search_path)

    assert [q.id for q in questions] == ["q01", "q09", "t01"]
    q01, q09, t01 = questions
    assert (q01.kind, q01.contract_id, q01.expect) == (
        AnswerKind.SEARCH,
        "own_003",
        AnswerStatus.ANSWERED,
    )
    assert q01.should_cite == ("own_003#030",)
    assert q01.note == "п. 7.2"
    assert q09.contract_id is None  # a law question: the doc_types filter of Е2.4 is not kept
    assert (t01.kind, t01.expect, t01.should_cite) == (
        AnswerKind.OFF_CORPUS,
        AnswerStatus.NOT_IN_DOCUMENTS,
        (),
    )


def test_an_unknown_search_question_is_refused(tmp_path: Path) -> None:
    answer_path, search_path = write(tmp_path, ANSWER_QUESTIONS.replace("id: q09", "id: q99"))

    with pytest.raises(ValueError, match="q99"):
        load_answer_questions(answer_path, search_path)


def test_an_id_used_twice_is_refused(tmp_path: Path) -> None:
    answer_path, search_path = write(tmp_path, ANSWER_QUESTIONS.replace("id: t01", "id: q01"))

    with pytest.raises(ValueError, match="q01"):
        load_answer_questions(answer_path, search_path)


def test_the_repository_questions_load_and_cover_every_kind() -> None:
    questions = load_answer_questions(ANSWER_QUESTIONS_PATH)
    search_ids = {q.id for q in load_questions(Path("data/rag/search_questions.yaml"))}

    assert len(questions) >= 20
    assert {q.kind for q in questions} == set(AnswerKind)
    assert search_ids <= {q.id for q in questions}


def test_a_search_question_takes_its_reference_from_the_answer_file(tmp_path: Path) -> None:
    answer_path, search_path = write(tmp_path)

    q01, q09, t01 = load_answer_questions(answer_path, search_path)

    assert q01.expected_answer == "Пеня в розмірі подвійної облікової ставки НБУ за кожен день."
    assert q01.key_facts == ("подвійн", "облікової ставки|облікову ставку")
    assert q01.forbidden_facts == ()
    assert q09.forbidden_facts == ("5%",)
    assert t01.expected_answer == "У документах цього немає."
    assert t01.key_facts == ()


def test_a_file_of_the_old_version_is_refused(tmp_path: Path) -> None:
    answer_path, search_path = write(tmp_path, ANSWER_QUESTIONS.replace("version: 2", "version: 1"))

    with pytest.raises(ValueError, match="version"):
        load_answer_questions(answer_path, search_path)


@pytest.mark.parametrize(
    ("fact", "text", "found"),
    [
        ("5%", "штраф 5% вартості", True),
        ("5%", "штраф 5 % вартості", True),
        ("5 %", "штраф 5% вартості", True),
        ("5%", "штраф 15% вартості", False),
        ("5%", "штраф 0,5% вартості", False),
        ("30%", "штраф 35% вартості", False),
        ("60", "понад 60 днів", True),
        ("60", "понад 600 днів", False),
        ("3", "ставка 3,5 річних", False),
        ("0,1%", "пеня 0,1% за день", True),
        ("подвійн", "ПОДВІЙНОЇ ставки", True),
        ("60|шістдесят", "понад шістдесят днів", True),
        ("60|шістдесят", "понад сорок днів", False),
        ("зобов'язання", "грошового зобов`язання", True),
        ("облікової ставки", "облікової\n ставки НБУ", True),
    ],
)
def test_a_fact_is_found_by_its_exact_number_or_by_any_alternative(
    fact: str, text: str, found: bool
) -> None:
    assert fact_found(fact, text) is found


def repository_chunks() -> dict[str, Chunk]:
    lines = CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
    chunks = (Chunk.model_validate_json(line) for line in lines)
    return {chunk.chunk_id: chunk for chunk in chunks}


def in_evidence(fact: str, question: AnswerQuestion, chunks: dict[str, Chunk]) -> bool:
    return any(fact_found(fact, chunks[chunk_id].text) for chunk_id in question.should_cite)


def test_every_repository_question_has_an_expected_answer() -> None:
    questions = load_answer_questions(ANSWER_QUESTIONS_PATH)

    assert [q.id for q in questions if not q.expected_answer] == []


def test_every_answerable_repository_question_has_key_facts_and_evidence() -> None:
    answerable = [
        q
        for q in load_answer_questions(ANSWER_QUESTIONS_PATH)
        if q.expect is AnswerStatus.ANSWERED
    ]

    assert [q.id for q in answerable if not q.key_facts or not q.should_cite] == []


def test_the_repository_evidence_chunks_exist_and_belong_to_the_contract() -> None:
    chunks = repository_chunks()
    questions = load_answer_questions(ANSWER_QUESTIONS_PATH)
    cited = [(q, chunk_id) for q in questions for chunk_id in q.should_cite]

    assert [chunk_id for _, chunk_id in cited if chunk_id not in chunks] == []
    assert [
        (q.id, chunk_id)
        for q, chunk_id in cited
        if q.contract_id and chunks[chunk_id].doc_id != q.contract_id
    ] == []


def test_every_key_fact_of_the_repository_is_in_its_evidence() -> None:
    chunks = repository_chunks()
    missing = [
        (q.id, fact)
        for q in load_answer_questions(ANSWER_QUESTIONS_PATH)
        for fact in q.key_facts
        if not in_evidence(fact, q, chunks)
    ]

    assert missing == []


def test_no_forbidden_fact_of_the_repository_is_in_its_evidence() -> None:
    chunks = repository_chunks()
    present = [
        (q.id, fact)
        for q in load_answer_questions(ANSWER_QUESTIONS_PATH)
        for fact in q.forbidden_facts
        if in_evidence(fact, q, chunks)
    ]

    assert present == []


def test_every_similar_contract_trap_names_what_it_must_not_borrow() -> None:
    traps = [
        q
        for q in load_answer_questions(ANSWER_QUESTIONS_PATH)
        if q.kind is AnswerKind.SIMILAR_CONTRACT
    ]

    assert traps
    assert [q.id for q in traps if not q.forbidden_facts] == []


def test_the_repository_has_enough_questions_with_no_answer_in_the_corpus() -> None:
    questions = load_answer_questions(ANSWER_QUESTIONS_PATH)

    assert sum(q.kind is AnswerKind.OFF_CORPUS for q in questions) >= MIN_OFF_CORPUS


def test_the_outcome_matches_only_the_expected_one() -> None:
    assert outcome_matches(result(), question()) is True
    assert outcome_matches(result(outcome=Outcome.FAILED), question()) is False
    assert outcome_matches(
        result(outcome=Outcome.NOT_IN_DOCUMENTS), question(expect=AnswerStatus.NOT_IN_DOCUMENTS)
    )


def test_citing_any_evidence_chunk_is_enough_and_no_evidence_means_no_check() -> None:
    expected = question(should_cite=("court_910_13071_19#113", "court_910_13071_19#083"))

    assert cites_expected(result(cited=("court_910_13071_19#083",)), expected) is True
    assert cites_expected(result(cited=("own_003#031",)), expected) is False
    assert cites_expected(result(cited=()), question()) is None


def test_the_summary_counts_outcomes_tokens_cost_and_latency() -> None:
    pairs = [
        (question(should_cite=("own_003#030",)), result(cited=("own_003#030",), latency_s=1.0)),
        (question(id="q02", should_cite=("own_003#031",)), result(latency_s=3.0)),
        (
            question(id="t01", expect=AnswerStatus.NOT_IN_DOCUMENTS),
            result(
                outcome=Outcome.NOT_IN_DOCUMENTS,
                model_called=False,
                attempts=0,
                input_tokens=0,
                output_tokens=0,
                latency_s=0.5,
            ),
        ),
    ]

    summary = summarize(pairs, price_input=0.1, price_output=0.5)

    assert summary.questions == 3
    assert summary.outcomes == {Outcome.ANSWERED: 2, Outcome.NOT_IN_DOCUMENTS: 1}
    assert (summary.matched, summary.cite_checked, summary.cite_hits) == (3, 2, 1)
    assert (summary.model_calls, summary.input_tokens, summary.output_tokens) == (2, 6000, 400)
    assert summary.cost_usd == pytest.approx((6000 * 0.1 + 400 * 0.5) / 1_000_000)
    assert (summary.latency_p50_s, summary.latency_p95_s) == (1.0, pytest.approx(2.8))
