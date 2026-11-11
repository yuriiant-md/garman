"""The questions stage 2 answers are tried on, and the checks Python can make without a judge.

The Е2.4 search questions are named by id, not copied: their text, contract and the chunks with
the evidence stay in data/rag/search_questions.yaml. Their filters (doc_types, skip_repealed) are
not kept: an answer searches the contract of the question, the laws and the decisions
(context.py). The traps (off the corpus, the repealed Commercial Code, an injection) are written
here, and the expected answers of all of them.

Two checks need no judge: the outcome is the expected one, and the answer cites a chunk that
holds the evidence. Whether the claims say what the chunks say is for the Е2.6 judge.
"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from src.extraction.metrics import cost_usd, percentile
from src.rag.answer import AnswerStatus
from src.rag.answerer import Outcome, RagResult
from src.rag.bm25 import unify_apostrophes
from src.rag.search_eval import SearchQuestion, load_questions

ANSWER_QUESTIONS_PATH = Path("data/rag/answer_questions.yaml")
SEARCH_QUESTIONS_PATH = Path("data/rag/search_questions.yaml")

FACT_ALTERNATIVE_SEPARATOR = "|"
SPACE_BEFORE_PERCENT = re.compile(r"(?<=\d)\s+%")
# a number stands alone: «5%» is not in «15%» or «0,5%», «60» is not in «600» or «60,5»
NOT_AFTER_A_NUMBER = r"(?<!\d)(?<!\d[,.])"
NOT_BEFORE_A_NUMBER = r"(?!\d|[,.]\d)"


class AnswerKind(StrEnum):
    SEARCH = "search"  # an Е2.4 question: the answer is in a known chunk
    OFF_CORPUS = "off_corpus"  # the documents do not answer it
    REPEALED = "repealed"  # the answer is in the Commercial Code, not in force since 28.08.2025
    INJECTION = "injection"  # the question tells the model to drop its rules
    SIMILAR_CONTRACT = "similar_contract"  # another contract has the same clause, other numbers


# one or more alternatives split by "|", none empty: "60|шістдесят"
Fact = Annotated[str, StringConstraints(pattern=r"^[^|]+(\|[^|]+)*$")]


class AnswerReference(BaseModel):
    """What a right answer says: the facts are for Python, the expected answer for the judge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_answer: str | None = Field(default=None, min_length=1)
    key_facts: tuple[Fact, ...] = ()  # short and in a form that does not inflect: «5%», «подвійн»
    forbidden_facts: tuple[Fact, ...] = ()  # e.g. the number of the similar contract


class SearchQuestionReference(AnswerReference):
    id: str  # a question of data/rag/search_questions.yaml


class AnswerQuestion(AnswerReference):
    id: str
    kind: AnswerKind
    question: str = Field(min_length=1)
    contract_id: str | None = None
    expect: AnswerStatus
    should_cite: tuple[str, ...] = ()  # chunks with the evidence; citing any one of them is enough
    note: str | None = None


class AnswerQuestionSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[2]
    search_questions: tuple[SearchQuestionReference, ...] = ()
    questions: tuple[AnswerQuestion, ...] = ()


@dataclass(frozen=True)
class Summary:
    questions: int
    outcomes: Mapping[Outcome, int]
    matched: int  # outcome as expected
    cite_checked: int  # questions with evidence chunks
    cite_hits: int  # of those, answers citing one of the chunks
    model_calls: int  # questions the model was asked; the rest the threshold answered
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_p50_s: float
    latency_p95_s: float


def from_search_question(
    question: SearchQuestion, reference: SearchQuestionReference
) -> AnswerQuestion:
    return AnswerQuestion(
        id=question.id,
        kind=AnswerKind.SEARCH,
        question=question.question,
        contract_id=question.filter.contract_id,
        expect=AnswerStatus.ANSWERED,
        should_cite=question.chunk_ids,
        note=question.note,
        expected_answer=reference.expected_answer,
        key_facts=reference.key_facts,
        forbidden_facts=reference.forbidden_facts,
    )


def load_answer_questions(
    path: Path = ANSWER_QUESTIONS_PATH, search_path: Path = SEARCH_QUESTIONS_PATH
) -> tuple[AnswerQuestion, ...]:
    """The named search questions first, in the order named, then the questions of the file."""
    try:
        question_set = AnswerQuestionSet.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error
    found = {question.id: question for question in load_questions(search_path)}
    references = question_set.search_questions
    if unknown := [reference.id for reference in references if reference.id not in found]:
        raise ValueError(f"{path}: search_questions not in {search_path}: {', '.join(unknown)}")
    questions = (
        *(from_search_question(found[reference.id], reference) for reference in references),
        *question_set.questions,
    )
    counts = Counter(question.id for question in questions)
    if repeated := sorted(qid for qid, count in counts.items() if count > 1):
        raise ValueError(f"{path}: question ids repeat: {', '.join(repeated)}")
    return questions


def outcome_matches(result: RagResult, question: AnswerQuestion) -> bool:
    return result.outcome.value == question.expect.value


def cites_expected(result: RagResult, question: AnswerQuestion) -> bool | None:
    """Whether the answer cites a chunk with the evidence; None when no chunk is known."""
    if not question.should_cite:
        return None
    return any(chunk_id in result.cited for chunk_id in question.should_cite)


def _normalized(text: str) -> str:
    """One case, one space, one apostrophe, and «5 %» written as «5%»."""
    spaced = " ".join(unify_apostrophes(text).casefold().split())
    return SPACE_BEFORE_PERCENT.sub("%", spaced)


def _alternative_pattern(alternative: str) -> str:
    start = NOT_AFTER_A_NUMBER if alternative[:1].isdigit() else ""
    end = NOT_BEFORE_A_NUMBER if alternative[-1:].isdigit() else ""
    return f"{start}{re.escape(alternative)}{end}"


def fact_found(fact: str, text: str) -> bool:
    """Whether the text holds any alternative of the fact."""
    normalized_text = _normalized(text)
    return any(
        re.search(_alternative_pattern(_normalized(alternative)), normalized_text)
        for alternative in fact.split(FACT_ALTERNATIVE_SEPARATOR)
    )


def summarize(
    pairs: Sequence[tuple[AnswerQuestion, RagResult]], *, price_input: float, price_output: float
) -> Summary:
    """Counts over a run; prices are USD per 1M tokens, as in configs/models.yaml."""
    if not pairs:
        raise ValueError("no answers to summarize")
    results = [result for _, result in pairs]
    cite_checks = [cites_expected(result, question) for question, result in pairs]
    input_tokens = sum(result.input_tokens for result in results)
    output_tokens = sum(result.output_tokens for result in results)
    latencies = [result.latency_s for result in results]
    return Summary(
        questions=len(pairs),
        outcomes=dict(Counter(result.outcome for result in results)),
        matched=sum(outcome_matches(result, question) for question, result in pairs),
        cite_checked=sum(check is not None for check in cite_checks),
        cite_hits=sum(check is True for check in cite_checks),
        model_calls=sum(result.model_called for result in results),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd(input_tokens, output_tokens, price_input, price_output),
        latency_p50_s=percentile(latencies, 50),
        latency_p95_s=percentile(latencies, 95),
    )
