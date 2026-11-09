"""The questions stage 2 answers are tried on, and the checks Python can make without a judge.

The Е2.4 search questions are named by id, not copied: their text, contract and the chunks with
the evidence stay in data/rag/search_questions.yaml. Their filters (doc_types, skip_repealed) are
not kept: an answer searches the contract of the question, the laws and the decisions
(context.py). The traps (off the corpus, the repealed Commercial Code, an injection) are written
here.

Two checks need no judge: the outcome is the expected one, and the answer cites a chunk that
holds the evidence. Whether the claims say what the chunks say is for the Е2.6 judge.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.extraction.metrics import cost_usd, percentile
from src.rag.answer import AnswerStatus
from src.rag.answerer import Outcome, RagResult
from src.rag.search_eval import SearchQuestion, load_questions

ANSWER_QUESTIONS_PATH = Path("data/rag/answer_questions.yaml")
SEARCH_QUESTIONS_PATH = Path("data/rag/search_questions.yaml")


class AnswerKind(StrEnum):
    SEARCH = "search"  # an Е2.4 question: the answer is in a known chunk
    OFF_CORPUS = "off_corpus"  # the documents do not answer it
    REPEALED = "repealed"  # the answer is in the Commercial Code, not in force since 28.08.2025
    INJECTION = "injection"  # the question tells the model to drop its rules


class AnswerQuestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: AnswerKind
    question: str = Field(min_length=1)
    contract_id: str | None = None
    expect: AnswerStatus
    should_cite: tuple[str, ...] = ()  # chunks with the evidence; citing any one of them is enough
    note: str | None = None


class AnswerQuestionSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1]
    search_questions: tuple[str, ...] = ()  # ids in data/rag/search_questions.yaml
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


def from_search_question(question: SearchQuestion) -> AnswerQuestion:
    return AnswerQuestion(
        id=question.id,
        kind=AnswerKind.SEARCH,
        question=question.question,
        contract_id=question.filter.contract_id,
        expect=AnswerStatus.ANSWERED,
        should_cite=question.chunk_ids,
        note=question.note,
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
    if unknown := [qid for qid in question_set.search_questions if qid not in found]:
        raise ValueError(f"{path}: search_questions not in {search_path}: {', '.join(unknown)}")
    questions = (
        *(from_search_question(found[qid]) for qid in question_set.search_questions),
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
