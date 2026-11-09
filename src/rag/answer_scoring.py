"""Scoring saved stage 2 answers by rules, with no model and no search.

Facts are looked for in the claims only: the rendered text also holds citations and warnings,
whose numbers («п. 60») the model did not say. Five metrics have thresholds in configs/eval.yaml.
The Commercial Code check is two plain rules: every cited chunk that needs a warning shows it, and
a claim resting on a repealed law does not call it in force. Calling an old norm valid in other
words is for the judge.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.rag.answer import AnswerStatus, warning
from src.rag.answer_eval import (
    AnswerKind,
    AnswerQuestion,
    cites_expected,
    fact_found,
    outcome_matches,
)
from src.rag.answerer import RagResult
from src.rag.chunking import Chunk

EVAL_CONFIG_PATH = Path("configs/eval.yaml")

IN_FORCE_WORD = re.compile(r"\bчинн\w*", re.IGNORECASE)
# «не чинна», «не є чинним», «втратив чинність», «припинив чинність»
NEGATION_BEFORE = re.compile(r"(?:\bне(?:\s+є)?|\bвтрат\w*|\bприпин\w*)\s+$", re.IGNORECASE)
# «чинність кодекс втратив»
LOSS_AFTER = re.compile(r"^\s+(?:\w+\s+){0,2}(?:втрат|припин)", re.IGNORECASE)


class Thresholds(BaseModel):
    """What a run must reach; configs/eval.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_min: float = Field(ge=0, le=1)
    key_facts_min: float = Field(ge=0, le=1)
    not_in_documents_min: float = Field(ge=0, le=1)
    forbidden_facts_max: int = Field(ge=0)
    repealed_as_in_force_max: int = Field(ge=0)


@dataclass(frozen=True)
class Ratio:
    hit: int
    total: int

    @property
    def rate(self) -> float:
        """0 when nothing was measured: a metric with no questions must not pass."""
        return self.hit / self.total if self.total else 0.0


@dataclass(frozen=True)
class QuestionScore:
    id: str
    kind: AnswerKind
    outcome_ok: bool
    cites_evidence: bool | None  # None: the question has no evidence chunk
    key_facts_total: int
    missing_key_facts: tuple[str, ...]
    forbidden_found: tuple[str, ...]
    repealed_problems: tuple[str, ...]


@dataclass(frozen=True)
class Scorecard:
    questions: tuple[QuestionScore, ...]
    outcome: Ratio  # shown, no threshold: the metrics below split it by what matters
    source: Ratio
    key_facts: Ratio
    not_in_documents: Ratio
    forbidden_facts: int
    repealed_as_in_force: int


@dataclass(frozen=True)
class Gate:
    name: str
    measured: float
    limit: float
    passed: bool


def load_thresholds(path: Path = EVAL_CONFIG_PATH) -> Thresholds:
    """The thresholds, or a ValueError that names the file and what is wrong in it."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return Thresholds.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error


def claims_text(result: RagResult) -> str:
    """What the model itself said; empty when there is no answer."""
    if result.answer is None:
        return ""
    return " ".join(claim.text for claim in result.answer.claims)


def missing_key_facts(question: AnswerQuestion, result: RagResult) -> tuple[str, ...]:
    said = claims_text(result)
    return tuple(fact for fact in question.key_facts if not fact_found(fact, said))


def forbidden_facts_present(question: AnswerQuestion, result: RagResult) -> tuple[str, ...]:
    said = claims_text(result)
    return tuple(fact for fact in question.forbidden_facts if fact_found(fact, said))


def _chunk(chunks: Mapping[str, Chunk], chunk_id: str) -> Chunk:
    if chunk_id not in chunks:
        raise ValueError(f"{chunk_id} is cited but is not in the chunks: the run is stale")
    return chunks[chunk_id]


def _calls_in_force(claim: str) -> bool:
    return any(
        not NEGATION_BEFORE.search(claim[: word.start()])
        and not LOSS_AFTER.search(claim[word.end() :])
        for word in IN_FORCE_WORD.finditer(claim)
    )


def _missing_warnings(result: RagResult, chunks: Mapping[str, Chunk]) -> list[str]:
    shown = result.text or ""
    return [
        f"{chunk_id}: no warning under the answer"
        for chunk_id in result.cited
        if (text := warning(_chunk(chunks, chunk_id))) is not None and text not in shown
    ]


def _claims_calling_in_force(result: RagResult, chunks: Mapping[str, Chunk]) -> list[str]:
    if result.answer is None:
        return []
    problems = []
    for claim in result.answer.claims:
        cited = [_chunk(chunks, result.sources[label]) for label in claim.sources]
        repealed = [chunk.chunk_id for chunk in cited if chunk.in_force is False]
        if repealed and _calls_in_force(claim.text):
            problems.append(f"{repealed[0]}: called in force: «{claim.text}»")
    return problems


def repealed_problems(result: RagResult, chunks: Mapping[str, Chunk]) -> tuple[str, ...]:
    return (*_missing_warnings(result, chunks), *_claims_calling_in_force(result, chunks))


def _score_question(
    question: AnswerQuestion, result: RagResult, chunks: Mapping[str, Chunk]
) -> QuestionScore:
    return QuestionScore(
        id=question.id,
        kind=question.kind,
        outcome_ok=outcome_matches(result, question),
        cites_evidence=cites_expected(result, question),
        key_facts_total=len(question.key_facts),
        missing_key_facts=missing_key_facts(question, result),
        forbidden_found=forbidden_facts_present(question, result),
        repealed_problems=repealed_problems(result, chunks),
    )


def score_run(
    pairs: Sequence[tuple[AnswerQuestion, RagResult]], chunks: Mapping[str, Chunk]
) -> Scorecard:
    scores = tuple(_score_question(question, result, chunks) for question, result in pairs)
    source_checks = [score.cites_evidence for score in scores if score.cites_evidence is not None]
    honest = [
        score.outcome_ok
        for (question, _), score in zip(pairs, scores, strict=True)
        if question.expect is AnswerStatus.NOT_IN_DOCUMENTS
    ]
    facts_total = sum(score.key_facts_total for score in scores)
    facts_missing = sum(len(score.missing_key_facts) for score in scores)
    return Scorecard(
        questions=scores,
        outcome=Ratio(sum(score.outcome_ok for score in scores), len(scores)),
        source=Ratio(sum(source_checks), len(source_checks)),
        key_facts=Ratio(facts_total - facts_missing, facts_total),
        not_in_documents=Ratio(sum(honest), len(honest)),
        forbidden_facts=sum(len(score.forbidden_found) for score in scores),
        repealed_as_in_force=sum(len(score.repealed_problems) for score in scores),
    )


def _at_least(name: str, ratio: Ratio, minimum: float) -> Gate:
    return Gate(name, ratio.rate, minimum, ratio.rate >= minimum)


def _at_most(name: str, count: int, maximum: int) -> Gate:
    return Gate(name, count, maximum, count <= maximum)


def check_thresholds(scorecard: Scorecard, thresholds: Thresholds) -> tuple[Gate, ...]:
    return (
        _at_least("source", scorecard.source, thresholds.source_min),
        _at_least("key_facts", scorecard.key_facts, thresholds.key_facts_min),
        _at_least(
            "not_in_documents", scorecard.not_in_documents, thresholds.not_in_documents_min
        ),
        _at_most("forbidden_facts", scorecard.forbidden_facts, thresholds.forbidden_facts_max),
        _at_most(
            "repealed_as_in_force",
            scorecard.repealed_as_in_force,
            thresholds.repealed_as_in_force_max,
        ),
    )
