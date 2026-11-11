"""Calibration of the stage 2 LLM judge on the author's manual marks.

The sheet (data/rag/judge_marks.yaml) holds what the judge judged in a saved run, without its
verdicts: the question, the reference, the answer the farmer saw, each claim with the documents
it cites. The author fills `correct` and every `supported`; then each mark is compared with the
judge's verdict. A judgement with no verdict is a disagreement: a judge that cannot answer must
not look reliable. Questions whose verdicts the author had already seen are marked `seen`, and
the agreement is counted on them apart, so the bias of those marks shows in the numbers.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.rag.answer import AnswerStatus
from src.rag.answer_eval import AnswerQuestion
from src.rag.answer_scoring import Gate, Ratio
from src.rag.answerer import Outcome, RagResult
from src.rag.judge import JudgeThresholds

MARKS_PATH = Path("data/rag/judge_marks.yaml")
YAML_WIDTH = 100_000  # no folding: a long line stays one line, as the model read it

SHEET_HEADER = """\
# Manual marks for the calibration of the stage 2 judge (scripts/calibrate_judge.py).
# Mark each answer as the judge would, from this file only:
#   correct: true when the answer reaches the reference's conclusion with the right citation
#            and numbers; false otherwise. Present only where there is a reference.
#   supported: true when the documents the claim cites say what the claim says; false when
#            any part of it is missing from them or says otherwise.
# In doubt, false: the farmer acts on the answer. seen: true — the judge's verdict on this
# question was seen before marking.
"""


class ClaimMark(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int = Field(ge=1)
    text: str
    sources: tuple[str, ...]
    supported: bool | None = None  # None: not marked yet


class AnswerMarks(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    seen: bool  # the author saw the judge's verdict on it before marking
    question: str
    reference: str | None = None  # None: the judge does not check this answer for correctness
    answer: str
    correct: bool | None = None  # None: not marked yet, or no reference
    claims: tuple[ClaimMark, ...] = ()
    documents: dict[str, str] = Field(default_factory=dict)  # label -> <document> block


class MarkSheet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1]
    run: str  # the folder name of the run the marks are for
    answers: tuple[AnswerMarks, ...] = ()


def _reference(question: AnswerQuestion, text: str | None) -> str | None:
    """The reference when the judge checks the answer for correctness, as judge_pair decides."""
    expects = question.expect is AnswerStatus.ANSWERED and question.expected_answer is not None
    return question.expected_answer if expects and text is not None else None


def _entry(question: AnswerQuestion, result: RagResult, seen: bool) -> AnswerMarks | None:
    text = result.text if result.outcome is Outcome.ANSWERED else None
    claims = result.answer.claims if result.answer else ()
    reference = _reference(question, text)
    if text is None or (reference is None and not claims):
        return None
    cited = {label for claim in claims for label in claim.sources}
    return AnswerMarks(
        id=question.id,
        seen=seen,
        question=question.question,
        reference=reference,
        answer=text,
        claims=tuple(
            ClaimMark(n=n, text=claim.text, sources=claim.sources)
            for n, claim in enumerate(claims, start=1)
        ),
        documents={
            label: block for label, block in result.cited_documents.items() if label in cited
        },
    )


def build_sheet(
    run: str, pairs: Sequence[tuple[AnswerQuestion, RagResult]], seen: frozenset[str]
) -> MarkSheet:
    """The empty sheet: what the judge judged in the run, without its verdicts."""
    if unknown := sorted(seen - {question.id for question, _ in pairs}):
        raise ValueError(f"seen names questions the run does not have: {', '.join(unknown)}")
    entries = (_entry(question, result, question.id in seen) for question, result in pairs)
    return MarkSheet(version=1, run=run, answers=tuple(e for e in entries if e is not None))


def unmarked(sheet: MarkSheet) -> list[str]:
    """Every mark still empty, as «q01 correct» or «q01 claim 2»."""
    missing = []
    for entry in sheet.answers:
        if entry.reference is not None and entry.correct is None:
            missing.append(f"{entry.id} correct")
        missing.extend(f"{entry.id} claim {c.n}" for c in entry.claims if c.supported is None)
    return missing


class _SheetDumper(yaml.SafeDumper):
    """Multi-line texts as blocks (|), so a document reads line by line as the model read it."""


def _text(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_SheetDumper.add_representer(str, _text)


def dump_sheet(sheet: MarkSheet, path: Path) -> None:
    body = yaml.dump(
        sheet.model_dump(mode="json"),
        Dumper=_SheetDumper,
        allow_unicode=True,
        sort_keys=False,
        width=YAML_WIDTH,
    )
    path.write_text(SHEET_HEADER + body, encoding="utf-8")


def load_sheet(path: Path) -> MarkSheet:
    """The sheet, or a ValueError that names the file and what is wrong in it."""
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        return MarkSheet.model_validate(data)
    except (yaml.YAMLError, ValidationError) as error:
        raise ValueError(f"{path}: {error}") from error


class Check(StrEnum):
    CORRECT = "correct"
    FAITHFUL = "faithful"


@dataclass(frozen=True)
class JudgeVerdicts:
    """What the judge said of one answer; None where it gave no verdict."""

    correct: bool | None
    supported: tuple[bool | None, ...]


@dataclass(frozen=True)
class Compared:
    """One mark of the author next to the judge's verdict on the same thing."""

    id: str
    check: Check
    n: int | None  # the claim's number; None for correct
    seen: bool
    author: bool
    judge: bool | None

    @property
    def agrees(self) -> bool:
        return self.judge is not None and self.judge == self.author


def _compare_answer(entry: AnswerMarks, verdicts: JudgeVerdicts) -> list[Compared]:
    if len(verdicts.supported) != len(entry.claims):
        raise ValueError(
            f"{entry.id}: {len(entry.claims)} claims in the sheet, "
            f"{len(verdicts.supported)} in the judgements: not the same run"
        )
    items = []
    if entry.reference is not None and entry.correct is not None:
        items.append(
            Compared(entry.id, Check.CORRECT, None, entry.seen, entry.correct, verdicts.correct)
        )
    for claim, judged in zip(entry.claims, verdicts.supported, strict=True):
        if claim.supported is not None:
            items.append(
                Compared(entry.id, Check.FAITHFUL, claim.n, entry.seen, claim.supported, judged)
            )
    return items


def calibrate(sheet: MarkSheet, judge: Mapping[str, JudgeVerdicts]) -> tuple[Compared, ...]:
    """Every mark next to the judge's verdict; refused while a mark is empty or does not fit."""
    if missing := unmarked(sheet):
        raise ValueError(f"not marked yet: {', '.join(missing)}")
    if absent := [entry.id for entry in sheet.answers if entry.id not in judge]:
        raise ValueError(f"the judgements have no record of: {', '.join(absent)}")
    return tuple(
        item for entry in sheet.answers for item in _compare_answer(entry, judge[entry.id])
    )


def agreement(items: Sequence[Compared], check: Check, seen: bool | None = None) -> Ratio:
    """The share of marks the judge agrees with; seen=None counts all, else only that group."""
    chosen = [i for i in items if i.check is check and (seen is None or i.seen is seen)]
    return Ratio(sum(item.agrees for item in chosen), len(chosen))


def calibration_gates(
    items: Sequence[Compared], thresholds: JudgeThresholds
) -> tuple[Gate, ...]:
    """Both checks must agree with the author often enough, or the judge is not to be used."""
    gates = []
    for check in Check:
        rate = agreement(items, check).rate
        gates.append(
            Gate(f"{check}_agreement", rate, thresholds.agreement_min,
                 rate >= thresholds.agreement_min)
        )
    return tuple(gates)
