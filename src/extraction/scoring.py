"""Score one extracted Deal against its golden Deal by the counting rules in docs/PLAN.md.

The unit is a leaf field (`price.amount_per_t`); an empty nested object equals one whose fields
are all null. Strings are compared after normalization and with a similarity threshold, because
golden Prozorro values keep OCR noise verbatim while a model may write the name correctly.
"""

import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields
from difflib import SequenceMatcher
from enum import Enum
from typing import get_args

from pydantic import BaseModel

from src.schemas.deal import Deal, Penalty

NUMBER_TOLERANCE = 0.005  # equal to the hundredth: 9200 = 9200.004, 0.1 != 0.11
# Calibrated on golden OCR noise: a corrected OCR name scores 0.87-0.99, a declined town name
# 0.82, a transliterated one 0. Names that differ only in a digit (0.92) still pass.
STRING_SIMILARITY_MIN = 0.85
UNSCORED_FIELDS = frozenset({"delivery.period_raw"})
PENALTIES = "penalties"
PENALTY_KEY = ("payer", "trigger", "unit")
PENALTY_REST = ("amount", "cap_pct")
QUOTES = re.compile(r"[\"«»“”„]")
APOSTROPHES = re.compile(r"['’ʼ`]")  # inside a word: "П'ятихатки" = "Пʼятихатки"
LEGAL_FORMS = re.compile(
    r"\b(?:товариство з обмеженою відповідальністю|приватне акціонерне товариство"
    r"|публічне акціонерне товариство|акціонерне товариство|приватне підприємство"
    r"|державне підприємство|фермерське господарство|тзов|тов|прат|пат|ат|пп|дп|фг)\b"
)

PenaltyPair = tuple[Penalty | None, Penalty | None]


@dataclass(frozen=True)
class FieldScore:
    field: str
    expected: object
    predicted: object
    correct: bool
    fabricated: bool  # the model gave a value where the golden deal has null


@dataclass(frozen=True)
class Counts:
    filled: int  # fields with a golden value: the base of the main accuracy
    filled_correct: int
    empty: int  # fields that are null in the golden deal: the base of the fabricated share
    fabricated: int
    total: int
    correct: int

    @classmethod
    def of(cls, scores: Iterable[FieldScore]) -> "Counts":
        scores = tuple(scores)
        filled = tuple(score for score in scores if score.expected is not None)
        return cls(
            filled=len(filled),
            filled_correct=sum(score.correct for score in filled),
            empty=len(scores) - len(filled),
            fabricated=sum(score.fabricated for score in scores),
            total=len(scores),
            correct=sum(score.correct for score in scores),
        )

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))


def normalize_text(value: str) -> str:
    """Case, quotes, the legal form ("ТОВ", "ФГ"…) and spacing do not change who is named."""
    text = QUOTES.sub(" ", APOSTROPHES.sub("", value.casefold()))
    return " ".join(LEGAL_FORMS.sub(" ", text).split())


def texts_match(expected: str, predicted: str) -> bool:
    left, right = normalize_text(expected), normalize_text(predicted)
    if not left or not right:
        return left == right
    return SequenceMatcher(None, left, right).ratio() >= STRING_SIMILARITY_MIN


def values_match(expected: object, predicted: object) -> bool:
    if expected is None or predicted is None:
        return expected is None and predicted is None
    if isinstance(expected, bool) or isinstance(predicted, bool):
        return expected is predicted  # True == 1.0 in Python, but a flag is not a number
    if isinstance(expected, Enum):
        return expected == predicted  # before str: a StrEnum is a str but needs an exact match
    if isinstance(expected, int | float) and isinstance(predicted, int | float):
        return math.isclose(expected, predicted, rel_tol=0, abs_tol=NUMBER_TOLERANCE)
    if isinstance(expected, str) and isinstance(predicted, str):
        return texts_match(expected, predicted)
    return expected == predicted  # dates


def nested_model(field: str) -> type[BaseModel] | None:
    """The model of a nested-object field of Deal (`price` -> Price), None for a plain field."""
    for arg in get_args(Deal.model_fields[field].annotation):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def leaf_values(deal: Deal) -> dict[str, object]:
    """Scored leaf fields outside penalties; a missing nested object gives nulls."""
    return {path: value for path, value in _leaves(deal) if path not in UNSCORED_FIELDS}


def _leaves(deal: Deal) -> Iterator[tuple[str, object]]:
    for name in Deal.model_fields:
        if name == PENALTIES:
            continue
        value = getattr(deal, name)
        model = nested_model(name)
        if model is None:
            yield name, value
            continue
        for sub in model.model_fields:
            yield f"{name}.{sub}", None if value is None else getattr(value, sub)


def _penalty_key(penalty: Penalty) -> tuple[object, ...]:
    return tuple(getattr(penalty, name) for name in PENALTY_KEY)


def _agreement(expected: Penalty, predicted: Penalty) -> int:
    return sum(
        values_match(getattr(expected, name), getattr(predicted, name)) for name in PENALTY_REST
    )


def pair_penalties(
    expected: Iterable[Penalty], predicted: Iterable[Penalty]
) -> tuple[PenaltyPair, ...]:
    """Pairs by (payer, trigger, unit) in any order. Candidate pairs are taken from the one
    agreeing on most of amount and cap_pct down, so neither list's order changes the result.
    An unpaired penalty has None on the other side: (golden, None) is missing, (None, predicted)
    is extra."""
    golden, answers = tuple(expected), tuple(predicted)
    candidates = sorted(
        (
            (i, j)
            for i, penalty in enumerate(golden)
            for j, answer in enumerate(answers)
            if _penalty_key(penalty) == _penalty_key(answer)
        ),
        key=lambda pair: -_agreement(golden[pair[0]], answers[pair[1]]),
    )
    matches: dict[int, int] = {}  # golden index -> answer index
    for i, j in candidates:
        if i not in matches and j not in matches.values():
            matches[i] = j
    paired = tuple(
        (penalty, answers[matches[i]] if i in matches else None)
        for i, penalty in enumerate(golden)
    )
    used = set(matches.values())
    return (*paired, *((None, answer) for j, answer in enumerate(answers) if j not in used))


def _score(field: str, expected: object, predicted: object) -> FieldScore:
    return FieldScore(
        field=field,
        expected=expected,
        predicted=predicted,
        correct=values_match(expected, predicted),
        fabricated=expected is None and predicted is not None,
    )


def _score_penalty(expected: Penalty | None, predicted: Penalty | None) -> Iterator[FieldScore]:
    """A missing penalty has all its fields wrong; an extra one too, and its values fabricated."""
    unpaired = expected is None or predicted is None
    for name in Penalty.model_fields:
        golden = None if expected is None else getattr(expected, name)
        answer = None if predicted is None else getattr(predicted, name)
        score = _score(f"{PENALTIES}.{name}", golden, answer)
        yield FieldScore(**{**vars(score), "correct": False}) if unpaired else score


def score_deal(expected: Deal, predicted: Deal | None) -> tuple[FieldScore, ...]:
    """`predicted` None is a failed answer: every field is wrong and every golden null counts as
    fabricated, so a failure never looks better than a wrong answer."""
    if predicted is None:
        golden = (*leaf_values(expected).items(), *(
            (f"{PENALTIES}.{name}", getattr(penalty, name))
            for penalty in expected.penalties or ()
            for name in Penalty.model_fields
        ))
        return tuple(
            FieldScore(field, value, None, correct=False, fabricated=value is None)
            for field, value in golden
        )
    golden, answer = leaf_values(expected), leaf_values(predicted)
    plain = tuple(_score(field, golden[field], answer[field]) for field in golden)
    pairs = pair_penalties(expected.penalties or (), predicted.penalties or ())
    return (*plain, *(score for pair in pairs for score in _score_penalty(*pair)))
