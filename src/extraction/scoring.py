"""Score one extracted Deal against its golden Deal by the counting rules in README.md.

The unit is a leaf field (`payment.trigger`); an empty nested object equals one whose fields
are all null. List items (penalties, grades, quality indicators) are paired first. Strings are
compared after normalization and with a similarity threshold, because golden Prozorro values
keep OCR noise verbatim while a model may write the name correctly.
"""

import math
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, fields, replace
from difflib import SequenceMatcher
from enum import Enum
from typing import get_args

from pydantic import BaseModel

from src.extraction.text import normalize_text
from src.schemas.deal import Deal, GradeTerms, IndicatorName, Penalty, QualityIndicator

NUMBER_TOLERANCE = 0.005  # equal to the hundredth: 9200 = 9200.004, 0.1 != 0.11
# Calibrated on golden OCR noise: a corrected OCR name scores 0.87-0.99, a declined town name
# 0.82, a transliterated one 0. Names that differ only in a digit (0.92) still pass.
STRING_SIMILARITY_MIN = 0.85
UNSCORED_FIELDS = frozenset({"delivery.period_raw"})
PENALTIES = "penalties"
GRADES = "grades"
INDICATORS = f"{GRADES}.indicators"
LIST_FIELDS = frozenset({PENALTIES, GRADES})  # scored item by item, not as one leaf field
PENALTY_KEY = ("payer", "trigger", "unit")
PENALTY_REST = ("amount", "cap_pct")
GRADE_FIELDS = ("grade", "amount_per_t")  # its indicators are scored one unit each
EXPLANATION = re.compile(r"\s*\([^()]*\)")  # a closed bracket after an indicator's name

type Pair[T] = tuple[T | None, T | None]
PenaltyPair = Pair[Penalty]


@dataclass(frozen=True)
class FieldScore:
    field: str
    expected: object
    predicted: object
    correct: bool
    fabricated: bool  # the model gave a value where the golden deal has null
    extra: bool = False  # a whole list item (penalty, grade, indicator) golden does not have


@dataclass(frozen=True)
class Counts:
    filled: int  # fields with a golden value: the base of the main accuracy
    filled_correct: int
    empty: int  # fields that are null in the golden deal: the base of the fabricated share
    fabricated: int
    total: int
    correct: int
    extra: int = 0  # list items golden does not have: one wrong unit each, outside `empty`

    @classmethod
    def of(cls, scores: Iterable[FieldScore]) -> "Counts":
        scores = tuple(scores)
        golden = tuple(score for score in scores if not score.extra)
        filled = tuple(score for score in golden if score.expected is not None)
        return cls(
            filled=len(filled),
            filled_correct=sum(score.correct for score in filled),
            empty=len(golden) - len(filled),
            fabricated=sum(score.fabricated for score in golden),
            total=len(scores),
            correct=sum(score.correct for score in scores),
            extra=len(scores) - len(golden),
        )

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))


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
    """Scored leaf fields outside the lists; a missing nested object gives nulls."""
    return {path: value for path, value in _leaves(deal) if path not in UNSCORED_FIELDS}


def _leaves(deal: Deal) -> Iterator[tuple[str, object]]:
    for name in Deal.model_fields:
        if name in LIST_FIELDS:
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


def pair_items[T](
    expected: Iterable[T],
    predicted: Iterable[T],
    same: Callable[[T, T], bool],
    agreement: Callable[[T, T], int],
) -> tuple[Pair[T], ...]:
    """Pairs items that are `same` in any order. Candidate pairs are taken from the one agreeing
    most down, so neither list's order changes the result. An unpaired item has None on the
    other side: (golden, None) is missing, (None, predicted) is extra."""
    golden, answers = tuple(expected), tuple(predicted)
    candidates = sorted(
        (
            (i, j)
            for i, item in enumerate(golden)
            for j, answer in enumerate(answers)
            if same(item, answer)
        ),
        key=lambda pair: -agreement(golden[pair[0]], answers[pair[1]]),
    )
    matches: dict[int, int] = {}  # golden index -> answer index
    for i, j in candidates:
        if i not in matches and j not in matches.values():
            matches[i] = j
    paired = tuple(
        (item, answers[matches[i]] if i in matches else None) for i, item in enumerate(golden)
    )
    used = set(matches.values())
    return (*paired, *((None, answer) for j, answer in enumerate(answers) if j not in used))


def pair_penalties(
    expected: Iterable[Penalty], predicted: Iterable[Penalty]
) -> tuple[PenaltyPair, ...]:
    """Pairs by (payer, trigger, unit), the closest on amount and cap_pct first."""
    return pair_items(
        expected, predicted, lambda left, right: _penalty_key(left) == _penalty_key(right),
        _agreement,
    )


def drop_explanation(name: str) -> str:
    """The written name of an indicator without what it explains in brackets: golden keeps the
    name only (RULES.md), so "(зерна жита, ячменю)" after it is not a different name. An
    unclosed bracket is OCR noise around the name itself and stays."""
    return EXPLANATION.sub("", name).strip()


def _same_indicator(expected: QualityIndicator, predicted: QualityIndicator) -> bool:
    """The same name and limit; an "other" indicator also needs a similar written name."""
    if (expected.name, expected.limit) != (predicted.name, predicted.limit):
        return False
    if expected.name != IndicatorName.OTHER:
        return True
    return texts_match(
        drop_explanation(expected.raw_name or ""), drop_explanation(predicted.raw_name or "")
    )


def _indicator_agreement(expected: QualityIndicator, predicted: QualityIndicator) -> int:
    return values_match(expected.value, predicted.value) + (expected.unit == predicted.unit)


def _grade_agreement(expected: GradeTerms, predicted: GradeTerms) -> int:
    return int(values_match(expected.amount_per_t, predicted.amount_per_t))


def _score(field: str, expected: object, predicted: object) -> FieldScore:
    return FieldScore(
        field=field,
        expected=expected,
        predicted=predicted,
        correct=values_match(expected, predicted),
        fabricated=expected is None and predicted is not None,
    )


def _score_fields(
    prefix: str, names: Iterable[str], expected: BaseModel, predicted: BaseModel | None
) -> Iterator[FieldScore]:
    """Fields of a golden list item; a missing item (`predicted` None) has all of them wrong."""
    for name in names:
        answer = None if predicted is None else getattr(predicted, name)
        score = _score(f"{prefix}.{name}", getattr(expected, name), answer)
        yield replace(score, correct=False) if predicted is None else score


def _score_items[T](
    field: str, pairs: Iterable[Pair[T]], score_pair: Callable[[T, T | None], Iterator[FieldScore]]
) -> Iterator[FieldScore]:
    """An extra item is one wrong unit: its values are not golden nulls, so they stay out of
    the fabricated share and are counted as `extra` instead."""
    for expected, predicted in pairs:
        if expected is None:
            yield FieldScore(field, None, predicted, correct=False, fabricated=False, extra=True)
        else:
            yield from score_pair(expected, predicted)


def _score_penalty(expected: Penalty, predicted: Penalty | None) -> Iterator[FieldScore]:
    return _score_fields(PENALTIES, Penalty.model_fields, expected, predicted)


def _score_indicator(
    expected: QualityIndicator, predicted: QualityIndicator | None
) -> Iterator[FieldScore]:
    """One unit per indicator, named after it: right when the value and the unit match."""
    correct = (
        predicted is not None
        and values_match(expected.value, predicted.value)
        and expected.unit == predicted.unit
    )
    yield FieldScore(f"{INDICATORS}.{expected.name}", expected, predicted, correct, False)


def _score_grade(expected: GradeTerms, predicted: GradeTerms | None) -> Iterator[FieldScore]:
    """A grade pairs by its class; a missing one has its price and every indicator wrong."""
    yield from _score_fields(GRADES, GRADE_FIELDS, expected, predicted)
    answers = () if predicted is None else predicted.indicators or ()
    pairs = pair_items(
        expected.indicators or (), answers, _same_indicator, _indicator_agreement
    )
    yield from _score_items(INDICATORS, pairs, _score_indicator)


def score_deal(expected: Deal, predicted: Deal | None) -> tuple[FieldScore, ...]:
    """`predicted` None is a failed answer: every field is wrong and every golden null counts as
    fabricated, so a failure never looks better than a wrong answer."""
    if predicted is None:
        return tuple(
            replace(score, correct=False, fabricated=score.expected is None)
            for score in score_deal(expected, Deal())
        )
    golden, answer = leaf_values(expected), leaf_values(predicted)
    plain = tuple(_score(field, golden[field], answer[field]) for field in golden)
    penalties = pair_penalties(expected.penalties or (), predicted.penalties or ())
    grades = pair_items(
        expected.grades or (), predicted.grades or (),
        lambda left, right: left.grade == right.grade, _grade_agreement,
    )
    return (
        *plain,
        *_score_items(PENALTIES, penalties, _score_penalty),
        *_score_items(GRADES, grades, _score_grade),
    )
