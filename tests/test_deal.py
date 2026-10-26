from datetime import date
from enum import StrEnum

import pytest
from pydantic import ValidationError

from src.schemas.deal import (
    Basis,
    Crop,
    Currency,
    DaysType,
    Deal,
    Delivery,
    GradeTerms,
    IndicatorName,
    IndicatorUnit,
    Lab,
    Limit,
    Party,
    Payment,
    PaymentTrigger,
    Penalty,
    PenaltyTrigger,
    PenaltyUnit,
    Price,
    Quality,
    QualityIndicator,
    Tolerance,
)


def test_deal_accepts_valid_nested_data() -> None:
    deal = Deal(
        crop="wheat",
        quantity_t=1000,
        tolerance={"pct": 5, "option": "buyer"},
        price={"currency": "UAH", "vat_included": True},
        basis="CPT",
        delivery={
            "place": "Odesa",
            "date_from": "2026-10-20",
            "date_to": "2026-10-25",
        },
        payment={"days": 5, "days_type": "banking", "trigger": "after_delivery"},
        penalties=[
            {
                "payer": "seller",
                "trigger": "late_delivery",
                "amount": 0.1,
                "unit": "pct_per_day",
                "cap_pct": 10,
            }
        ],
        quality={"lab": "independent", "analysis_paid_by": "buyer"},
        grades=[
            {
                "grade": "3",
                "amount_per_t": 9200,
                "indicators": [{"name": "protein", "limit": "min", "value": 12.5, "unit": "pct"}],
            }
        ],
        buyer="ТОВ Покупець",
    )

    assert deal.crop.value == "wheat"
    assert deal.tolerance == Tolerance(pct=5, option=Party.BUYER)
    assert deal.price == Price(currency=Currency.UAH, vat_included=True)
    assert deal.delivery.date_from == date(2026, 10, 20)
    assert deal.payment == Payment(
        days=5,
        days_type="banking",
        trigger=PaymentTrigger.AFTER_DELIVERY,
    )
    assert deal.penalties == [
        Penalty(
            payer=Party.SELLER,
            trigger=PenaltyTrigger.LATE_DELIVERY,
            amount=0.1,
            unit=PenaltyUnit.PCT_PER_DAY,
            cap_pct=10,
        )
    ]
    assert (deal.quality.lab, deal.quality.analysis_paid_by) == (Lab.INDEPENDENT, Party.BUYER)
    assert deal.grades == [
        GradeTerms(
            grade="3",
            amount_per_t=9200,
            indicators=[QualityIndicator(name="protein", limit="min", value=12.5, unit="pct")],
        )
    ]


def test_deal_defaults_missing_fields_to_none() -> None:
    deal = Deal()

    assert deal.crop is None
    assert deal.quantity_t is None
    assert deal.price is None
    assert deal.penalties is None
    assert deal.model_dump(mode="json")["quality"] is None


@pytest.mark.parametrize(
    "data",
    [
        {"quantity_t": 0},
        {"tolerance": {"pct": 101}},
        {"penalties": [{"amount": 0}]},
        {"quality": {"lab": "elevator"}},
        {"quality": {"analysis_paid_by": "elevator"}},
        {"crop": "rice"},
    ],
)
def test_deal_rejects_invalid_values(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Deal.model_validate(data)


def test_delivery_rejects_reversed_date_range() -> None:
    with pytest.raises(ValidationError, match="date_from must be on or before date_to"):
        Delivery(date_from="2026-10-25", date_to="2026-10-20")


@pytest.mark.parametrize(
    "enum_type",
    [Crop, Party, Currency, Basis, DaysType, PaymentTrigger, PenaltyTrigger, PenaltyUnit, Lab],
)
def test_every_enum_accepts_other(enum_type: type[StrEnum]) -> None:
    assert enum_type("other").value == "other"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3", "3"),
        ("3 клас", "3"),
        ("3кл", "3"),
        ("3-го класу", "3"),
        ("III клас", "3"),
        ("клас 2", "2"),
        ("фуражна", "фураж"),
        ("Фураж", "фураж"),
        ("екстра", "екстра"),
    ],
)
def test_grade_is_normalized_to_vocabulary(raw: str, expected: str) -> None:
    assert GradeTerms(grade=raw).grade == expected


@pytest.mark.parametrize("model", [Price, Quality])
def test_price_and_quality_no_longer_hold_per_grade_values(model: type) -> None:
    assert not {"amount_per_t", "grade", "protein_min"} & set(model.model_fields)


def test_deal_keeps_each_grade_with_its_own_price_and_indicators() -> None:
    deal = Deal(
        price={"currency": "UAH", "vat_included": True},
        grades=[
            {
                "grade": "2-го класу",
                "amount_per_t": 7100,
                "indicators": [
                    {"name": "test_weight", "limit": "min", "value": 770, "unit": "g_per_l"},
                    {"name": "gluten", "limit": "min", "value": 23, "unit": "pct"},
                    {"name": "fusarium", "limit": "max", "value": 0, "unit": "pct"},
                ],
            },
            {"grade": "4", "amount_per_t": 6300},
        ],
    )

    assert [grade.grade for grade in deal.grades] == ["2", "4"]
    assert deal.grades[0].indicators[0] == QualityIndicator(
        name=IndicatorName.TEST_WEIGHT, limit=Limit.MIN, value=770, unit=IndicatorUnit.G_PER_L
    )
    assert deal.grades[1] == GradeTerms(grade="4", amount_per_t=6300)


def test_other_indicator_keeps_its_name_as_written() -> None:
    indicator = QualityIndicator(
        name="other", limit="max", value=4, unit="pct", raw_name="Зерна злакових культур"
    )

    assert indicator.raw_name == "Зерна злакових культур"


@pytest.mark.parametrize(
    "indicators",
    [
        [  # two "other" indicators differ by their names
            {"name": "other", "limit": "max", "value": 4, "unit": "pct", "raw_name": "жито"},
            {"name": "other", "limit": "max", "value": 1, "unit": "pct", "raw_name": "ячмінь"},
        ],
        [  # a range: moisture not below 10% and not above 14%
            {"name": "moisture", "limit": "min", "value": 10, "unit": "pct"},
            {"name": "moisture", "limit": "max", "value": 14, "unit": "pct"},
        ],
    ],
)
def test_grade_accepts_indicators_that_differ_by_name_or_limit(
    indicators: list[dict[str, object]],
) -> None:
    assert len(GradeTerms(indicators=indicators).indicators) == 2


def _one_grade(*indicators: dict[str, object]) -> dict[str, object]:
    return {"grades": [{"indicators": list(indicators)}]}


GLUTEN = {"name": "gluten", "limit": "min", "value": 23, "unit": "pct"}


@pytest.mark.parametrize(
    "data",
    [
        {"grades": [{"amount_per_t": 0}]},
        {"grades": [{"amount_per_t": 7100}]},  # a price needs price.currency
        {"grades": [{"grade": "3"}, {"grade": "3-го класу"}]},
        {"grades": [{}, {}]},  # two grades without a class are one grade
        _one_grade({"name": "gluten", "limit": "min", "unit": "pct"}),  # no value
        _one_grade({**GLUTEN, "value": -1}),
        _one_grade({**GLUTEN, "value": 101}),  # a percent above 100
        _one_grade({**GLUTEN, "name": "other"}),  # "other" without raw_name
        _one_grade({**GLUTEN, "name": "starch"}),
        _one_grade(GLUTEN, {**GLUTEN, "value": 18}),  # the same limit twice
    ],
)
def test_deal_rejects_invalid_grades(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Deal.model_validate(data)


@pytest.mark.parametrize("enum_type", [IndicatorName, IndicatorUnit])
def test_indicator_enums_accept_other(enum_type: type[StrEnum]) -> None:
    assert enum_type("other").value == "other"
