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
    Lab,
    Party,
    Payment,
    PaymentTrigger,
    Penalty,
    PenaltyTrigger,
    PenaltyUnit,
    Price,
    Quality,
    Tolerance,
)


def test_deal_accepts_valid_nested_data() -> None:
    deal = Deal(
        crop="wheat",
        quantity_t=1000,
        tolerance={"pct": 5, "option": "buyer"},
        price={"amount_per_t": 9200, "currency": "UAH", "vat_included": True},
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
        quality={
            "grade": "3",
            "protein_min": 12.5,
            "moisture_max": 14,
            "foreign_matter_max": 2,
            "lab": "independent",
        },
        buyer="ТОВ Покупець",
    )

    assert deal.crop.value == "wheat"
    assert deal.tolerance == Tolerance(pct=5, option=Party.BUYER)
    assert deal.price == Price(
        amount_per_t=9200, currency=Currency.UAH, vat_included=True
    )
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
    assert deal.quality == Quality(
        grade="3",
        protein_min=12.5,
        moisture_max=14,
        foreign_matter_max=2,
        lab=Lab.INDEPENDENT,
    )


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
        {"price": {"amount_per_t": 0, "currency": "UAH"}},
        {"price": {"amount_per_t": 9200}},
        {"tolerance": {"pct": 101}},
        {"penalties": [{"amount": 0}]},
        {"quality": {"moisture_max": -1}},
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
def test_quality_normalizes_grade_to_vocabulary(raw: str, expected: str) -> None:
    assert Quality(grade=raw).grade == expected
