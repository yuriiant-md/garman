from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas.deal import Basis, Crop, Currency, DaysType, PaymentTrigger
from src.tools.price import Discount, EffectivePrice, PriceTerms, effective_price
from tests.tools_fixtures import TODAY, make_playbook, make_rates

WAREHOUSE = "Склад, с. Зелене"


def terms(**changes: Any) -> PriceTerms:
    """10 000 UAH/t without VAT at the farmer's warehouse, prepaid: nothing to take off."""
    values: dict[str, Any] = {
        "crop": Crop.WHEAT,
        "amount_per_t": Decimal("10000"),
        "currency": Currency.UAH,
        "vat_included": False,
        "basis": Basis.FCA,
        "place": WAREHOUSE,
        "trigger": PaymentTrigger.PREPAYMENT,
    }
    return PriceTerms.model_validate(values | changes)


def compute(price_terms: PriceTerms, **playbook_changes: Any) -> EffectivePrice:
    return effective_price(price_terms, make_playbook(**playbook_changes), make_rates(), TODAY)


def amounts(result: EffectivePrice) -> list[Decimal]:
    return [step.uah_per_t for step in result.steps]


# --- the example from docs/PLAN.md ---


def test_cpt_odesa_minus_freight_loses_to_fca_at_the_warehouse() -> None:
    cpt = compute(terms(amount_per_t=Decimal("9200"), basis=Basis.CPT, place="Одеса"))
    fca = compute(terms(amount_per_t=Decimal("8500")))

    assert cpt.uah_per_t == Decimal("8450.00")  # 9 200 − 750 freight
    assert fca.uah_per_t == Decimal("8500.00")
    assert fca.uah_per_t > cpt.uah_per_t


def test_each_step_keeps_its_number_and_source() -> None:
    result = compute(terms(amount_per_t=Decimal("9200"), basis=Basis.CPT, place="порт Одеса"))

    price, freight = result.steps
    assert price.uah_per_t == Decimal("9200.00")
    assert "CPT" in price.label
    assert freight.uah_per_t == Decimal("-750.00")
    assert "Одеса" in freight.label
    assert freight.source == "плейбук фермера, 01.10.2026"


def test_a_freight_older_than_30_days_is_marked_stale() -> None:
    odesa = compute(terms(place="Одеса"))  # 01.10 → 20.11: 50 days
    mykolaiv = compute(terms(place="Миколаїв"))  # 10.11 → 20.11: 10 days

    assert odesa.steps[-1].stale
    assert not mykolaiv.steps[-1].stale
    assert odesa.stale
    assert not mykolaiv.stale


def test_delivery_to_the_warehouse_costs_no_freight() -> None:
    result = compute(terms())

    assert amounts(result) == [Decimal("10000.00"), Decimal("0.00")]
    assert result.uah_per_t == Decimal("10000.00")


# --- currency ---


def test_a_usd_price_is_converted_at_the_nbu_rate() -> None:
    result = compute(terms(amount_per_t=Decimal("200"), currency=Currency.USD))

    price = result.steps[0]
    assert price.uah_per_t == Decimal("8300.00")  # 200 × 41.50
    assert price.source == "курс НБУ, 15.11.2026"
    assert not price.stale


def test_a_currency_without_a_rate_is_reported_missing() -> None:
    rates = make_rates(fx_uah=())

    result = effective_price(terms(currency=Currency.EUR), make_playbook(), rates, TODAY)

    assert result.uah_per_t is None
    assert any("EUR" in gap for gap in result.missing)


def test_other_currency_is_refused() -> None:
    with pytest.raises(ValidationError, match="UAH, USD or EUR"):
        terms(currency=Currency.OTHER)


# --- VAT ---


def test_a_price_with_vat_is_compared_without_it() -> None:
    result = compute(terms(amount_per_t=Decimal("11400"), vat_included=True))

    assert amounts(result) == [Decimal("11400.00"), Decimal("-1400.00"), Decimal("0.00")]
    assert result.uah_per_t == Decimal("10000.00")  # 11 400 / 1.14 for wheat
    assert result.steps[1].source == "ПКУ, ст. 193.1"


def test_a_crop_outside_the_reduced_list_takes_20_percent_vat() -> None:
    result = compute(terms(crop=Crop.SOY, amount_per_t=Decimal("12000"), vat_included=True))

    assert result.uah_per_t == Decimal("10000.00")  # 12 000 / 1.20


def test_a_farmer_comparing_with_vat_gets_it_added() -> None:
    result = compute(terms(), compare_with_vat=True)

    assert amounts(result) == [Decimal("10000.00"), Decimal("1400.00"), Decimal("0.00")]
    assert result.uah_per_t == Decimal("11400.00")


def test_unknown_vat_is_reported_missing() -> None:
    result = compute(terms(vat_included=None))

    assert result.uah_per_t is None
    assert result.steps == ()
    assert any("ПДВ" in gap for gap in result.missing)


# --- freight ---


def test_a_place_not_in_the_playbook_is_reported_with_the_known_places() -> None:
    result = compute(terms(place="Ізмаїл"))

    assert result.uah_per_t is None
    (gap,) = result.missing
    assert "Ізмаїл" in gap
    assert "Одеса" in gap
    assert "Миколаїв" in gap


# --- quality discounts ---


def test_quality_discounts_are_percents_of_the_compared_price() -> None:
    discounts = (
        Discount(reason="вологість", pct=Decimal("1.5")),
        Discount(reason="смітна", pct=Decimal("0.5")),
    )

    result = compute(terms(discounts=discounts))

    assert amounts(result)[-2:] == [Decimal("-150.00"), Decimal("-50.00")]
    assert result.uah_per_t == Decimal("9800.00")
    assert "вологість" in result.steps[-2].label


def test_discounts_over_100_percent_together_are_refused() -> None:
    discounts = [{"reason": "a", "pct": 60}, {"reason": "b", "pct": 41}]

    with pytest.raises(ValidationError, match="100"):
        terms(discounts=discounts)


# --- waiting for the money ---


def test_waiting_30_calendar_days_costs_the_farmers_rate() -> None:
    result = compute(terms(trigger=PaymentTrigger.AFTER_DELIVERY, payment_days=30))

    waiting = result.steps[-1]
    assert waiting.uah_per_t == Decimal("-147.95")  # 10 000 × 18 % × 30 / 365
    assert result.uah_per_t == Decimal("9852.05")
    assert "30" in waiting.label
    assert waiting.source == "ставка фермера: кредит фермера, 01.11.2026"


def test_banking_days_are_counted_as_calendar_days() -> None:
    result = compute(
        terms(trigger=PaymentTrigger.AFTER_DELIVERY, payment_days=10, days_type=DaysType.BANKING)
    )

    assert result.steps[-1].uah_per_t == Decimal("-69.04")  # 14 days: 10 000 × 18 % × 14 / 365
    assert "14" in result.steps[-1].label


def test_payment_after_the_documents_adds_the_days_for_the_originals() -> None:
    result = compute(terms(trigger=PaymentTrigger.AFTER_DOCUMENTS, payment_days=10))

    assert result.steps[-1].uah_per_t == Decimal("-83.84")  # 10 + 7 = 17 days
    assert "17" in result.steps[-1].label
    assert "оригінал" in result.steps[-1].label


def test_without_the_farmers_rate_the_nbu_discount_rate_is_used_and_named() -> None:
    after_delivery = terms(trigger=PaymentTrigger.AFTER_DELIVERY, payment_days=30)

    result = compute(after_delivery, money_rate_pct=None)

    waiting = result.steps[-1]
    assert waiting.uah_per_t == Decimal("-127.40")  # 10 000 × 15.5 % × 30 / 365
    assert "НБУ" in waiting.source


def test_unknown_days_type_is_counted_as_calendar_and_said_so() -> None:
    result = compute(terms(trigger=PaymentTrigger.AFTER_DELIVERY, payment_days=30, days_type=None))

    assert result.steps[-1].uah_per_t == Decimal("-147.95")
    assert "календарн" in result.steps[-1].label


def test_unknown_days_type_after_the_documents_adds_the_days_for_the_originals() -> None:
    result = compute(
        terms(trigger=PaymentTrigger.AFTER_DOCUMENTS, payment_days=10, days_type=DaysType.OTHER)
    )

    assert result.steps[-1].uah_per_t == Decimal("-83.84")  # 10 + 7 = 17 days
    assert "календарн" in result.steps[-1].label
    assert "оригінал" in result.steps[-1].label


def test_an_unwritten_payment_event_is_counted_from_delivery_and_said_so() -> None:
    result = compute(terms(trigger=None, payment_days=30))

    assert result.steps[-1].uah_per_t == Decimal("-147.95")
    assert "від поставки" in result.steps[-1].label


def test_an_other_payment_event_is_reported_missing() -> None:
    result = compute(terms(trigger=PaymentTrigger.OTHER, payment_days=5))

    assert result.uah_per_t is None
    assert any("подія" in gap or "події" in gap for gap in result.missing)


def test_payment_days_unknown_after_delivery_are_reported_missing() -> None:
    result = compute(terms(trigger=PaymentTrigger.AFTER_DELIVERY, payment_days=None))

    assert result.uah_per_t is None
    assert any("строк оплати" in gap for gap in result.missing)


def test_prepayment_costs_no_waiting_even_with_days_written() -> None:
    result = compute(terms(payment_days=30))

    assert result.uah_per_t == Decimal("10000.00")


def test_waiting_is_counted_on_the_price_after_discounts() -> None:
    discounts = (Discount(reason="вологість", pct=Decimal("2")),)

    result = compute(
        terms(trigger=PaymentTrigger.AFTER_DELIVERY, payment_days=30, discounts=discounts)
    )

    assert result.steps[-1].uah_per_t == Decimal("-144.99")  # 9 800 × 18 % × 30 / 365
    assert result.uah_per_t == Decimal("9655.01")


def test_the_vat_part_paid_after_registration_waits_the_extra_days() -> None:
    result = compute(
        terms(amount_per_t=Decimal("11400"), vat_included=True, vat_after_registration=True)
    )

    vat_wait = result.steps[-1]
    assert vat_wait.uah_per_t == Decimal("-20.71")  # 1 400 × 18 % × 30 / 365
    assert "накладн" in vat_wait.label
    assert result.uah_per_t == Decimal("9979.29")


def test_the_vat_part_waits_after_registration_even_with_prepayment() -> None:
    # prepaid without VAT, the VAT part after registration: 1 400 × 18 % × 30 / 365 = 20.71
    result = compute(terms(vat_after_registration=True))

    assert amounts(result) == [Decimal("10000.00"), Decimal("0.00"), Decimal("-20.71")]


def test_the_vat_part_waits_when_the_farmer_compares_with_vat_too() -> None:
    result = compute(terms(vat_after_registration=True), compare_with_vat=True)

    assert amounts(result) == [
        Decimal("10000.00"),
        Decimal("1400.00"),
        Decimal("0.00"),
        Decimal("-20.71"),
    ]
    assert result.uah_per_t == Decimal("11379.29")


# --- the whole result ---


def test_freight_dearer_than_the_price_gives_a_price_below_zero() -> None:
    result = compute(terms(amount_per_t=Decimal("500"), place="Одеса"))

    assert result.uah_per_t == Decimal("-250.00")


def test_every_gap_is_reported_at_once() -> None:
    result = compute(
        terms(place="Ізмаїл", vat_included=None, trigger=PaymentTrigger.AFTER_DELIVERY)
    )

    assert result.uah_per_t is None
    assert len(result.missing) == 3


def test_the_effective_price_is_the_sum_of_its_steps() -> None:
    discounts = (Discount(reason="вологість", pct=Decimal("1.3")),)
    result = compute(
        terms(
            amount_per_t=Decimal("215.37"),
            currency=Currency.USD,
            vat_included=True,
            place="Одеса",
            trigger=PaymentTrigger.AFTER_DOCUMENTS,
            payment_days=9,
            days_type=DaysType.BANKING,
            vat_after_registration=True,
            discounts=discounts,
        )
    )

    assert result.uah_per_t == sum(amounts(result))
    assert len(result.steps) == 6  # price, VAT, freight, discount, waiting, VAT waiting


@pytest.mark.parametrize(
    "changes",
    [
        {"amount_per_t": 0},
        {"amount_per_t": -1},
        {"amount_per_t": Decimal("0.001")},
        {"amount_per_t": Decimal("NaN")},
        {"place": ""},
        {"place": "   "},
        {"payment_days": -1},
        {"payment_days": True},
        {"payment_days": "10"},
        {"vat_included": "yes"},
        {"vat_after_registration": 1},
        {"unknown": 1},
        {"discounts": [{"reason": "a", "pct": 0}]},
        {"discounts": [{"reason": " ", "pct": 1}]},
    ],
)
def test_terms_out_of_range_are_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        terms(**changes)
