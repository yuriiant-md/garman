import json
from pathlib import Path

import pytest

from src.extraction.checks import MAX_QUANTITY_T, clean_buyer, content_errors
from src.schemas.deal import Deal

ROOT = Path(__file__).resolve().parents[1]
LABELED = sorted((ROOT / "data" / "golden").glob("*.json")) + sorted(
    (ROOT / "data" / "synthetic").glob("*.json")
)
LATE_PAYMENT = {"trigger": "late_payment", "amount": 2, "unit": "nbu_rate_multiple"}
LATE_DELIVERY = {
    "payer": "seller", "trigger": "late_delivery", "amount": 0.1, "unit": "pct_per_day",
}


def test_late_payment_paid_by_the_seller_is_an_error_that_names_the_penalty():
    deal = Deal(penalties=[LATE_DELIVERY, {**LATE_PAYMENT, "payer": "seller"}])

    (error,) = content_errors(deal)

    assert error.startswith("penalties[1].payer:")
    assert "buyer" in error


@pytest.mark.parametrize(
    "penalty",
    [
        {**LATE_PAYMENT, "payer": "buyer"},
        {**LATE_PAYMENT, "payer": "other"},  # «винна Сторона», own_009
        {**LATE_PAYMENT, "payer": None},
        LATE_DELIVERY,  # the seller pays for its own late delivery
    ],
)
def test_other_penalties_pass(penalty):
    assert content_errors(Deal(penalties=[penalty])) == []


@pytest.mark.parametrize("quantity", [71_000, 400_000])  # own_004 and own_001 read as thousands
def test_quantity_above_what_a_farm_can_sell_is_an_error_with_the_decimal_comma_hint(quantity):
    (error,) = content_errors(Deal(quantity_t=quantity))

    assert error.startswith("quantity_t:")
    assert "decimal" in error


@pytest.mark.parametrize("quantity", [25_000, MAX_QUANTITY_T])  # 25 000 t: own_010
def test_quantity_up_to_the_threshold_passes(quantity):
    assert content_errors(Deal(quantity_t=quantity)) == []


@pytest.mark.parametrize(
    "placeholder",
    [  # what the models wrote for the hidden buyer of the own contracts
        "ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ",
        "ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ “”",
        'ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ ""',
        "ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ “ ”",
        "ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ «назва покупця»",
        "ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ (назва не вказана в тексті)",
        'ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ "" [АДРЕСА], IBAN в АТ "УкрСиббанк"',
    ],
)
def test_placeholder_instead_of_the_buyer_becomes_null(placeholder):
    assert clean_buyer(Deal(buyer=placeholder)).buyer is None


@pytest.mark.parametrize(
    "name", ["Луї Дрейфус Компані Україна", 'ТОВ "Зерно Поле 01"', "Alpine Commodities SA"]
)
def test_a_real_buyer_name_is_kept(name):
    assert clean_buyer(Deal(buyer=name)).buyer == name


def test_cleaning_returns_a_new_deal_and_keeps_the_original():
    deal = Deal(crop="wheat", buyer="ТОВ “”")

    cleaned = clean_buyer(deal)

    assert deal.buyer == "ТОВ “”"
    assert (cleaned.crop, cleaned.buyer) == (deal.crop, None)


@pytest.mark.parametrize("path", LABELED, ids=lambda path: path.stem)
def test_no_labeled_deal_trips_a_check(path):
    deal = Deal.model_validate(json.loads(path.read_text(encoding="utf-8"))["deal"])

    assert content_errors(deal) == []
    assert clean_buyer(deal) == deal
