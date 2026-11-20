"""A playbook and rates with round numbers for the calculator tests; each test checks by hand."""

from datetime import date
from decimal import Decimal
from typing import Any

from src.schemas.deal import Crop, Currency
from src.tools.playbook import DatedValue, Freight, FxRate, Place, Playbook, Rates, Vat

TODAY = date(2026, 11, 20)


def dated(value: str, as_of: date, source: str) -> DatedValue:
    return DatedValue(value=Decimal(value), as_of=as_of, source=source)


def fx_rate(currency: Currency, value: str) -> FxRate:
    return FxRate(currency=currency, value=Decimal(value), as_of=date(2026, 11, 15), source="курс НБУ")


def make_rates(**changes: Any) -> Rates:
    rates = Rates(
        nbu_discount_rate_pct=dated("15.5", date(2026, 11, 1), "облікова ставка НБУ"),
        fx_uah=(
            fx_rate(Currency.USD, "41.50"),
            fx_rate(Currency.EUR, "48.00"),
        ),
        vat=Vat(
            default_pct=Decimal("20"),
            reduced_pct=Decimal("14"),
            reduced_crops=frozenset({Crop.WHEAT, Crop.CORN, Crop.BARLEY}),
            source="ПКУ, ст. 193.1",
        ),
        vat_registration_days=30,
    )
    return rates.model_copy(update=changes)


def make_playbook(**changes: Any) -> Playbook:
    playbook = Playbook(
        farmer_id="farmer_a",
        warehouse=Place(name="Склад, с. Зелене", aliases=("с. Зелене",)),
        compare_with_vat=False,
        money_rate_pct=dated("18", date(2026, 11, 1), "кредит фермера"),
        penalty_threshold_pct_per_day=Decimal("0.1"),
        documents_days=7,
        freight=(
            Freight(
                name="Одеса",
                aliases=("порт Одеса",),
                uah_per_t=dated("750", date(2026, 10, 1), "плейбук фермера"),
            ),
            Freight(name="Миколаїв", uah_per_t=dated("620", date(2026, 11, 10), "плейбук фермера")),
        ),
    )
    return playbook.model_copy(update=changes)
