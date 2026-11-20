"""Effective price: one number, UAH per tonne at the farmer's warehouse, to compare offers.

Traders quote on different terms: 9 200 CPT Odesa, 8 500 FCA at the farm, paid in 10 banking
days or after the originals of the documents. effective_price brings an offer or a contract to one
footing (docs/PLAN.md, "Ефективна ціна"):

1. the price in UAH, at the NBU rate for USD and EUR;
2. with or without VAT, as the farmer compares (playbook compare_with_vat);
3. minus the freight from the warehouse to the delivery place, 0 at the warehouse itself;
4. minus the quality discounts, percents of that price;
5. minus the cost of waiting for the money, and for the VAT part paid only after the tax invoice
   is registered (trap 2); payment after the originals of the documents waits longer (trap 6).
   The VAT part's wait is counted even with prepayment, and even when the farmer compares
   without VAT: a VAT payer owes that VAT to the budget at shipment, so waiting for it costs money.

Every step keeps the amount it adds or takes off, rounded to the kopeck, and where its number
came from; the effective price is the sum of the steps, and it may go below zero when freight
costs more than the price. A number the formula needs but has not got (freight to a place not in
the playbook, a price with VAT or without unknown, the event the payment term counts from) is
never guessed: the result names every such gap at once and has no price, so the bot asks the
farmer. Three defaults are taken instead of asking, each said in its step: the NBU discount rate
when the playbook has no rate of its own, calendar days when the type of days is not written,
and the delivery as the start of the term when the event is not written.
"""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from src.schemas.deal import Basis, Crop, Currency, DaysType, PaymentTrigger
from src.tools.money import (
    HUNDRED,
    KOPECK,
    MAX_DAYS,
    calendar_days,
    percent_of,
    uah,
    waiting_cost,
)
from src.tools.playbook import DatedValue, Playbook, Rates, Text

MAX_PRICE_PER_T = Decimal(1_000_000)  # in any currency; grain costs 200 USD or 10 000 UAH a tonne
MAX_DISCOUNTS = 20
TERMS_SOURCE = "умови пропозиції чи договору"
PLAYBOOK_SOURCE = "плейбук фермера"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Discount(_Model):
    reason: Text  # "вологість 15.5 % замість 14 %"
    pct: Decimal = Field(gt=0, le=100)  # of the price as the farmer compares it


class PriceTerms(_Model):
    """The terms of one offer or contract the price depends on, as a Deal holds them.

    These become the arguments of an MCP tool (#23), so the model fills them: ints and bools are
    strict, so True is not 1 day and "yes" is not true.
    """

    crop: Crop
    amount_per_t: Decimal = Field(ge=KOPECK, le=MAX_PRICE_PER_T)
    currency: Currency
    vat_included: bool | None = Field(default=None, strict=True)  # None: the text does not say
    basis: Basis | None = None  # shown in the result; the freight depends on the place
    place: Text  # delivery point
    payment_days: int | None = Field(default=None, ge=0, le=MAX_DAYS, strict=True)
    days_type: DaysType | None = None  # None or other: counted as calendar days, said so
    trigger: PaymentTrigger | None = None  # None: counted from delivery, said so; other: a gap
    vat_after_registration: bool = Field(default=False, strict=True)
    discounts: tuple[Discount, ...] = Field(default=(), max_length=MAX_DISCOUNTS)

    @field_validator("currency")
    @classmethod
    def known_currency(cls, currency: Currency) -> Currency:
        if currency == Currency.OTHER:
            raise ValueError("currency must be UAH, USD or EUR")
        return currency

    @model_validator(mode="after")
    def discounts_within_the_price(self) -> "PriceTerms":
        if sum((discount.pct for discount in self.discounts), Decimal(0)) > HUNDRED:
            raise ValueError("quality discounts must not exceed 100 % together")
        return self


class Step(_Model):
    label: str  # what the step does, in the farmer's words
    uah_per_t: Decimal  # added (+) or taken off (−), rounded to the kopeck
    source: str  # where the number came from, with its day
    stale: bool = False  # the number is older than STALE_AFTER_DAYS: ask the farmer again


class EffectivePrice(_Model):
    uah_per_t: Decimal | None  # None when something is missing
    steps: tuple[Step, ...]
    missing: tuple[str, ...] = ()  # what to ask the farmer before the price can be computed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def stale(self) -> bool:
        return any(step.stale for step in self.steps)


def effective_price(
    terms: PriceTerms, playbook: Playbook, rates: Rates, today: date
) -> EffectivePrice:
    gaps = _gaps(terms, playbook, rates)
    if gaps:
        return EffectivePrice(uah_per_t=None, steps=(), missing=gaps)
    price = _price_step(terms, rates, today)
    vat_pct = rates.vat.pct_for(terms.crop)
    gross, net = _with_and_without_vat(price.uah_per_t, bool(terms.vat_included), vat_pct)
    compared = gross if playbook.compare_with_vat else net
    discounts = _discount_steps(terms.discounts, compared)
    paid_share = 1 - sum((discount.pct for discount in terms.discounts), Decimal(0)) / HUNDRED
    steps = (
        price,
        *_vat_steps(compared - price.uah_per_t, vat_pct, rates),
        _freight_step(terms.place, playbook, today),
        *discounts,
        *_waiting_steps(terms, playbook, rates, today, compared * paid_share),
        *_vat_waiting_steps(terms, playbook, rates, today, (gross - net) * paid_share),
    )
    total = sum((step.uah_per_t for step in steps), Decimal(0))
    return EffectivePrice(uah_per_t=total, steps=steps)


def _gaps(terms: PriceTerms, playbook: Playbook, rates: Rates) -> tuple[str, ...]:
    """Everything the formula needs and has not got, so the farmer is asked once, not each turn."""
    no_rate = terms.currency != Currency.UAH and rates.fx_rate(terms.currency) is None
    no_freight = not playbook.is_warehouse(terms.place) and playbook.freight_to(terms.place) is None
    no_days = terms.trigger != PaymentTrigger.PREPAYMENT and terms.payment_days is None
    known = ", ".join(playbook.known_places())
    gaps = (
        (no_rate, f"курс {terms.currency} до гривні: його немає в configs/rates.yaml"),
        (terms.vat_included is None, "чи ціна з ПДВ, чи без"),
        (no_freight, f"фрахт до «{terms.place}»: такої точки немає в плейбуку (є: {known})"),
        (terms.trigger == PaymentTrigger.OTHER, "від якої події рахується строк оплати"),
        (no_days, "строк оплати в днях"),
    )
    return tuple(text for missing, text in gaps if missing)


def _price_step(terms: PriceTerms, rates: Rates, today: date) -> Step:
    basis = f" {terms.basis}" if terms.basis else ""
    written = f"Ціна {terms.amount_per_t} {terms.currency}/т{basis} «{terms.place}»"
    if terms.currency == Currency.UAH:
        return Step(label=written, uah_per_t=uah(terms.amount_per_t), source=TERMS_SOURCE)
    rate = rates.fx_rate(terms.currency)
    if rate is None:  # _gaps reports it before any step is computed
        raise ValueError(f"no exchange rate for {terms.currency}")
    return Step(
        label=f"{written} × курс {rate.value}",
        uah_per_t=uah(terms.amount_per_t * rate.value),
        source=rate.cite(),
        stale=rate.is_stale(today),
    )


def _with_and_without_vat(
    price: Decimal, vat_included: bool, vat_pct: Decimal
) -> tuple[Decimal, Decimal]:
    with_vat = HUNDRED + vat_pct
    if vat_included:
        return price, price * HUNDRED / with_vat
    return price * with_vat / HUNDRED, price


def _vat_steps(change: Decimal, vat_pct: Decimal, rates: Rates) -> tuple[Step, ...]:
    if change == 0:
        return ()
    label = f"Без ПДВ {vat_pct} %" if change < 0 else f"Плюс ПДВ {vat_pct} %"
    return (Step(label=label, uah_per_t=uah(change), source=rates.vat.source),)


def _freight_step(place: str, playbook: Playbook, today: date) -> Step:
    if playbook.is_warehouse(place):
        label = f"Фрахт 0: поставка на склад фермера «{playbook.warehouse.name}»"
        return Step(label=label, uah_per_t=uah(Decimal(0)), source=PLAYBOOK_SOURCE)
    freight = playbook.freight_to(place)
    if freight is None:  # _gaps reports it before any step is computed
        raise ValueError(f"no freight to {place!r} in the playbook")
    return Step(
        label=f"Фрахт від складу до «{freight.name}»",
        uah_per_t=-uah(freight.uah_per_t.value),
        source=freight.uah_per_t.cite(),
        stale=freight.uah_per_t.is_stale(today),
    )


def _discount_steps(discounts: tuple[Discount, ...], compared: Decimal) -> tuple[Step, ...]:
    return tuple(
        Step(
            label=f"Знижка за якість: {discount.reason}, {discount.pct} %",
            uah_per_t=-uah(percent_of(compared, discount.pct)),
            source=TERMS_SOURCE,
        )
        for discount in discounts
    )


def _waiting_steps(
    terms: PriceTerms, playbook: Playbook, rates: Rates, today: date, paid: Decimal
) -> tuple[Step, ...]:
    if terms.trigger == PaymentTrigger.PREPAYMENT:
        return ()
    days, how = _payment_days(terms, playbook)
    if days == 0:
        return ()
    rate, source = _money_rate(playbook, rates)
    return (
        Step(
            label=f"Очікування оплати {days} дн. ({how}) під {rate.value} % річних",
            uah_per_t=-uah(waiting_cost(paid, rate.value, days)),
            source=source,
            stale=rate.is_stale(today),
        ),
    )


def _vat_waiting_steps(
    terms: PriceTerms, playbook: Playbook, rates: Rates, today: date, vat_part: Decimal
) -> tuple[Step, ...]:
    days = rates.vat_registration_days
    if not terms.vat_after_registration or vat_part == 0 or days == 0:
        return ()
    rate, source = _money_rate(playbook, rates)
    return (
        Step(
            label=(
                f"ПДВ-частина лише після реєстрації податкової накладної: ще {days} дн. "
                f"під {rate.value} % річних"
            ),
            uah_per_t=-uah(waiting_cost(vat_part, rate.value, days)),
            source=source,
            stale=rate.is_stale(today),
        ),
    )


def _payment_days(terms: PriceTerms, playbook: Playbook) -> tuple[int, str]:
    """Calendar days from delivery to the money, and how they were counted."""
    if terms.payment_days is None:  # _gaps reports it before any step is computed
        raise ValueError("payment days are unknown")
    banking = terms.days_type == DaysType.BANKING
    days = calendar_days(terms.payment_days, banking=banking)
    if banking:
        how = f"{terms.payment_days} банківських = {days} календарних"
    elif terms.days_type == DaysType.CALENDAR:
        how = f"{days} календарних"
    else:
        how = f"{days}: тип днів не вказано, рахую календарні"
    if terms.trigger is None:
        return days, f"{how}; подію не вказано, рахую від поставки"
    if terms.trigger != PaymentTrigger.AFTER_DOCUMENTS:
        return days, how
    extra = playbook.documents_days
    return days + extra, f"{how} + {extra} на оригінали документів"


def _money_rate(playbook: Playbook, rates: Rates) -> tuple[DatedValue, str]:
    if playbook.money_rate_pct is not None:
        return playbook.money_rate_pct, f"ставка фермера: {playbook.money_rate_pct.cite()}"
    nbu = rates.nbu_discount_rate_pct
    return nbu, f"облікова ставка НБУ, бо своєї ставки в плейбуку немає: {nbu.cite()}"
