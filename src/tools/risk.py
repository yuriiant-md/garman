"""Risk in UAH of a contract penalty (trap 1): what the farmer pays when late by `days`.

    pct_per_day        sum × pct / 100 × days
    nbu_rate_multiple  sum × multiple × NBU discount rate / 100 × days / 365
    one_time_pct       sum × pct / 100, whatever the days

cap_pct limits the total to a percent of the sum. A daily rate above the farmer's threshold
(playbook penalty_threshold_pct_per_day, 0.1 % in docs/PLAN.md) is flagged; a one-time penalty has
no daily rate to compare. The other traps have no formula: the agent shows them as red flags.
"""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.schemas.deal import PenaltyUnit
from src.tools.money import DAYS_IN_YEAR, MAX_DAYS, percent_of, uah
from src.tools.playbook import Playbook, Rates

MAX_SUM_UAH = Decimal(10_000_000_000)  # a 5 000 ha farm sells for well under a billion a year
MAX_PENALTY = Decimal(100)  # % a day, % at once, or a multiple of the NBU rate
DAILY_RATE_PLACES = Decimal("0.0001")  # 2 × 15.5 / 365 = 0.0849 % a day
TERMS_SOURCE = "умови договору"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PenaltyCase(_Model):
    """One penalty of a contract applied to a sum and a delay."""

    sum_uah: Decimal = Field(gt=0, le=MAX_SUM_UAH)  # the price of the goods or the late payment
    days: int = Field(ge=0, le=MAX_DAYS, strict=True)  # days late; strict: True is not 1 day
    amount: Decimal = Field(gt=0, le=MAX_PENALTY)  # in `unit`
    unit: PenaltyUnit
    # % of the sum the contract caps the penalty at. No cap is None: a model filling "no cap"
    # with 0 would otherwise turn every penalty into 0.00
    cap_pct: Decimal | None = Field(default=None, gt=0, le=100)

    @field_validator("unit")
    @classmethod
    def unit_with_a_formula(cls, unit: PenaltyUnit) -> PenaltyUnit:
        if unit == PenaltyUnit.OTHER:
            raise ValueError("a penalty of unit 'other' has no formula: show it as a red flag")
        return unit


class PenaltyRisk(_Model):
    uah: Decimal  # rounded to the kopeck
    formula: str  # the numbers, so the farmer can check
    source: str  # of the rate: the contract or the NBU discount rate with its day
    stale: bool = False
    capped: bool
    pct_per_day: Decimal | None  # None for a one-time penalty
    above_threshold: bool | None  # None when there is no daily rate to compare


def penalty_risk(case: PenaltyCase, playbook: Playbook, rates: Rates, today: date) -> PenaltyRisk:
    nbu = rates.nbu_discount_rate_pct
    daily = _daily_pct(case, nbu.value)
    if daily is None:
        raw = percent_of(case.sum_uah, case.amount)
    else:
        raw = percent_of(case.sum_uah, daily) * case.days
    cap = None if case.cap_pct is None else percent_of(case.sum_uah, case.cap_pct)
    capped = cap is not None and raw > cap
    total = uah(raw if cap is None else min(raw, cap))
    by_nbu = case.unit == PenaltyUnit.NBU_RATE_MULTIPLE
    return PenaltyRisk(
        uah=total,
        formula=_formula(case, nbu.value, uah(raw), capped, total),
        source=nbu.cite() if by_nbu else TERMS_SOURCE,
        stale=by_nbu and nbu.is_stale(today),
        capped=capped,
        pct_per_day=None if daily is None else daily.quantize(DAILY_RATE_PLACES),
        above_threshold=None if daily is None else daily > playbook.penalty_threshold_pct_per_day,
    )


def _daily_pct(case: PenaltyCase, nbu_pct: Decimal) -> Decimal | None:
    """The penalty as % of the sum a day; None for a one-time penalty."""
    match case.unit:
        case PenaltyUnit.PCT_PER_DAY:
            return case.amount
        case PenaltyUnit.NBU_RATE_MULTIPLE:
            return case.amount * nbu_pct / DAYS_IN_YEAR
        case PenaltyUnit.ONE_TIME_PCT:
            return None
    raise ValueError(f"no formula for a penalty in {case.unit}")  # a new unit must get one


def _formula(
    case: PenaltyCase, nbu_pct: Decimal, raw: Decimal, capped: bool, total: Decimal
) -> str:
    """"1000000 × 0.5 % × 10 дн. = 50000.00": the numbers behind the risk, for the farmer."""
    sum_uah = case.sum_uah
    match case.unit:
        case PenaltyUnit.PCT_PER_DAY:
            text = f"{sum_uah} × {case.amount} % × {case.days} дн. = {raw}"
        case PenaltyUnit.NBU_RATE_MULTIPLE:
            text = (
                f"{sum_uah} × {case.amount} × {nbu_pct} % облікової ставки НБУ"
                f" × {case.days} / {DAYS_IN_YEAR} дн. = {raw}"
            )
        case PenaltyUnit.ONE_TIME_PCT:
            text = f"{sum_uah} × {case.amount} % разово = {raw}"
        case _:
            raise ValueError(f"no formula for a penalty in {case.unit}")
    if capped:
        text += f"; обмежено договором до {case.cap_pct} % суми = {total}"
    return text
