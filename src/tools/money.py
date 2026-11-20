"""Money arithmetic of the calculators: Decimal throughout, rounded to the kopeck once per step.

A float cannot hold 0.1 exactly, so 10 000 × 18 % × 30 / 365 in floats ends a hair off the
hand-checked 147.95 and a test on the last kopeck fails. Every amount here is a Decimal. Numbers
that come as floats (a Deal, YAML) reach the calculators through Pydantic, which converts a float
to Decimal through its text, so 0.1 stays 0.1.
"""

from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal

KOPECK = Decimal("0.01")
HUNDRED = Decimal(100)
DAYS_IN_YEAR = 365  # interest and penalties by the day are counted on a 365-day year
# Five banking days take a calendar week. Holidays are not counted: the MVP has no calendar.
CALENDAR_PER_BANKING_DAY = Decimal(7) / Decimal(5)
MAX_DAYS = 730  # payment terms and delays the calculators accept: two years is beyond any contract


def uah(amount: Decimal) -> Decimal:
    """Round to the kopeck, half away from zero: 0.005 → 0.01, −0.005 → −0.01."""
    return amount.quantize(KOPECK, rounding=ROUND_HALF_UP)


def percent_of(amount: Decimal, pct: Decimal) -> Decimal:
    return amount * pct / HUNDRED


def waiting_cost(amount: Decimal, rate_pct_per_year: Decimal, days: int) -> Decimal:
    """What waiting `days` for `amount` costs at a yearly rate: simple interest, not compounded."""
    if days < 0:
        raise ValueError(f"days must not be negative, got {days}")
    return percent_of(amount, rate_pct_per_year) * days / DAYS_IN_YEAR


def calendar_days(days: int, *, banking: bool) -> int:
    """Banking days as calendar days, rounded up: 10 banking days are 14 calendar days."""
    if days < 0:
        raise ValueError(f"days must not be negative, got {days}")
    if not banking:
        return days
    return int((days * CALENDAR_PER_BANKING_DAY).to_integral_value(rounding=ROUND_CEILING))
