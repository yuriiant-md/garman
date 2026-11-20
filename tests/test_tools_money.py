from decimal import Decimal

import pytest

from src.tools.money import calendar_days, percent_of, uah, waiting_cost


def test_uah_rounds_half_up_to_the_kopeck() -> None:
    assert uah(Decimal("147.945205")) == Decimal("147.95")
    assert uah(Decimal("0.005")) == Decimal("0.01")
    assert uah(Decimal("0.004999")) == Decimal("0.00")


def test_uah_rounds_a_deduction_away_from_zero_like_an_addition() -> None:
    assert uah(Decimal("-0.005")) == Decimal("-0.01")


def test_percent_of_is_exact_without_rounding() -> None:
    assert percent_of(Decimal("10000"), Decimal("1.5")) == Decimal("150")
    assert percent_of(Decimal("0.1"), Decimal("0.2")) == Decimal("0.0002")


def test_waiting_cost_is_simple_interest_by_the_day() -> None:
    # 10 000 × 18 % × 30 / 365 = 147.945205…
    cost = waiting_cost(Decimal("10000"), Decimal("18"), 30)

    assert uah(cost) == Decimal("147.95")


def test_waiting_cost_of_no_days_is_zero() -> None:
    assert waiting_cost(Decimal("10000"), Decimal("18"), 0) == 0


def test_waiting_cost_refuses_negative_days() -> None:
    with pytest.raises(ValueError, match="days"):
        waiting_cost(Decimal("10000"), Decimal("18"), -1)


@pytest.mark.parametrize(
    ("banking", "calendar"),
    [(0, 0), (1, 2), (3, 5), (5, 7), (10, 14), (11, 16)],
)
def test_banking_days_take_seven_calendar_days_per_five_rounded_up(
    banking: int, calendar: int
) -> None:
    assert calendar_days(banking, banking=True) == calendar


def test_calendar_days_stay_as_they_are() -> None:
    assert calendar_days(11, banking=False) == 11


def test_calendar_days_refuse_negative_days() -> None:
    with pytest.raises(ValueError, match="days"):
        calendar_days(-1, banking=True)
