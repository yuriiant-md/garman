from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas.deal import PenaltyUnit
from src.tools.risk import PenaltyCase, PenaltyRisk, penalty_risk
from tests.tools_fixtures import TODAY, make_playbook, make_rates

MILLION = Decimal("1000000")


def case(**changes: Any) -> PenaltyCase:
    """0.5 % a day of 1 000 000 UAH for 10 days, no cap."""
    values: dict[str, Any] = {
        "sum_uah": MILLION,
        "days": 10,
        "amount": Decimal("0.5"),
        "unit": PenaltyUnit.PCT_PER_DAY,
    }
    return PenaltyCase.model_validate(values | changes)


def risk(penalty: PenaltyCase) -> PenaltyRisk:
    return penalty_risk(penalty, make_playbook(), make_rates(), TODAY)


def test_a_penalty_by_the_day_is_sum_times_percent_times_days() -> None:
    result = risk(case())

    assert result.uah == Decimal("50000.00")  # 1 000 000 × 0.5 % × 10
    assert not result.capped
    assert result.source == "умови договору"


def test_a_penalty_above_the_farmers_threshold_is_flagged() -> None:
    result = risk(case())

    assert result.pct_per_day == Decimal("0.5")
    assert result.above_threshold is True


def test_a_penalty_exactly_at_the_threshold_is_not_flagged() -> None:
    assert risk(case(amount=Decimal("0.1"))).above_threshold is False


def test_the_cap_limits_the_total() -> None:
    result = risk(case(days=30, cap_pct=Decimal("10")))  # 15 % uncapped

    assert result.uah == Decimal("100000.00")
    assert result.capped
    assert "10" in result.formula


def test_a_cap_not_reached_changes_nothing() -> None:
    result = risk(case(cap_pct=Decimal("10")))  # 5 %

    assert result.uah == Decimal("50000.00")
    assert not result.capped


def test_a_multiple_of_the_nbu_rate_uses_the_rate_from_the_config() -> None:
    result = risk(case(days=30, amount=Decimal("2"), unit=PenaltyUnit.NBU_RATE_MULTIPLE))

    assert result.uah == Decimal("25479.45")  # 1 000 000 × 2 × 15.5 % × 30 / 365
    assert result.pct_per_day == Decimal("0.0849")  # 2 × 15.5 / 365
    assert result.above_threshold is False
    assert result.source == "облікова ставка НБУ, 01.11.2026"
    assert not result.stale


def test_a_one_time_penalty_ignores_the_days_and_has_no_daily_rate() -> None:
    result = risk(case(days=90, amount=Decimal("5"), unit=PenaltyUnit.ONE_TIME_PCT))

    assert result.uah == Decimal("50000.00")
    assert result.pct_per_day is None
    assert result.above_threshold is None


def test_no_days_late_costs_nothing_by_the_day() -> None:
    assert risk(case(days=0)).uah == Decimal("0.00")


def test_the_formula_shows_the_numbers() -> None:
    formula = risk(case()).formula

    assert "1000000" in formula
    assert "0.5" in formula
    assert "10" in formula
    assert "50000.00" in formula


@pytest.mark.parametrize(
    "changes",
    [
        {"unit": PenaltyUnit.OTHER},
        {"sum_uah": 0},
        {"days": -1},
        {"amount": 0},
        {"cap_pct": 101},
        {"cap_pct": 0},  # "no cap" is None: a cap of 0 would make every penalty 0.00
        {"days": True},
        {"days": "10"},
        {"payer": "seller"},
    ],
)
def test_a_case_out_of_range_is_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        case(**changes)
