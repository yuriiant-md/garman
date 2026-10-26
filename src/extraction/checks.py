"""Checks of an answer's content that the Deal schema does not express: values that are valid
but implausible for a farm's grain sale.

They run only on model answers, after the schema (validate_deal). Golden deals and stored results
are not re-checked, so a rule here never makes old data unreadable.
"""

import re

from src.extraction.scoring import normalize_text
from src.schemas.deal import Deal, Party, PenaltyTrigger

MAX_FARM_AREA_HA = 5_000  # the largest farm the product serves
MAX_YIELD_T_PER_HA = 10  # corn at its best; wheat and barley give less
MAX_QUANTITY_T = MAX_FARM_AREA_HA * MAX_YIELD_T_PER_HA
# What is left of a hidden name without its legal form and quotes: «назва покупця»,
# «(назва не вказана в тексті)», or an anonymisation mask such as [АДРЕСА]
HIDDEN_NAME = re.compile(r"^(?:\[|\W*назва\b)")


def clean_buyer(deal: Deal) -> Deal:
    """The deal with a placeholder buyer set to null; the model's own answer stays in raw_output."""
    if deal.buyer is None or not is_placeholder(deal.buyer):
        return deal
    return deal.model_copy(update={"buyer": None})


def is_placeholder(name: str) -> bool:
    rest = normalize_text(name)
    return not rest or HIDDEN_NAME.match(rest) is not None


def content_errors(deal: Deal) -> list[str]:
    """What to tell the model about values that pass the schema but cannot be right."""
    return [*penalty_errors(deal), *quantity_errors(deal)]


def penalty_errors(deal: Deal) -> list[str]:
    return [
        f"penalties[{index}].payer: a penalty for late payment is paid by the buyer, who pays "
        "for the goods, not by the seller; check who pays whom in the document"
        for index, penalty in enumerate(deal.penalties or ())
        if penalty.trigger is PenaltyTrigger.LATE_PAYMENT and penalty.payer is Party.SELLER
    ]


def quantity_errors(deal: Deal) -> list[str]:
    if deal.quantity_t is None or deal.quantity_t <= MAX_QUANTITY_T:
        return []
    return [
        f"quantity_t: {deal.quantity_t:g} t is more than a farm of up to {MAX_FARM_AREA_HA} ha "
        f"can sell ({MAX_QUANTITY_T} t). In Ukrainian numbers the comma is the decimal "
        "separator: «400,000 т» is 400 t, not 400 000 t"
    ]
