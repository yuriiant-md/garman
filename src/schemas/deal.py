import re
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

FEED_GRADE = "фураж"
ROMAN_GRADES = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6"}


class Crop(StrEnum):
    WHEAT = "wheat"
    CORN = "corn"
    BARLEY = "barley"
    SUNFLOWER = "sunflower"
    SOY = "soy"
    RAPESEED = "rapeseed"
    OTHER = "other"


class Party(StrEnum):
    BUYER = "buyer"
    SELLER = "seller"
    OTHER = "other"


class Currency(StrEnum):
    UAH = "UAH"
    USD = "USD"
    EUR = "EUR"
    OTHER = "other"


class Basis(StrEnum):
    EXW = "EXW"
    FCA = "FCA"
    CPT = "CPT"
    DAP = "DAP"
    FOB = "FOB"
    OTHER = "other"


class DaysType(StrEnum):
    CALENDAR = "calendar"
    BANKING = "banking"
    OTHER = "other"


class PaymentTrigger(StrEnum):
    PREPAYMENT = "prepayment"
    AFTER_DELIVERY = "after_delivery"
    AFTER_DOCUMENTS = "after_documents"
    OTHER = "other"


class PenaltyTrigger(StrEnum):
    LATE_DELIVERY = "late_delivery"
    LATE_PAYMENT = "late_payment"
    UNDER_DELIVERY = "under_delivery"
    OTHER = "other"


class PenaltyUnit(StrEnum):
    PCT_PER_DAY = "pct_per_day"
    NBU_RATE_MULTIPLE = "nbu_rate_multiple"
    ONE_TIME_PCT = "one_time_pct"
    OTHER = "other"


class Lab(StrEnum):
    BUYER = "buyer"
    SELLER = "seller"
    INDEPENDENT = "independent"
    OTHER = "other"


def normalize_grade(value: str) -> str:
    """Reduce "3 клас", "III", "фуражна" to the grade vocabulary: "1"…"6" or "фураж"."""
    text = re.sub(r"клас\w*|кл\.?", " ", value.casefold()).strip(" .-")
    if text.startswith(FEED_GRADE):
        return FEED_GRADE
    ordinal = re.fullmatch(r"(\d)\s*-?\s*(?:го|ий|ій|й)?", text)
    if ordinal:
        return ordinal.group(1)
    return ROMAN_GRADES.get(text, text)


class Tolerance(BaseModel):
    pct: float | None = Field(default=None, ge=0, le=100)
    option: Party | None = None


class Price(BaseModel):
    amount_per_t: float | None = Field(default=None, gt=0)
    currency: Currency | None = None
    vat_included: bool | None = None

    @model_validator(mode="after")
    def require_currency_for_amount(self) -> "Price":
        if self.amount_per_t is not None and self.currency is None:
            raise ValueError("currency is required when amount_per_t is provided")
        return self


class Delivery(BaseModel):
    place: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    period_raw: str | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> "Delivery":
        if (
            self.date_from is not None
            and self.date_to is not None
            and self.date_from > self.date_to
        ):
            raise ValueError("date_from must be on or before date_to")
        return self


class Payment(BaseModel):
    days: int | None = None
    days_type: DaysType | None = None
    trigger: PaymentTrigger | None = None
    vat_after_registration: bool | None = None


class Penalty(BaseModel):
    payer: Party | None = None
    trigger: PenaltyTrigger | None = None
    amount: float | None = Field(default=None, gt=0)
    unit: PenaltyUnit | None = None
    cap_pct: float | None = Field(default=None, ge=0, le=100)


class Quality(BaseModel):
    grade: str | None = None
    protein_min: float | None = Field(default=None, ge=0, le=100)
    moisture_max: float | None = Field(default=None, ge=0, le=100)
    foreign_matter_max: float | None = Field(default=None, ge=0, le=100)
    lab: Lab | None = None

    @field_validator("grade")
    @classmethod
    def canonical_grade(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_grade(value) or None


class Deal(BaseModel):
    crop: Crop | None = None
    quantity_t: float | None = Field(default=None, gt=0)
    tolerance: Tolerance | None = None
    price: Price | None = None
    basis: Basis | None = None
    delivery: Delivery | None = None
    payment: Payment | None = None
    penalties: list[Penalty] | None = None
    quality: Quality | None = None
    buyer: str | None = None
