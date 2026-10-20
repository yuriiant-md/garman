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
    DAT = "DAT"
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
    pct: float | None = Field(
        default=None, ge=0, le=100, description="Allowed quantity deviation, percent."
    )
    option: Party | None = Field(
        default=None, description="Party that chooses the tolerance (buyer's or seller's option)."
    )


class Price(BaseModel):
    amount_per_t: float | None = Field(default=None, gt=0, description="Price per tonne.")
    currency: Currency | None = None
    vat_included: bool | None = Field(
        default=None, description="True if the price includes VAT, false if it excludes VAT."
    )

    @model_validator(mode="after")
    def require_currency_for_amount(self) -> "Price":
        if self.amount_per_t is not None and self.currency is None:
            raise ValueError("currency is required when amount_per_t is provided")
        return self


class Delivery(BaseModel):
    place: str | None = Field(default=None, description="Delivery point: elevator, port or town.")
    date_from: date | None = Field(default=None, description="First day of the delivery period.")
    date_to: date | None = Field(default=None, description="Last day of the delivery period.")
    period_raw: str | None = Field(
        default=None,
        description=(
            "Relative delivery period copied verbatim, e.g. 'within 10 days after payment'; "
            "date_from and date_to stay null then."
        ),
    )

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
    days: int | None = Field(default=None, description="Number of days to pay.")
    days_type: DaysType | None = Field(default=None, description="Calendar or banking days.")
    trigger: PaymentTrigger | None = Field(
        default=None, description="Event the payment term is counted from."
    )
    vat_after_registration: bool | None = Field(
        default=None,
        description="True if the VAT part is paid only after the tax invoice is registered.",
    )


class Penalty(BaseModel):
    payer: Party | None = Field(default=None, description="Party that pays the penalty.")
    trigger: PenaltyTrigger | None = Field(
        default=None, description="Breach that triggers the penalty."
    )
    amount: float | None = Field(default=None, gt=0, description="Penalty size in the given unit.")
    unit: PenaltyUnit | None = Field(
        default=None,
        description=(
            "pct_per_day: percent of the sum per day; nbu_rate_multiple: multiple of the "
            "NBU discount rate; one_time_pct: one-time percent of the sum."
        ),
    )
    cap_pct: float | None = Field(
        default=None, ge=0, le=100, description="Maximum total penalty, percent of the sum."
    )


class Quality(BaseModel):
    grade: str | None = Field(
        default=None, description="Wheat class as written: '2', '3', '4' or 'фураж'."
    )
    protein_min: float | None = Field(
        default=None, ge=0, le=100, description="Minimum protein, percent."
    )
    moisture_max: float | None = Field(
        default=None, ge=0, le=100, description="Maximum moisture, percent."
    )
    foreign_matter_max: float | None = Field(
        default=None, ge=0, le=100, description="Maximum foreign matter, percent."
    )
    lab: Lab | None = Field(default=None, description="Whose laboratory determines quality.")

    @field_validator("grade")
    @classmethod
    def canonical_grade(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_grade(value) or None


class Deal(BaseModel):
    crop: Crop | None = None
    quantity_t: float | None = Field(default=None, gt=0, description="Quantity in tonnes.")
    tolerance: Tolerance | None = None
    price: Price | None = None
    basis: Basis | None = Field(default=None, description="Incoterms delivery basis.")
    delivery: Delivery | None = None
    payment: Payment | None = None
    penalties: list[Penalty] | None = Field(
        default=None, description="Every penalty stated in the text."
    )
    quality: Quality | None = None
    buyer: str | None = Field(default=None, description="Buyer company name, not a person.")
