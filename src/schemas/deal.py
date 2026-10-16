from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


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


class Currency(StrEnum):
    UAH = "UAH"
    USD = "USD"
    EUR = "EUR"


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


class Lab(StrEnum):
    BUYER = "buyer"
    SELLER = "seller"
    INDEPENDENT = "independent"


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
