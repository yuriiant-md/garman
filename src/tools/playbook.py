"""The numbers the calculators take from outside a deal: a farmer's playbook and the shared rates.

A playbook (configs/playbooks/<farmer_id>.yaml) is one farmer's own: the warehouse, the freight
from it to the usual delivery points, the farmer's cost of money, the penalty the farmer accepts.
The rates (configs/rates.yaml) are the same for every farmer: the NBU discount rate, the NBU
exchange rates, VAT.

A number that goes out of date carries the day it was taken and where from (DatedValue). Every
calculation shows that next to the number and marks it stale after STALE_AFTER_DAYS, so the bot
asks the farmer whether it still holds (docs/PLAN.md, "Звідки числа для формули").
"""

import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from src.schemas.deal import Crop, Currency
from src.servers.contract_access import ID_PATTERN
from src.tools.money import MAX_DAYS

STALE_AFTER_DAYS = 30
PLAYBOOKS_DIR = Path("configs/playbooks")
RATES_PATH = Path("configs/rates.yaml")
MAX_TEXT = 200  # a place name, an alias, a source
FX_CURRENCIES = frozenset({Currency.USD, Currency.EUR})

# A place, an alias, a source, a reason: stripped, so "   " is refused like "" and cannot match
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT)]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DatedValue(_Model):
    value: Decimal = Field(ge=0)
    as_of: date  # the day the number was taken
    source: Text

    def is_stale(self, today: date) -> bool:
        """Older than STALE_AFTER_DAYS, or dated in the future: a typo like 2062 for 2026."""
        age = (today - self.as_of).days
        return age > STALE_AFTER_DAYS or age < 0

    def cite(self) -> str:
        """"плейбук фермера, 01.10.2026": shown next to the number in every calculation."""
        return f"{self.source}, {self.as_of:%d.%m.%Y}"


def normalize_place(place: str) -> str:
    """"  Порт   ОДЕСА " and "порт Одеса" are one place."""
    return " ".join(place.casefold().split())


class Place(_Model):
    name: Text
    aliases: tuple[Text, ...] = ()  # other ways a contract names it: "порт Одеса", "м. Одеса"

    def keys(self) -> tuple[str, ...]:
        return tuple(normalize_place(name) for name in (self.name, *self.aliases))

    def matches(self, place: str) -> bool:
        return normalize_place(place) in self.keys()


class Freight(Place):
    uah_per_t: DatedValue  # from the warehouse to this place, on the footing the farmer compares


class Playbook(_Model):
    farmer_id: str = Field(pattern=f"^{ID_PATTERN}$")
    warehouse: Place
    compare_with_vat: bool  # a VAT payer compares prices without VAT
    money_rate_pct: DatedValue | None = None  # % a year, e.g. the farmer's loan; else NBU's rate
    penalty_threshold_pct_per_day: Decimal = Field(gt=0, le=100)  # a higher penalty is flagged
    documents_days: int = Field(ge=0, le=MAX_DAYS)  # for the originals of the documents to arrive
    freight: tuple[Freight, ...] = ()

    @model_validator(mode="after")
    def each_place_once(self) -> "Playbook":
        keys = [key for place in (self.warehouse, *self.freight) for key in place.keys()]
        if len(keys) != len(set(keys)):
            raise ValueError("each place and alias must appear once, the warehouse included")
        return self

    def is_warehouse(self, place: str) -> bool:
        return self.warehouse.matches(place)

    def freight_to(self, place: str) -> Freight | None:
        return next((freight for freight in self.freight if freight.matches(place)), None)

    def known_places(self) -> tuple[str, ...]:
        return (self.warehouse.name, *(freight.name for freight in self.freight))


class Vat(_Model):
    default_pct: Decimal = Field(ge=0, le=100)
    reduced_pct: Decimal = Field(ge=0, le=100)
    reduced_crops: frozenset[Crop]
    source: Text

    def pct_for(self, crop: Crop) -> Decimal:
        return self.reduced_pct if crop in self.reduced_crops else self.default_pct


class FxRate(DatedValue):
    """UAH for one unit of `currency`, the NBU official rate."""

    currency: Currency


class Rates(_Model):
    nbu_discount_rate_pct: DatedValue
    # A tuple, not a dict: a dict inside a frozen model can still be changed in place
    fx_uah: tuple[FxRate, ...]
    vat: Vat
    vat_registration_days: int = Field(ge=0, le=MAX_DAYS)  # until the VAT part is paid

    @field_validator("fx_uah")
    @classmethod
    def usd_or_eur_once_above_zero(cls, rates: tuple[FxRate, ...]) -> tuple[FxRate, ...]:
        currencies = [rate.currency for rate in rates]
        if not set(currencies) <= FX_CURRENCIES:
            raise ValueError("exchange rates are kept for USD or EUR only")
        if len(currencies) != len(set(currencies)):
            raise ValueError("each currency must have one exchange rate")
        if any(rate.value == 0 for rate in rates):
            raise ValueError("an exchange rate must be above zero")
        return rates

    def fx_rate(self, currency: Currency) -> FxRate | None:
        return next((rate for rate in self.fx_uah if rate.currency == currency), None)


def load_playbook(farmer_id: str, directory: Path = PLAYBOOKS_DIR) -> Playbook:
    """The farmer's playbook, or a ValueError that names the file and what is wrong in it."""
    if not re.fullmatch(ID_PATTERN, farmer_id):
        raise ValueError(f"invalid farmer id {farmer_id!r}")
    path = directory / f"{farmer_id}.yaml"
    playbook = _load(path, Playbook)
    if playbook.farmer_id != farmer_id:
        raise ValueError(f"{path}: farmer_id {playbook.farmer_id!r} does not match the file name")
    return playbook


def load_rates(path: Path = RATES_PATH) -> Rates:
    """The shared rates, or a ValueError that names the file and what is wrong in it."""
    return _load(path, Rates)


def _load[M: BaseModel](path: Path, model: type[M]) -> M:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ValueError(f"{path}: no such file") from error
    try:
        return model.model_validate(yaml.safe_load(text))
    except (yaml.YAMLError, ValidationError) as error:
        raise ValueError(f"{path}: {error}") from error
