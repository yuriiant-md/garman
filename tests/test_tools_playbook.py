from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas.deal import Crop, Currency
from src.tools.playbook import (
    STALE_AFTER_DAYS,
    FxRate,
    Playbook,
    Rates,
    load_playbook,
    load_rates,
)
from tests.tools_fixtures import TODAY, dated, fx_rate, make_playbook, make_rates

PLAYBOOK_YAML = """\
farmer_id: farmer_x
warehouse:
  name: "Склад, с. Тестове"
compare_with_vat: false
penalty_threshold_pct_per_day: 0.1
documents_days: 7
freight:
  - name: "Одеса"
    uah_per_t: {value: 750, as_of: 2026-10-01, source: "плейбук фермера"}
"""


def playbook_data(**changes: Any) -> dict[str, Any]:
    return make_playbook().model_dump() | changes


# --- DatedValue ---


def test_a_value_is_fresh_for_exactly_stale_after_days() -> None:
    value = dated("750", TODAY - timedelta(days=STALE_AFTER_DAYS), "плейбук")

    assert not value.is_stale(TODAY)
    assert value.is_stale(TODAY + timedelta(days=1))


def test_a_value_cites_its_source_and_day() -> None:
    value = dated("750", date(2026, 10, 1), "плейбук фермера")

    assert value.cite() == "плейбук фермера, 01.10.2026"


def test_a_value_dated_in_the_future_is_stale() -> None:
    assert dated("750", TODAY + timedelta(days=1), "плейбук").is_stale(TODAY)


def test_a_value_cannot_be_negative() -> None:
    with pytest.raises(ValidationError):
        dated("-1", TODAY, "плейбук")


@pytest.mark.parametrize("source", ["", "   "])
def test_a_value_needs_a_source(source: str) -> None:
    with pytest.raises(ValidationError):
        dated("750", TODAY, source)


# --- Playbook ---


@pytest.mark.parametrize("place", ["Одеса", "  ОДЕСА ", "порт   Одеса", "Порт одеса"])
def test_freight_is_found_by_name_or_alias_whatever_the_case_and_spaces(place: str) -> None:
    freight = make_playbook().freight_to(place)

    assert freight is not None
    assert freight.name == "Одеса"
    assert freight.uah_per_t.value == Decimal("750")


def test_an_unknown_place_has_no_freight() -> None:
    assert make_playbook().freight_to("Ізмаїл") is None


@pytest.mark.parametrize("place", ["Склад, с. Зелене", "с. зелене"])
def test_the_warehouse_is_recognised_by_name_or_alias(place: str) -> None:
    playbook = make_playbook()

    assert playbook.is_warehouse(place)
    assert playbook.freight_to(place) is None


def test_known_places_list_the_warehouse_and_every_freight_point() -> None:
    assert make_playbook().known_places() == ("Склад, с. Зелене", "Одеса", "Миколаїв")


def test_a_place_listed_twice_is_refused() -> None:
    data = playbook_data(warehouse={"name": "Одеса"})

    with pytest.raises(ValidationError, match="once"):
        Playbook.model_validate(data)


def test_an_alias_repeating_another_place_is_refused() -> None:
    freight = [*playbook_data()["freight"]]
    freight[1] = freight[1] | {"aliases": ["ОДЕСА"]}

    with pytest.raises(ValidationError, match="once"):
        Playbook.model_validate(playbook_data(freight=freight))


@pytest.mark.parametrize("alias", ["", "   "])
def test_a_blank_alias_is_refused(alias: str) -> None:
    data = playbook_data(warehouse={"name": "Склад", "aliases": [alias]})

    with pytest.raises(ValidationError):
        Playbook.model_validate(data)


def test_a_playbook_refuses_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Playbook.model_validate(playbook_data(freight_to_mars=1))


@pytest.mark.parametrize("farmer_id", ["../farmer_a", "Farmer", "a b", "farmer_a\n"])
def test_a_playbook_refuses_a_farmer_id_outside_the_pattern(farmer_id: str) -> None:
    with pytest.raises(ValidationError):
        Playbook.model_validate(playbook_data(farmer_id=farmer_id))


# --- load_playbook ---


def test_load_playbook_reads_the_farmers_file(tmp_path: Path) -> None:
    (tmp_path / "farmer_x.yaml").write_text(PLAYBOOK_YAML, encoding="utf-8")

    playbook = load_playbook("farmer_x", tmp_path)

    assert playbook.farmer_id == "farmer_x"
    assert playbook.money_rate_pct is None
    assert playbook.freight[0].uah_per_t == dated("750", date(2026, 10, 1), "плейбук фермера")


@pytest.mark.parametrize("farmer_id", ["../etc/passwd", "/abs", "farmer_x.yaml", ""])
def test_load_playbook_refuses_an_id_that_is_not_a_plain_name(
    tmp_path: Path, farmer_id: str
) -> None:
    with pytest.raises(ValueError, match="farmer id"):
        load_playbook(farmer_id, tmp_path)


def test_load_playbook_names_the_file_it_did_not_find(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="farmer_y.yaml"):
        load_playbook("farmer_y", tmp_path)


def test_load_playbook_refuses_a_file_of_another_farmer(tmp_path: Path) -> None:
    (tmp_path / "farmer_y.yaml").write_text(PLAYBOOK_YAML, encoding="utf-8")

    with pytest.raises(ValueError, match="farmer_x"):
        load_playbook("farmer_y", tmp_path)


def test_load_playbook_names_the_file_with_a_wrong_field(tmp_path: Path) -> None:
    path = tmp_path / "farmer_x.yaml"
    path.write_text(PLAYBOOK_YAML.replace("documents_days: 7", "documents_days: -1"), "utf-8")

    with pytest.raises(ValueError, match=r"(?s)farmer_x\.yaml.*documents_days"):
        load_playbook("farmer_x", tmp_path)


def test_load_playbook_names_the_file_with_broken_yaml(tmp_path: Path) -> None:
    (tmp_path / "farmer_x.yaml").write_text("farmer_id: [unclosed", encoding="utf-8")

    with pytest.raises(ValueError, match="farmer_x.yaml"):
        load_playbook("farmer_x", tmp_path)


def test_the_example_playbook_in_configs_is_valid() -> None:
    playbook = load_playbook("farmer_a")

    assert playbook.farmer_id == "farmer_a"
    assert playbook.freight


# --- Rates ---


@pytest.mark.parametrize(
    ("crop", "pct"), [(Crop.WHEAT, "14"), (Crop.CORN, "14"), (Crop.SOY, "20"), (Crop.OTHER, "20")]
)
def test_vat_is_reduced_only_for_the_listed_crops(crop: Crop, pct: str) -> None:
    assert make_rates().vat.pct_for(crop) == Decimal(pct)


def rates_with_fx(*rates: FxRate) -> dict[str, Any]:
    return make_rates().model_dump() | {"fx_uah": [rate.model_dump() for rate in rates]}


def test_an_exchange_rate_is_found_by_currency() -> None:
    rate = make_rates().fx_rate(Currency.EUR)

    assert rate is not None
    assert rate.value == Decimal("48.00")
    assert make_rates().fx_rate(Currency.UAH) is None


def test_an_exchange_rate_of_zero_is_refused() -> None:
    with pytest.raises(ValidationError, match="above zero"):
        Rates.model_validate(rates_with_fx(fx_rate(Currency.USD, "0")))


@pytest.mark.parametrize("currency", [Currency.UAH, Currency.OTHER])
def test_rates_refuse_an_exchange_rate_for_uah_or_other(currency: Currency) -> None:
    with pytest.raises(ValidationError, match="USD or EUR"):
        Rates.model_validate(rates_with_fx(fx_rate(currency, "1")))


def test_rates_refuse_two_rates_for_one_currency() -> None:
    data = rates_with_fx(fx_rate(Currency.USD, "41"), fx_rate(Currency.USD, "42"))

    with pytest.raises(ValidationError, match="one exchange rate"):
        Rates.model_validate(data)


def test_the_exchange_rates_cannot_be_changed_in_place() -> None:
    rates = make_rates()

    with pytest.raises((TypeError, AttributeError)):
        rates.fx_uah[0] = fx_rate(Currency.USD, "1")  # type: ignore[index]


def test_load_rates_names_the_file_it_did_not_find(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rates.yaml"):
        load_rates(tmp_path / "rates.yaml")


def test_the_rates_in_configs_are_valid() -> None:
    rates = load_rates()

    assert {rate.currency for rate in rates.fx_uah} == {Currency.USD, Currency.EUR}
    assert rates.vat.pct_for(Crop.WHEAT) == Decimal("14")
