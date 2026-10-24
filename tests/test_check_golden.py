import copy
import json
from pathlib import Path

import pytest

from scripts.check_golden import check_file, main

COMPLETE_DEAL = {
    "crop": "barley",
    "quantity_t": 10.0,
    "tolerance": None,
    "price": {"amount_per_t": 6000.0, "currency": "UAH", "vat_included": False},
    "basis": "other",
    "delivery": {
        "place": None,
        "date_from": None,
        "date_to": None,
        "period_raw": "протягом 5 робочих днів",
    },
    "payment": {
        "days": 30,
        "days_type": "banking",
        "trigger": "after_documents",
        "vat_after_registration": None,
    },
    "penalties": [
        {
            "payer": "seller",
            "trigger": "late_delivery",
            "amount": 0.1,
            "unit": "pct_per_day",
            "cap_pct": None,
        },
    ],
    "quality": None,
    "buyer": "Державна установа «Виправна колонія»",
}


def write_golden(root: Path, deal: dict | None = None, **overrides: object) -> Path:
    (root / "data" / "texts").mkdir(parents=True, exist_ok=True)
    (root / "data" / "texts" / "prozorro_001.txt").write_text("текст договору")
    golden_dir = root / "data" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "id": "prozorro_001",
        "text_path": "data/texts/prozorro_001.txt",
        "split": "test",
        "source": "prozorro",
        "deal": COMPLETE_DEAL if deal is None else deal,
        **overrides,
    }
    path = golden_dir / "prozorro_001.json"
    path.write_text(json.dumps(record, ensure_ascii=False))
    return path


def changed_deal(**changes: object) -> dict:
    deal = copy.deepcopy(COMPLETE_DEAL)
    deal.update(changes)
    return deal


def test_complete_golden_file_has_no_errors(tmp_path):
    assert check_file(write_golden(tmp_path), tmp_path) == []


def test_reports_misspelled_key_that_pydantic_would_drop(tmp_path):
    deal = changed_deal()
    deal["quantity"] = deal.pop("quantity_t")

    errors = check_file(write_golden(tmp_path, deal), tmp_path)

    assert "deal.quantity: unknown key" in errors
    assert any(error.startswith("deal.quantity_t: missing") for error in errors)


def test_reports_unknown_and_missing_keys_inside_penalty(tmp_path):
    penalty = {"payer": "seller", "trigger": "late_delivery", "amount": 0.1, "units": "pct_per_day"}

    errors = check_file(write_golden(tmp_path, changed_deal(penalties=[penalty])), tmp_path)

    assert "deal.penalties[0].units: unknown key" in errors
    assert any(error.startswith("deal.penalties[0].unit: missing") for error in errors)
    assert any(error.startswith("deal.penalties[0].cap_pct: missing") for error in errors)


def test_reports_schema_violation_with_field_path(tmp_path):
    price = {"amount_per_t": 6000.0, "currency": "грн", "vat_included": False}

    errors = check_file(write_golden(tmp_path, changed_deal(price=price)), tmp_path)

    assert any(error.startswith("deal.price.currency:") for error in errors)


def test_reports_id_that_does_not_match_file_name(tmp_path):
    errors = check_file(write_golden(tmp_path, id="prozorro_002"), tmp_path)

    assert errors == ["id 'prozorro_002' does not match file name 'prozorro_001'"]


def test_reports_missing_text_file(tmp_path):
    errors = check_file(write_golden(tmp_path, text_path="data/texts/absent.txt"), tmp_path)

    assert errors == ["text_path data/texts/absent.txt does not exist"]


def test_reports_missing_top_level_keys(tmp_path):
    path = write_golden(tmp_path)
    path.write_text(json.dumps({"id": "prozorro_001"}))

    assert check_file(path, tmp_path) == [
        "missing key: text_path",
        "missing key: split",
        "missing key: source",
        "missing key: deal",
    ]


@pytest.mark.parametrize("key", ["split", "source"])
def test_reports_missing_split_or_source_that_the_metrics_read(tmp_path, key):
    path = write_golden(tmp_path)
    record = json.loads(path.read_text())
    del record[key]
    path.write_text(json.dumps(record, ensure_ascii=False))

    assert check_file(path, tmp_path) == [f"missing key: {key}"]


def test_reports_golden_file_outside_the_test_split(tmp_path):
    path = write_golden(tmp_path, split="dev")

    assert check_file(path, tmp_path) == ["split must be 'test' for a golden file, not 'dev'"]


def test_reports_broken_json(tmp_path):
    path = write_golden(tmp_path)
    path.write_text('{"id": "prozorro_001",')

    errors = check_file(path, tmp_path)

    assert len(errors) == 1 and errors[0].startswith("not valid JSON:")


@pytest.mark.parametrize(("broken", "expected_code"), [(False, 0), (True, 1)])
def test_main_exit_code_reflects_errors(tmp_path, capsys, broken, expected_code):
    deal = changed_deal(crop="пшениця") if broken else None
    write_golden(tmp_path, deal)

    code = main(["--root", str(tmp_path)])

    assert code == expected_code
    assert "prozorro_001" in capsys.readouterr().out


def test_main_without_golden_files_says_so(tmp_path, capsys):
    assert main(["--root", str(tmp_path)]) == 0
    assert "no golden files" in capsys.readouterr().out


def test_reports_json_that_is_not_an_object(tmp_path):
    path = write_golden(tmp_path)
    path.write_text("[]")

    assert check_file(path, tmp_path) == ["file must hold a JSON object"]


@pytest.mark.parametrize("deal", [None, "crop: barley"])
def test_reports_deal_that_is_not_an_object(tmp_path, deal):
    path = write_golden(tmp_path)
    record = {**json.loads(path.read_text()), "deal": deal}
    path.write_text(json.dumps(record))

    assert check_file(path, tmp_path) == ["deal must be a JSON object"]
