from datetime import date
from pathlib import Path

import pytest

from src.extraction.prompts import load_prompt
from src.schemas.deal import IndicatorName, Quality

ROOT = Path(__file__).resolve().parents[1]


def write_prompt(path: Path, body: str, version: str = "v9") -> Path:
    path.write_text(f"---\nversion: {version}\ndate: 2026-10-18\n---\n\n{body}\n", encoding="utf-8")
    return path


def test_load_prompt_reads_version_and_body_without_header(tmp_path):
    prompt = load_prompt(write_prompt(tmp_path / "p.md", "Витягни умови."))

    assert prompt.version == "v9"
    assert prompt.text == "Витягни умови."


def test_prompt_hash_changes_when_text_changes_under_the_same_version(tmp_path):
    first = load_prompt(write_prompt(tmp_path / "a.md", "Витягни умови."))
    second = load_prompt(write_prompt(tmp_path / "b.md", "Витягни всі умови."))

    assert first.version == second.version
    assert first.sha256 != second.sha256


@pytest.mark.parametrize(
    "content",
    ["Без шапки.\n", "---\ndate: 2026-10-18\n---\nТекст\n", "---\nversion: v1\n---\n\n"],
)
def test_load_prompt_rejects_missing_header_version_or_body(tmp_path, content):
    path = tmp_path / "bad.md"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_prompt(path)


@pytest.mark.parametrize("version", ["v2/fix", "../v2", "v 3", "в3"])
def test_load_prompt_rejects_a_version_that_cannot_name_a_results_file(tmp_path, version):
    path = write_prompt(tmp_path / "p.md", "Витягни умови.", version=version)

    with pytest.raises(ValueError, match="version"):
        load_prompt(path)


@pytest.mark.parametrize("version", ["v3", "v3.1", "v3-fix", "v3_a"])
def test_load_prompt_accepts_letters_digits_dots_dashes_and_underscores(tmp_path, version):
    prompt = load_prompt(write_prompt(tmp_path / "p.md", "Витягни умови.", version=version))

    assert prompt.version == version


def test_today_placeholder_becomes_the_run_date(tmp_path):
    path = write_prompt(tmp_path / "p.md", "Today is {today}.")

    prompt = load_prompt(path, today=date(2026, 10, 20))

    assert prompt.text == "Today is 2026-10-20."


def test_prompt_hash_is_the_same_on_different_run_dates(tmp_path):
    path = write_prompt(tmp_path / "p.md", "Today is {today}.")

    first = load_prompt(path, today=date(2026, 10, 20))
    second = load_prompt(path, today=date(2026, 12, 1))

    assert first.text != second.text
    assert first.sha256 == second.sha256


def test_baseline_prompt_is_version_v1():
    prompt = load_prompt(ROOT / "prompts" / "extract_v1.md")

    assert prompt.version == "v1"
    assert "null" in prompt.text
    assert "other" in prompt.text


def test_prompt_v2_gets_the_run_date_and_keeps_the_null_and_other_rules():
    prompt = load_prompt(ROOT / "prompts" / "extract_v2.md", today=date(2026, 10, 20))

    assert prompt.version == "v2"
    assert "{today}" not in prompt.text
    assert "2026-10-20" in prompt.text
    assert "null" in prompt.text
    assert "other" in prompt.text


@pytest.mark.parametrize(
    "removed", ["protein_min", "moisture_max", "foreign_matter_max", "price.amount_per_t"]
)
def test_prompt_v2_names_no_field_the_schema_dropped(removed):
    assert removed not in load_prompt(ROOT / "prompts" / "extract_v2.md").text


def test_prompt_v2_explains_grades_and_every_indicator_name():
    text = load_prompt(ROOT / "prompts" / "extract_v2.md").text

    assert "**grades**" in text
    assert text.count("- `amount_per_t`:") == 1  # described once, under grades, not under price
    assert all(f"→ {name.value}" in text for name in IndicatorName)


def test_prompt_v2_has_a_rule_for_every_quality_field():
    text = load_prompt(ROOT / "prompts" / "extract_v2.md").text

    assert all(f"- `{field}`:" in text for field in Quality.model_fields)
