import random

import pytest

from scripts.generate_synthetic import (
    ROOT,
    generate_valid_text,
    make_deal,
    missing_values,
    output_paths,
    text_numbers,
)
from src.schemas.deal import Deal


def test_make_deal_is_reproducible_and_valid() -> None:
    first = make_deal(random.Random(123), index=1)
    second = make_deal(random.Random(123), index=1)

    assert first == second
    assert Deal.model_validate(first.model_dump())
    assert first.buyer is not None


def test_missing_values_recognizes_number_format_and_enum_aliases() -> None:
    deal = Deal(
        crop="wheat",
        quantity_t=9200,
        buyer="ТОВ Зерно",
    )

    assert text_numbers("Пшениця, 9 200 тонн. ТОВ «Зерно».") == {9200}
    assert missing_values(deal, "Пшениця, 9 200 тонн. ТОВ «Зерно».") == []


def test_missing_values_reports_unmentioned_field() -> None:
    deal = Deal(crop="corn", buyer="ТОВ Зерно")

    assert missing_values(deal, "ТОВ «Зерно».") == ["crop"]


def test_output_paths_match_prompt_example_output_location() -> None:
    paths = [path.relative_to(ROOT).as_posix() for path in output_paths(1, 1)]

    assert "data/texts/synthetic_001.txt" in paths
    assert "data/synthetic/synthetic_001.json" in paths
    assert "data/synthetic/prompt_examples/prompt_example_001.txt" in paths
    assert "data/synthetic/prompt_example_001.json" in paths


def test_generate_valid_text_retries_with_missing_values() -> None:
    deal = Deal(crop="wheat", quantity_t=100, buyer="ТОВ Зерно")
    prompts: list[str] = []
    responses = iter(("Пшениця, ТОВ Зерно.", "Пшениця 100 тонн, ТОВ Зерно."))

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    text, attempts = generate_valid_text(deal, style_index=0, generate=generate)

    assert text == "Пшениця 100 тонн, ТОВ Зерно."
    assert attempts == 2
    assert "quantity_t" in prompts[1]


def test_generate_valid_text_fails_after_retry_limit() -> None:
    deal = Deal(crop="wheat", buyer="ТОВ Зерно")

    with pytest.raises(ValueError, match="after 2 attempts"):
        generate_valid_text(
            deal,
            style_index=0,
            generate=lambda _: "ТОВ «Зерно».",
            max_attempts=2,
        )
