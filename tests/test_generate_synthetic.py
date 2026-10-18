import functools
import itertools
import json
import random
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from scripts import generate_synthetic
from scripts.generate_synthetic import (
    LANGUAGES,
    LENGTHS,
    MAX_ATTEMPTS,
    ROOT,
    TYPOS,
    Dirtiness,
    build_prompt,
    generate_dataset,
    generate_valid_text,
    main,
    make_deal,
    make_dirtiness,
    ollama_generate,
    output_paths,
)
from scripts.synthetic_checks import missing_values
from scripts.synthetic_render import render_conditions
from src.schemas.deal import FEED_GRADE, Crop, Deal, Party, PenaltyTrigger, PenaltyUnit

SEEDS = range(300)
DIRTY = Dirtiness(length="short", language="uk", typos="none", abbreviations=False)
GRADE_VOCABULARY = {"1", "2", "3", "4", "5", "6", FEED_GRADE}
EXPECTED_PAYER = {
    PenaltyTrigger.LATE_DELIVERY: Party.SELLER,
    PenaltyTrigger.UNDER_DELIVERY: Party.SELLER,
    PenaltyTrigger.LATE_PAYMENT: Party.BUYER,
}


def _deals() -> list[Deal]:
    return [make_deal(random.Random(seed), seed) for seed in SEEDS]


def _leaf_values(data: object) -> list[object]:
    if isinstance(data, dict):
        return [leaf for value in data.values() for leaf in _leaf_values(value)]
    if isinstance(data, list):
        return [leaf for value in data for leaf in _leaf_values(value)]
    return [data]


def _echo_conditions(prompt: str) -> str:
    conditions = prompt.split("Умови:\n", 1)[1].split("\n\n", 1)[0]
    return conditions.replace("- ", "")


def test_make_deal_is_reproducible_and_valid() -> None:
    first = make_deal(random.Random(123), index=1)
    second = make_deal(random.Random(123), index=1)

    assert first == second
    assert Deal.model_validate(first.model_dump())
    assert first.buyer is not None


def test_generated_deals_are_wheat_with_concrete_enum_values() -> None:
    deals = _deals()

    assert {deal.crop for deal in deals} <= {Crop.WHEAT, None}
    assert all("other" not in _leaf_values(deal.model_dump(mode="json")) for deal in deals)
    assert {deal.quality.grade for deal in deals if deal.quality} <= GRADE_VOCABULARY


def test_higher_grade_requires_more_protein() -> None:
    qualities = [deal.quality for deal in _deals() if deal.quality]
    protein = {
        grade: {quality.protein_min for quality in qualities if quality.grade == grade}
        for grade in ("2", "3", "4", FEED_GRADE)
    }

    assert max(protein["4"]) < min(protein["3"])
    assert max(protein["3"]) < min(protein["2"])
    assert protein[FEED_GRADE] == {None}


def test_generated_penalties_follow_breach_rules() -> None:
    penalties = [penalty for deal in _deals() for penalty in deal.penalties or []]
    nbu = [penalty for penalty in penalties if penalty.unit == PenaltyUnit.NBU_RATE_MULTIPLE]

    assert all(penalty.payer == EXPECTED_PAYER[penalty.trigger] for penalty in penalties)
    assert nbu and {penalty.amount for penalty in nbu} <= {1.0, 2.0}
    assert all(penalty.trigger == PenaltyTrigger.LATE_PAYMENT for penalty in nbu)
    assert all(
        penalty.amount >= 5 for penalty in penalties if penalty.unit == PenaltyUnit.ONE_TIME_PCT
    )
    assert any(len(deal.penalties or []) == 2 for deal in _deals())


def test_rendered_conditions_pass_the_checker() -> None:
    failures = {
        seed: missing
        for seed, deal in zip(SEEDS, _deals())
        if (missing := missing_values(deal, "\n".join(render_conditions(deal))))
    }

    assert failures == {}


def test_make_dirtiness_varies_every_axis() -> None:
    profiles = [make_dirtiness(random.Random(seed)) for seed in range(100)]

    assert {profile.length for profile in profiles} == set(LENGTHS)
    assert {profile.language for profile in profiles} == set(LANGUAGES)
    assert {profile.typos for profile in profiles} == set(TYPOS)
    assert {profile.abbreviations for profile in profiles} == {True, False}


def test_build_prompt_lists_conditions_in_ukrainian_with_dirtiness() -> None:
    deal = Deal(
        tolerance={"pct": 5, "option": "buyer"},
        payment={"days": 5, "days_type": "banking", "trigger": "after_delivery"},
    )
    dirtiness = Dirtiness(length="line", language="uk_en", typos="few", abbreviations=True)

    prompt = build_prompt(deal, dirtiness)
    conditions = prompt.split("Умови:\n", 1)[1]

    assert "на вибір покупця" in conditions
    assert "5 банківських днів після поставки" in conditions
    assert "after_delivery" not in conditions
    assert LENGTHS["line"] in prompt
    assert LANGUAGES["uk_en"] in prompt


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

    text, attempts = generate_valid_text(deal, DIRTY, generate=generate)

    assert text == "Пшениця 100 тонн, ТОВ Зерно."
    assert attempts == 2
    assert "quantity_t" in prompts[1]


def test_generate_valid_text_fails_after_retry_limit() -> None:
    deal = Deal(crop="wheat", buyer="ТОВ Зерно")

    with pytest.raises(ValueError, match="after 2 attempts"):
        generate_valid_text(deal, DIRTY, generate=lambda _: "ТОВ «Зерно».", max_attempts=2)


def test_generate_dataset_skips_failing_deal_and_saves_the_rest(tmp_path: Path) -> None:
    calls = itertools.count()

    def generate(prompt: str) -> str:
        return "" if next(calls) < MAX_ATTEMPTS else _echo_conditions(prompt)

    skipped = generate_dataset(
        generate, model="llama3.1:8b", seed=7, count=2, examples=1, root=tmp_path
    )

    record = json.loads((tmp_path / "data/synthetic/synthetic_001.json").read_text("utf-8"))
    assert skipped == 1
    assert sorted(path.name for path in (tmp_path / "data/texts").glob("*.txt")) == [
        "synthetic_001.txt",
        "synthetic_002.txt",
    ]
    assert (tmp_path / "data/synthetic/prompt_examples/prompt_example_001.txt").exists()
    assert (record["seed"], record["candidate"]) == (7, 2)
    assert set(record["dirtiness"]) == {"length", "language", "typos", "abbreviations"}


def test_neighbouring_seeds_do_not_share_deals(tmp_path: Path) -> None:
    def deals_for(seed: int) -> list[dict[str, object]]:
        root = tmp_path / str(seed)
        generate_dataset(_echo_conditions, model="llama3.1:8b", seed=seed, count=3, examples=0, root=root)
        records = [
            json.loads(path.read_text("utf-8"))["deal"]
            for path in sorted((root / "data/synthetic").glob("synthetic_*.json"))
        ]
        return [{key: value for key, value in deal.items() if key != "buyer"} for deal in records]

    first, second = deals_for(42), deals_for(43)

    assert not any(deal in first for deal in second)


def test_overwrite_removes_stale_outputs_before_generating(tmp_path: Path) -> None:
    stale = tmp_path / "data/texts/synthetic_002.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("old text\n", encoding="utf-8")
    calls = itertools.count()

    def generate(prompt: str) -> str:
        return _echo_conditions(prompt) if next(calls) == 0 else ""

    with pytest.raises(ValueError):
        generate_dataset(
            generate, model="llama3.1:8b", seed=7, count=2, examples=0, overwrite=True, root=tmp_path
        )

    assert (tmp_path / "data/texts/synthetic_001.txt").exists()
    assert not stale.exists()


def test_main_reports_ollama_timeout_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def time_out(*_args: object) -> str:
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(generate_synthetic, "ollama_generate", time_out)
    # Keep the run away from the repository's data/, which may already hold a dev set.
    monkeypatch.setattr(
        generate_synthetic, "generate_dataset", functools.partial(generate_dataset, root=tmp_path)
    )

    assert main(["--model", "llama3.1:8b", "--count", "1", "--examples", "0"]) == 1
    assert "Generation failed: timed out" in capsys.readouterr().err


def test_ollama_generate_treats_missing_content_as_empty_text() -> None:
    reply = SimpleNamespace(message=SimpleNamespace(content=None))
    client = SimpleNamespace(chat=lambda **_kwargs: reply)

    assert ollama_generate(client, "llama3.1:8b", "prompt") == ""


def test_generate_dataset_keeps_saved_texts_when_candidates_run_out(tmp_path: Path) -> None:
    calls = itertools.count()

    def generate(prompt: str) -> str:
        return _echo_conditions(prompt) if next(calls) == 0 else ""

    with pytest.raises(ValueError, match="1 of 2"):
        generate_dataset(
            generate, model="llama3.1:8b", seed=7, count=2, examples=0, root=tmp_path
        )

    assert (tmp_path / "data/texts/synthetic_001.txt").exists()


@pytest.mark.parametrize("model", ["lapa-v0.1.2-q4km:latest", "gemma3:12b", "qwen2.5:7b"])
def test_main_rejects_models_from_compared_families(
    model: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--model", model]) == 2
    assert "under comparison" in capsys.readouterr().err
