from pathlib import Path

import pytest

from src.agents.limits import (
    AGENT_CONFIG_PATH,
    Limits,
    StopReason,
    Usage,
    load_agent_config,
    stop_reason,
)
from src.agents.profile import FactKey
from src.extraction.run import MODELS

LIMITS = Limits(
    max_steps=3,
    max_tokens=1000,
    max_cost_usd=0.01,
    tool_timeout_s=5,
    request_timeout_s=60,
)
PRICES = (1.0, 5.0)  # USD per 1M tokens, input and output

VALID = """\
model: haiku-5.5
limits:
  max_steps: 8
  max_tokens: 50000
  max_cost_usd: 0.01
  tool_timeout_s: 20
  request_timeout_s: 120
memory:
  history_max_tokens: 2000
  summary_max_tokens: 400
  keep_turns: 2
  fact_ttl_days:
    region: 730
    crops: 365
    area_ha: 365
    warehouse: 730
    vat_payer: 365
    basis: 180
    preferred_port: 180
    note: 180
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "agent.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- config


def test_the_config_names_the_model_and_every_limit(tmp_path: Path) -> None:
    config = load_agent_config(write(tmp_path, VALID))

    assert config.model == "haiku-5.5"
    assert config.limits == Limits(
        max_steps=8,
        max_tokens=50000,
        max_cost_usd=0.01,
        tool_timeout_s=20,
        request_timeout_s=120,
    )


def test_the_config_names_the_memory_and_a_ttl_of_every_fact(tmp_path: Path) -> None:
    memory = load_agent_config(write(tmp_path, VALID)).memory

    assert (memory.history_max_tokens, memory.summary_max_tokens, memory.keep_turns) == (
        2000,
        400,
        2,
    )
    assert memory.fact_ttl_days[FactKey.BASIS] == 180
    assert set(memory.fact_ttl_days) == set(FactKey)


def test_the_repository_config_loads_and_names_a_section_of_models_yaml() -> None:
    config = load_agent_config(AGENT_CONFIG_PATH)

    assert config.model in MODELS
    assert set(config.memory.fact_ttl_days) == set(FactKey)


@pytest.mark.parametrize(
    "broken",
    [
        VALID.replace("max_steps: 8", "max_steps: 0"),  # a limit must be above zero
        VALID.replace("max_cost_usd: 0.01", "max_cost_usd: -1"),
        VALID.replace("  max_steps: 8\n", ""),  # every limit is required
        VALID.replace("  request_timeout_s: 120\n", "  request_timeout_s: 120\n  max_retries: 3\n"),
        VALID.replace("history_max_tokens: 2000", "history_max_tokens: 0"),
        VALID.replace("keep_turns: 2", "keep_turns: 0"),
        VALID.replace("summary_max_tokens: 400", "summary_max_tokens: 0"),
        # the summary must leave room in the cap for the turns kept word for word
        VALID.replace("summary_max_tokens: 400", "summary_max_tokens: 2000"),
        VALID.replace("    basis: 180", "    basis: 0"),  # a fact lives at least a day
        VALID.replace("    note: 180\n", ""),  # every key of the profile needs its ttl
        VALID + "    favourite_colour: 30\n",  # a key the profile does not have
        VALID.split("memory:")[0],  # the memory section is required
    ],
)
def test_a_broken_config_is_an_error_that_names_the_file(tmp_path: Path, broken: str) -> None:
    path = write(tmp_path, broken)

    with pytest.raises(ValueError, match="agent.yaml"):
        load_agent_config(path)


def test_a_model_missing_from_models_yaml_is_an_error(tmp_path: Path) -> None:
    path = write(tmp_path, VALID.replace("haiku-5.5", "no-such-model"))

    with pytest.raises(ValueError, match="no-such-model"):
        load_agent_config(path)


# --- usage


def test_a_model_call_adds_one_step_its_tokens_and_their_cost() -> None:
    usage = Usage().add_call(input_tokens=1000, output_tokens=200, prices=PRICES)

    assert (usage.steps, usage.input_tokens, usage.output_tokens) == (1, 1000, 200)
    assert usage.tokens == 1200
    assert usage.cost_usd == pytest.approx(0.002)  # (1000 * 1 + 200 * 5) / 1M


def test_adding_a_call_returns_a_new_usage_and_keeps_the_old_one() -> None:
    before = Usage()

    after = before.add_call(input_tokens=10, output_tokens=10, prices=PRICES)

    assert before == Usage()
    assert after.steps == 1


# --- stop_reason


def test_under_every_limit_the_agent_goes_on() -> None:
    usage = Usage(steps=2, input_tokens=500, output_tokens=100, cost_usd=0.001)

    assert stop_reason(usage, LIMITS, elapsed_s=10) is None


def test_the_step_limit_stops_the_run_once_reached() -> None:
    usage = Usage(steps=3)

    assert stop_reason(usage, LIMITS, elapsed_s=0) == StopReason.MAX_STEPS


def test_the_token_budget_stops_the_run_once_reached() -> None:
    usage = Usage(steps=1, input_tokens=900, output_tokens=100)

    assert stop_reason(usage, LIMITS, elapsed_s=0) == StopReason.BUDGET


def test_the_cost_budget_stops_the_run_once_reached() -> None:
    usage = Usage(steps=1, input_tokens=10, output_tokens=10, cost_usd=0.01)

    assert stop_reason(usage, LIMITS, elapsed_s=0) == StopReason.BUDGET


def test_the_request_timeout_stops_the_run_once_reached() -> None:
    assert stop_reason(Usage(), LIMITS, elapsed_s=60) == StopReason.TIMEOUT


def test_when_several_limits_are_reached_timeout_wins_then_budget() -> None:
    everything = Usage(steps=3, input_tokens=1000, output_tokens=0, cost_usd=0.01)
    no_time_left = 60

    assert stop_reason(everything, LIMITS, elapsed_s=no_time_left) == StopReason.TIMEOUT
    assert stop_reason(everything, LIMITS, elapsed_s=0) == StopReason.BUDGET
