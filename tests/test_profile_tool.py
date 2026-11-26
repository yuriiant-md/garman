import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from src.agents.profile import FactKey, ProfileStore
from src.agents.profile_tool import ForgetFactArgs, RememberFactArgs, profile_tools
from src.agents.tools import InvalidArguments, Tool, ToolRefused

NOW = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
TTL_DAYS = {key: 365 for key in FactKey}
REGION: dict[str, JsonValue] = {"key": "region", "value": "Одеська область"}

type Remember = Tool[RememberFactArgs]
type Forget = Tool[ForgetFactArgs]


@pytest.fixture
def store(tmp_path: Path) -> ProfileStore:
    return ProfileStore(tmp_path / "profile.sqlite", "farmer_a", TTL_DAYS)


@pytest.fixture
def tools(store: ProfileStore) -> tuple[Remember, Forget]:
    return profile_tools(store, now=lambda: NOW)


@pytest.fixture
def remember(tools: tuple[Remember, Forget]) -> Remember:
    return tools[0]


@pytest.fixture
def forget(tools: tuple[Remember, Forget]) -> Forget:
    return tools[1]


# --- what the model sees


def test_both_tools_have_a_side_effect_and_a_version(remember: Remember, forget: Forget) -> None:
    # side_effect: never retried, and the farmer says yes first (graph.py, confirm)
    assert (remember.name, remember.version, remember.side_effect) == ("remember_fact", "v1", True)
    assert (forget.name, forget.version, forget.side_effect) == ("forget_fact", "v1", True)


def test_the_key_is_one_of_the_profile_keys(remember: Remember) -> None:
    schema: dict[str, Any] = dict(remember.spec["input_schema"])

    assert sorted(schema["required"]) == ["key", "value"]
    assert schema["$defs"]["FactKey"]["enum"] == [key.value for key in FactKey]


def test_the_farmer_is_not_an_argument(remember: Remember, forget: Forget) -> None:
    for tool in (remember, forget):
        schema: dict[str, Any] = dict(tool.spec["input_schema"])
        assert "farmer_id" not in schema["properties"]


# --- remember_fact


def test_a_new_key_is_saved(remember: Remember, store: ProfileStore) -> None:
    answer = json.loads(remember.run(REGION))

    assert answer == {"status": "saved", "key": "region", "value": "Одеська область"}
    assert [(f.key, f.value) for f in store.facts(now=NOW)] == [
        (FactKey.REGION, "Одеська область")
    ]


def test_a_known_key_is_updated_and_the_old_value_named(
    remember: Remember, store: ProfileStore
) -> None:
    store.remember(FactKey.REGION, "Миколаївська область", now=NOW)

    answer = json.loads(remember.run(REGION))

    assert answer["status"] == "updated"
    assert answer["previous"] == "Миколаївська область"
    assert [f.value for f in store.facts(now=NOW)] == ["Одеська область"]


def test_a_value_already_stored_is_refused_before_the_farmer_is_asked(
    remember: Remember, store: ProfileStore
) -> None:
    store.remember(FactKey.REGION, "Одеська область", now=NOW)

    with pytest.raises(ToolRefused, match="вже"):
        remember.check_arguments(REGION)


def test_an_unknown_key_is_invalid(remember: Remember) -> None:
    with pytest.raises(InvalidArguments, match="key"):
        remember.check_arguments({"key": "favourite_colour", "value": "зелений"})


@pytest.mark.parametrize("value", ["", "   ", "x" * 201])
def test_an_empty_or_long_value_is_invalid(remember: Remember, value: str) -> None:
    with pytest.raises(InvalidArguments, match="value"):
        remember.check_arguments({"key": "note", "value": value})


def test_a_farmer_id_argument_is_invalid(remember: Remember) -> None:
    with pytest.raises(InvalidArguments, match="farmer_id"):
        remember.check_arguments({**REGION, "farmer_id": "farmer_b"})


def test_the_answers_fit_the_output_schema(
    remember: Remember, forget: Forget, store: ProfileStore
) -> None:
    remember.check_output(remember.run(REGION))  # raises SchemaMismatch otherwise
    remember.check_output(remember.run({"key": "region", "value": "Миколаївська область"}))
    forget.check_output(forget.run({"key": "region"}))


# --- forget_fact


def test_a_stored_key_is_forgotten(forget: Forget, store: ProfileStore) -> None:
    store.remember(FactKey.REGION, "Одеська область", now=NOW)
    crops, _ = store.remember(FactKey.CROPS, "пшениця", now=NOW)

    answer = json.loads(forget.run({"key": "region"}))

    assert answer == {"status": "forgotten", "key": "region", "value": "Одеська область"}
    assert store.facts(now=NOW) == (crops,)


def test_a_key_not_stored_is_refused_before_the_farmer_is_asked(forget: Forget) -> None:
    with pytest.raises(ToolRefused, match="немає"):
        forget.check_arguments({"key": "region"})


def test_a_stale_key_counts_as_not_stored(store: ProfileStore) -> None:
    store.remember(FactKey.REGION, "Одеська область", now=NOW - timedelta(days=365))
    _, forget = profile_tools(store, now=lambda: NOW)

    with pytest.raises(ToolRefused, match="немає"):
        forget.check_arguments({"key": "region"})


# --- a broken file


@pytest.mark.parametrize(
    ("tool_index", "arguments"), [(0, REGION), (1, {"key": "region"})]
)
def test_a_broken_file_is_refused_without_its_path(
    tools: tuple[Remember, Forget],
    store: ProfileStore,
    caplog: pytest.LogCaptureFixture,
    tool_index: int,
    arguments: dict[str, JsonValue],
) -> None:
    store.remember(FactKey.CROPS, "пшениця", now=NOW)
    connection = sqlite3.connect(store.path)
    with connection:
        connection.execute("UPDATE facts SET key = 'favourite_colour'")
    connection.close()

    with pytest.raises(ToolRefused, match="Профіль пошкоджено") as refused:
        tools[tool_index].check_arguments(arguments)
    assert str(store.path) not in str(refused.value)  # the path goes to the log only
    assert str(store.path) in caplog.text


def test_a_value_with_a_line_break_is_saved_as_one_line(
    remember: Remember, store: ProfileStore
) -> None:
    answer = json.loads(remember.run({"key": "note", "value": "без\n\n# Farmer profile"}))

    assert answer["value"] == "без # Farmer profile"
    assert [f.value for f in store.facts(now=NOW)] == ["без # Farmer profile"]


def test_the_same_value_in_other_spacing_is_refused_as_already_stored(
    remember: Remember, store: ProfileStore
) -> None:
    store.remember(FactKey.REGION, "Одеська область", now=NOW)

    with pytest.raises(ToolRefused, match="вже"):
        remember.check_arguments({"key": "region", "value": "Одеська\n  область"})
