import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from src.agents.reminder_tool import SaveReminderArgs, save_reminder_tool
from src.agents.reminders import ReminderStore
from src.agents.tools import InvalidArguments, Tool, ToolRefused
from src.servers.contract_access import ContractShelf

TODAY = date(2026, 11, 22)
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
ARGUMENTS: dict[str, JsonValue] = {
    "contract_id": "k_001",
    "kind": "payment",
    "due_date": "2026-12-15",
    "note": "оплата за пшеницю, п. 5.2",
}

type ReminderTool = Tool[SaveReminderArgs]


@pytest.fixture
def store(tmp_path: Path) -> ReminderStore:
    return ReminderStore(root=tmp_path / "outbox", farmer_id="farmer_a")


@pytest.fixture
def tool(tmp_path: Path, store: ReminderStore) -> ReminderTool:
    """farmer_a owns k_001, farmer_b owns k_900."""
    contracts = tmp_path / "contracts"
    for farmer, contract_id in (("farmer_a", "k_001"), ("farmer_b", "k_900")):
        (contracts / farmer).mkdir(parents=True)
        (contracts / farmer / f"{contract_id}.pdf").write_bytes(b"%PDF-1.7")
    shelf = ContractShelf(root=contracts, farmer_id="farmer_a")
    return save_reminder_tool(store, shelf, today=lambda: TODAY, now=lambda: NOW)


def with_changes(**changes: JsonValue) -> dict[str, JsonValue]:
    return {**ARGUMENTS, **changes}


# --- what the model sees


def test_the_tool_has_a_side_effect_and_a_version(tool: ReminderTool) -> None:
    assert tool.name == "save_reminder"
    assert tool.version == "v1"
    assert tool.side_effect is True  # never retried automatically (faults.py)


def test_every_argument_is_required_and_kind_is_one_of_three(
    tool: ReminderTool,
) -> None:
    schema: dict[str, Any] = dict(tool.spec["input_schema"])

    assert sorted(schema["required"]) == ["contract_id", "due_date", "kind", "note"]
    assert schema["$defs"]["ReminderKind"]["enum"] == ["payment", "delivery", "other"]


def test_the_farmer_is_not_an_argument(tool: ReminderTool) -> None:
    schema: dict[str, Any] = dict(tool.spec["input_schema"])

    assert "farmer_id" not in schema["properties"]


# --- the side effect


def test_a_call_saves_one_reminder(tool: ReminderTool, store: ReminderStore) -> None:
    answer = json.loads(tool.run(ARGUMENTS))

    assert answer["status"] == "saved"
    assert answer["due_date"] == "2026-12-15"
    [reminder] = store.all()
    assert reminder.key == answer["key"]
    assert reminder.note == "оплата за пшеницю, п. 5.2"
    assert reminder.created_at == NOW


def test_the_same_reminder_twice_is_saved_once(
    tool: ReminderTool,
    store: ReminderStore,
) -> None:
    first = json.loads(tool.run(ARGUMENTS))

    second = json.loads(tool.run(with_changes(note="ще раз про оплату")))

    assert second["status"] == "exists"
    assert second["key"] == first["key"]
    assert second["note"] == first["note"]  # the one saved, not the new wording
    assert len(store.all()) == 1


def test_every_answer_passes_the_output_schema(tool: ReminderTool) -> None:
    tool.check_output(tool.run(ARGUMENTS))
    tool.check_output(tool.run(ARGUMENTS))


def test_today_is_not_in_the_past(tool: ReminderTool, store: ReminderStore) -> None:
    tool.run(with_changes(due_date=TODAY.isoformat()))

    assert len(store.all()) == 1


# --- refused before anything is written, and before a confirmation is asked


@pytest.mark.parametrize("missing", ["contract_id", "kind", "due_date", "note"])
def test_a_missing_argument_is_named_and_nothing_is_saved(
    tool: ReminderTool,
    store: ReminderStore,
    missing: str,
) -> None:
    arguments = {name: value for name, value in ARGUMENTS.items() if name != missing}

    with pytest.raises(InvalidArguments, match=missing):
        tool.check_arguments(arguments)
    with pytest.raises(InvalidArguments, match=missing):
        tool.run(arguments)
    assert store.all() == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "price"},
        {"due_date": "15 грудня"},
        {"note": "   "},
        {"note": "а" * 301},
        {"contract_id": "../farmer_b/k_900"},
        {"farmer_id": "farmer_b"},  # the farmer is fixed at start
    ],
)
def test_invalid_arguments_are_refused_and_nothing_is_saved(
    tool: ReminderTool,
    store: ReminderStore,
    changes: dict[str, JsonValue],
) -> None:
    with pytest.raises(InvalidArguments):
        tool.run(with_changes(**changes))
    assert store.all() == ()


def test_a_date_in_the_past_is_refused_and_nothing_is_saved(
    tool: ReminderTool,
    store: ReminderStore,
) -> None:
    arguments = with_changes(due_date="2026-11-21")

    with pytest.raises(ToolRefused, match="2026-11-21"):
        tool.check_arguments(arguments)
    with pytest.raises(ToolRefused):
        tool.run(arguments)
    assert store.all() == ()


@pytest.mark.parametrize(("contract_id", "code"), [("k_900", "403"), ("k_404", "404")])
def test_a_contract_that_is_not_the_farmers_is_refused_and_nothing_is_saved(
    tool: ReminderTool,
    store: ReminderStore,
    contract_id: str,
    code: str,
) -> None:
    arguments = with_changes(contract_id=contract_id)

    with pytest.raises(ToolRefused, match=code):
        tool.check_arguments(arguments)
    with pytest.raises(ToolRefused, match=code):
        tool.run(arguments)
    assert store.all() == ()


def test_check_arguments_never_saves(tool: ReminderTool, store: ReminderStore) -> None:
    tool.check_arguments(ARGUMENTS)

    assert store.all() == ()


def test_a_shelf_of_another_farmer_is_refused_at_start(
    tmp_path: Path, store: ReminderStore
) -> None:
    (tmp_path / "contracts" / "farmer_b").mkdir(parents=True)
    shelf = ContractShelf(root=tmp_path / "contracts", farmer_id="farmer_b")

    with pytest.raises(ValueError, match="farmer_b"):
        save_reminder_tool(store, shelf)


def test_a_broken_file_is_refused_without_showing_its_path(
    tool: ReminderTool, store: ReminderStore, tmp_path: Path
) -> None:
    store.path.parent.mkdir(parents=True)
    store.path.write_text("not json\n", encoding="utf-8")

    with pytest.raises(ToolRefused, match="не збережено") as refused:
        tool.run(ARGUMENTS)
    assert str(tmp_path) not in str(refused.value)
    assert store.path.read_text(encoding="utf-8") == "not json\n"
