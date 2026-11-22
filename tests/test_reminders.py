import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import IO

import pytest

from src.agents.reminders import Reminder, ReminderKind, ReminderStore, reminder_key

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
DUE = date(2026, 12, 15)


def save(store: ReminderStore, **changes: object) -> tuple[Reminder, bool]:
    fields: dict[str, object] = {
        "contract_id": "k_001",
        "kind": ReminderKind.PAYMENT,
        "due_date": DUE,
        "note": "оплата за пшеницю",
        "now": NOW,
    }
    return store.save(**{**fields, **changes})  # type: ignore[arg-type]


def lines(store: ReminderStore) -> list[dict[str, object]]:
    return [json.loads(line) for line in store.path.read_text(encoding="utf-8").splitlines()]


def test_key_is_the_same_for_the_same_reminder() -> None:
    first = reminder_key("farmer_a", "k_001", ReminderKind.PAYMENT, DUE)
    second = reminder_key("farmer_a", "k_001", ReminderKind.PAYMENT, DUE)

    assert first == second
    assert len(first) == 64


@pytest.mark.parametrize(
    "other",
    [
        ("farmer_b", "k_001", ReminderKind.PAYMENT, DUE),
        ("farmer_a", "k_002", ReminderKind.PAYMENT, DUE),
        ("farmer_a", "k_001", ReminderKind.DELIVERY, DUE),
        ("farmer_a", "k_001", ReminderKind.PAYMENT, date(2026, 12, 16)),
    ],
)
def test_key_differs_when_any_part_differs(
    other: tuple[str, str, ReminderKind, date],
) -> None:
    assert reminder_key("farmer_a", "k_001", ReminderKind.PAYMENT, DUE) != reminder_key(*other)


def test_key_parts_cannot_run_into_each_other() -> None:
    # glued without a separator both would be "farmer_ak1"; "|" is in no id, so they differ
    assert reminder_key("farmer_a", "k1", ReminderKind.OTHER, DUE) != reminder_key(
        "farmer_ak", "1", ReminderKind.OTHER, DUE
    )


def test_first_save_writes_one_record(tmp_path: Path) -> None:
    store = ReminderStore(root=tmp_path, farmer_id="farmer_a")

    reminder, created = save(store)

    assert created is True
    assert store.path == tmp_path / "farmer_a" / "reminders.jsonl"
    assert lines(store) == [json.loads(reminder.model_dump_json())]
    assert reminder.key == reminder_key("farmer_a", "k_001", ReminderKind.PAYMENT, DUE)
    assert reminder.farmer_id == "farmer_a"
    assert reminder.created_at == NOW


def test_same_key_twice_keeps_one_record_and_returns_the_first(tmp_path: Path) -> None:
    store = ReminderStore(root=tmp_path, farmer_id="farmer_a")
    first, _ = save(store)

    # a retry after a timeout, worded differently: still the same reminder
    second, created = save(store, note="ще раз про оплату", now=NOW.replace(hour=13))

    assert created is False
    assert second == first
    assert len(lines(store)) == 1


def test_another_date_is_another_reminder(tmp_path: Path) -> None:
    store = ReminderStore(root=tmp_path, farmer_id="farmer_a")
    save(store)

    _, created = save(store, due_date=date(2026, 12, 20))

    assert created is True
    assert len(store.all()) == 2


def test_parallel_saves_of_one_reminder_write_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ReminderStore(root=tmp_path, farmer_id="farmer_a")
    read = ReminderStore._read

    def slow_read(self: ReminderStore, file: IO[str]) -> tuple[Reminder, ...]:
        found = read(self, file)
        time.sleep(0.05)  # every thread reads before any writes, unless the lock keeps them apart
        return found

    monkeypatch.setattr(ReminderStore, "_read", slow_read)

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: save(store), range(8)))

    assert [created for _, created in outcomes].count(True) == 1
    assert len(lines(store)) == 1


def test_farmers_have_separate_files(tmp_path: Path) -> None:
    store_a = ReminderStore(root=tmp_path, farmer_id="farmer_a")
    store_b = ReminderStore(root=tmp_path, farmer_id="farmer_b")
    save(store_a)

    _, created = save(store_b)

    assert created is True
    assert len(store_a.all()) == 1
    assert len(store_b.all()) == 1


def test_all_is_empty_before_the_first_save(tmp_path: Path) -> None:
    assert ReminderStore(root=tmp_path, farmer_id="farmer_a").all() == ()


@pytest.mark.parametrize("farmer_id", ["", "../farmer_b", "Farmer A", "a/b", "a|b"])
def test_invalid_farmer_id_is_refused(tmp_path: Path, farmer_id: str) -> None:
    with pytest.raises(ValueError, match="farmer id"):
        ReminderStore(root=tmp_path, farmer_id=farmer_id)


def test_a_broken_line_is_an_error_not_a_silent_gap(tmp_path: Path) -> None:
    store = ReminderStore(root=tmp_path, farmer_id="farmer_a")
    save(store)
    with store.path.open("a", encoding="utf-8") as file:
        file.write("not json\n")

    with pytest.raises(ValueError, match="line 2"):
        save(store, due_date=date(2026, 12, 20))


def test_a_blank_line_is_skipped(tmp_path: Path) -> None:
    store = ReminderStore(root=tmp_path, farmer_id="farmer_a")
    save(store)
    with store.path.open("a", encoding="utf-8") as file:
        file.write("\n")

    _, created = save(store, due_date=date(2026, 12, 20))

    assert created is True
    assert len(store.all()) == 2
