"""A farmer's reminders of a contract's deadlines: a local file, the agent's only side effect.

Nothing leaves the machine: a reminder is one line in data/outbox/<farmer_id>/reminders.jsonl,
which the bot of stage 4 will read to remind the farmer. The farmer is fixed when the store is
made, as on the MCP servers: the model never names whose reminder it saves.

The idempotency key is made by Python from what makes a reminder the same (farmer, contract,
kind, date), never by the model, and the note is not part of it: a retry after a timeout, a
reconnect or a farmer asking twice in other words find the reminder already there and get it
back instead of a second one. The check and the write happen under one exclusive lock on the
file, so two calls at once (two threads or two processes) still write it once. The lock is
fcntl.flock: a local file on macOS or Linux, not Windows, not a network share.
"""

import fcntl
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import IO

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.servers.contract_access import ID_PATTERN

OUTBOX_DIR = Path("data/outbox")
REMINDERS_FILE = "reminders.jsonl"
MAX_NOTE_CHARS = 300  # a line the farmer reads in a message, not a letter
KEY_SEPARATOR = "|"  # ID_PATTERN allows no "|", so the joined parts cannot run into each other


class ReminderKind(StrEnum):
    PAYMENT = "payment"  # the trader pays the farmer
    DELIVERY = "delivery"  # the farmer delivers the grain
    OTHER = "other"  # any other date of the contract: price fixing, documents, ...


class Reminder(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(pattern="^[0-9a-f]{64}$")
    farmer_id: str = Field(pattern=f"^{ID_PATTERN}$")
    contract_id: str = Field(pattern=f"^{ID_PATTERN}$")
    kind: ReminderKind
    due_date: date
    note: str = Field(min_length=1, max_length=MAX_NOTE_CHARS)
    created_at: datetime


def reminder_key(farmer_id: str, contract_id: str, kind: ReminderKind, due_date: date) -> str:
    """The same reminder always gets the same key; any other one, another key."""
    parts = (farmer_id, contract_id, kind.value, due_date.isoformat())
    return hashlib.sha256(KEY_SEPARATOR.join(parts).encode()).hexdigest()


@dataclass(frozen=True)
class ReminderStore:
    """One farmer's reminders. Holds no state but the path: every call reads the file anew."""

    root: Path
    farmer_id: str

    def __post_init__(self) -> None:
        if not re.fullmatch(ID_PATTERN, self.farmer_id):
            raise ValueError(f"invalid farmer id {self.farmer_id!r}: expected {ID_PATTERN}")

    @property
    def path(self) -> Path:
        return self.root / self.farmer_id / REMINDERS_FILE

    def all(self) -> tuple[Reminder, ...]:
        if not self.path.exists():
            return ()
        with self.path.open(encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_SH)  # not in the middle of another call's write
            return self._read(file)

    def save(
        self,
        *,
        contract_id: str,
        kind: ReminderKind,
        due_date: date,
        note: str,
        now: datetime,
    ) -> tuple[Reminder, bool]:
        """The reminder with its key, and True when this call wrote it (False: it was there)."""
        key = reminder_key(self.farmer_id, contract_id, kind, due_date)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # "a+": created if missing, read from the start, every write goes to the end
        with self.path.open("a+", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)  # held until the file is closed
            file.seek(0)
            for reminder in self._read(file):
                if reminder.key == key:
                    return reminder, False
            reminder = Reminder(
                key=key,
                farmer_id=self.farmer_id,
                contract_id=contract_id,
                kind=kind,
                due_date=due_date,
                note=note,
                created_at=now,
            )
            file.write(f"{reminder.model_dump_json()}\n")
            return reminder, True

    def _read(self, file: IO[str]) -> tuple[Reminder, ...]:
        """Every line of the file; a broken one is an error: a lost key would let a duplicate in."""
        reminders = []
        for number, line in enumerate(file, start=1):
            if not line.strip():  # an empty line holds no key: nothing is lost by skipping it
                continue
            try:
                reminders.append(Reminder.model_validate_json(line))
            except ValidationError as error:
                raise ValueError(f"{self.path}: line {number} is not a reminder: {error}") from None
        return tuple(reminders)
