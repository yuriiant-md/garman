"""A farmer's profile: the facts the agent keeps from one conversation to the next.

The long-term memory of stage 3; the dialogue of one conversation is short-term (memory.py).
A fact is one key of FactKey and a short value the farmer said yes to (profile_tool.py asks
first), with the time it was written. The farmer is fixed when the store is made, as in
ReminderStore: the model never names whose profile it changes.

Facts go out of date: crops change with the season, a basis with the market. Each key has its
own time to live (configs/agent.yaml, memory.fact_ttl_days). A fact past it is deleted from the
file on the next read, not only hidden, so the file holds only what the agent may still use, and
the agent learns it anew. The store renews a fact written again with the same value; the agent's
tool refuses such a call (profile_tool.py), so the farmer is never asked to confirm what is
already there. The farmer can also forget one key or every fact.

A value is one line: every line break and run of spaces becomes one space, and any other control
character is refused. The profile goes into the system prompt as one bullet a fact
(conversation.py), and a line break in a value could forge a header or another fact there.

SQLite of the standard library: one file for every farmer, one row per farmer and key. Each
call opens the file anew and keeps nothing in the object, so a new session (a new store) reads
what the last one wrote. A read and its write share one write lock (BEGIN IMMEDIATE): a second
writer waits for it. Queries take their values as parameters, never formatted into the SQL.
"""

import logging
import re
import sqlite3
import unicodedata
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from src.servers.contract_access import ID_PATTERN

PROFILE_PATH = Path("data/memory/profile.sqlite")
MAX_VALUE_CHARS = 200  # a fact, not a letter: a longer text belongs in the dialogue
LOCK_TIMEOUT_S = 5.0  # how long a call waits for another writer before it fails
SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    farmer_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (farmer_id, key)
)
"""


class FactKey(StrEnum):
    """What the agent may remember about a farmer; the order is the order the agent reads."""

    REGION = "region"  # the oblast or district of the farm
    CROPS = "crops"  # what the farmer grows and sells
    AREA_HA = "area_ha"
    WAREHOUSE = "warehouse"  # where the grain is kept: freight starts there
    VAT_PAYER = "vat_payer"
    BASIS = "basis"  # the delivery terms the farmer sells on: FCA, CPT, ...
    PREFERRED_PORT = "preferred_port"
    NOTE = "note"  # any other standing wish, e.g. "не продаю без передоплати"


KEY_ORDER = {key: position for position, key in enumerate(FactKey)}
CONTROL = "Cc"  # the Unicode category of control characters: NUL, ESC, DEL, ...

logger = logging.getLogger(__name__)


def one_line(value: object) -> object:
    """A value as one line: whitespace runs, line breaks included, become one space."""
    if not isinstance(value, str):
        return value  # not text: the str field refuses it
    line = " ".join(value.split())  # split() breaks on every Unicode whitespace, U+2028 too
    if any(unicodedata.category(char) == CONTROL for char in line):
        raise ValueError("a fact must not hold a control character")
    return line


# what remember_fact accepts and what a row may hold: one line of 1-200 characters
type FactValue = Annotated[
    str, BeforeValidator(one_line), Field(min_length=1, max_length=MAX_VALUE_CHARS)
]


class BrokenProfile(ValueError):
    """A row of the file is not a fact: the file was changed by hand or by a newer FactKey."""


class Fact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    farmer_id: str = Field(pattern=f"^{ID_PATTERN}$")
    key: FactKey
    value: FactValue  # its length is checked on the one line
    updated_at: AwareDatetime


@dataclass(frozen=True)
class ProfileStore:
    """One farmer's facts in a SQLite file shared by every farmer."""

    path: Path
    farmer_id: str
    ttl_days: Mapping[FactKey, int]  # every key's time to live, in days

    def __post_init__(self) -> None:
        if not re.fullmatch(ID_PATTERN, self.farmer_id):
            raise ValueError(f"invalid farmer id {self.farmer_id!r}: expected {ID_PATTERN}")
        missing = [key.value for key in FactKey if key not in self.ttl_days]
        if missing:
            raise ValueError(f"no ttl_days for: {', '.join(missing)}")
        not_positive = [key.value for key in FactKey if self.ttl_days[key] <= 0]
        if not_positive:
            raise ValueError(f"ttl_days must be positive: {', '.join(not_positive)}")
        # a copy no caller can change: the times to live are fixed with the store
        object.__setattr__(self, "ttl_days", MappingProxyType(dict(self.ttl_days)))

    def facts(self, *, now: datetime) -> tuple[Fact, ...]:
        """The farmer's facts in the order of FactKey; the stale ones are deleted first."""
        _require_zone(now)
        if not self.path.exists():  # nothing remembered yet: a read creates no file
            return ()
        with self._transaction() as connection:
            facts = self._read(connection)
            stale = [fact for fact in facts if self._is_stale(fact, now)]
            connection.executemany(
                "DELETE FROM facts WHERE farmer_id = ? AND key = ?",
                [(self.farmer_id, fact.key.value) for fact in stale],
            )
        fresh = (fact for fact in facts if fact not in stale)
        return tuple(sorted(fresh, key=lambda fact: KEY_ORDER[fact.key]))

    def remember(self, key: FactKey, value: str, *, now: datetime) -> tuple[Fact, Fact | None]:
        """The fact as written, and the one it replaced (None: the key was new)."""
        _require_zone(now)
        # validated before the file is opened: a refused value writes nothing
        fact = Fact(farmer_id=self.farmer_id, key=key, value=value, updated_at=now)
        with self._transaction() as connection:
            previous = next((f for f in self._read(connection) if f.key == fact.key), None)
            connection.execute(
                "INSERT INTO facts (farmer_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (farmer_id, key) "
                "DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (fact.farmer_id, fact.key.value, fact.value, fact.updated_at.isoformat()),
            )
        return fact, previous

    def forget(self, key: FactKey) -> bool:
        """Delete one fact; False when there was none."""
        key = FactKey(key)
        if not self.path.exists():
            return False
        with self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM facts WHERE farmer_id = ? AND key = ?", (self.farmer_id, key.value)
            )
        return cursor.rowcount > 0

    def forget_all(self) -> int:
        """Delete every fact of this farmer, and no one else's; how many there were."""
        if not self.path.exists():
            return 0
        with self._transaction() as connection:
            cursor = connection.execute("DELETE FROM facts WHERE farmer_id = ?", (self.farmer_id,))
        return cursor.rowcount

    def _is_stale(self, fact: Fact, now: datetime) -> bool:
        return now >= fact.updated_at + timedelta(days=self.ttl_days[fact.key])

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """One connection and one write lock for the whole call; any error rolls it back."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: no hidden BEGIN of the sqlite3 module, the one below is ours
        connect = sqlite3.connect(self.path, timeout=LOCK_TIMEOUT_S, isolation_level=None)
        with closing(connect) as connection:
            connection.execute(SCHEMA)
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                _roll_back(connection)
                raise
            connection.execute("COMMIT")

    def _read(self, connection: sqlite3.Connection) -> tuple[Fact, ...]:
        """Every row of this farmer; a broken one is an error, not a fact silently lost."""
        rows = connection.execute(
            "SELECT key, value, updated_at FROM facts WHERE farmer_id = ?", (self.farmer_id,)
        )
        facts = []
        for key, value, updated_at in rows:
            try:
                fact = Fact(farmer_id=self.farmer_id, key=key, value=value, updated_at=updated_at)
            except ValidationError as error:
                raise BrokenProfile(f"{self.path}: row {key!r} is not a fact: {error}") from None
            facts.append(fact)
        return tuple(facts)


def _roll_back(connection: sqlite3.Connection) -> None:
    """Undo the call's writes; a failure here is logged, the error that caused it is raised."""
    try:
        connection.execute("ROLLBACK")
    except sqlite3.Error as error:  # the file is closed next, which rolls back anyway
        logger.error("rollback failed: %s", error)


def _require_zone(now: datetime) -> None:
    # a naive time cannot be compared with the stored aware ones, and would shift by the zone
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must carry a time zone")
