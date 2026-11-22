"""The agent's tool save_reminder: a reminder of a contract's date in the farmer's outbox.

It is the agent's only tool with a side effect, and its YAML says so (side_effect: true): it is
never retried automatically (faults.py), and the graph asks the farmer to confirm each call
before it runs (graph.py). The confirmation comes from outside the model, so the model cannot
give it for the farmer.

Everything that can refuse a call is checked before the confirmation is asked (Tool.check): the
arguments by the schema, then that the date is not in the past and that the contract is the
farmer's own (ContractShelf.require: 403 for another farmer's, 404 for an unknown one, without
reading the PDF). The farmer is asked only about a call that would go through, and the check runs
again just before the write.

The same reminder asked again comes back as "exists" with the key and the note first saved:
reminders.py makes the key and keeps the file free of duplicates.
"""

import logging
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.agents.reminders import MAX_NOTE_CHARS, ReminderKind, ReminderStore
from src.agents.tools import Tool, ToolArgs, ToolRefused, load_tool
from src.servers.contract_access import (
    ID_PATTERN,
    AccessDenied,
    ContractShelf,
    InvalidRequest,
    NotFound,
)

REPO = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO / "tools" / "save_reminder.yaml"

logger = logging.getLogger(__name__)


class SaveReminderArgs(ToolArgs):
    contract_id: str = Field(pattern=f"^{ID_PATTERN}$", description='з list_contracts, "k_001"')
    kind: ReminderKind
    due_date: date = Field(description="дата, яку назвав фермер чи договір: РРРР-ММ-ДД")
    note: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_NOTE_CHARS)
    ]


class SavedReminder(BaseModel):
    """What the model reads back: the reminder, and whether this call wrote it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["saved", "exists"]  # exists: asked before, nothing new was written
    key: str
    contract_id: str
    kind: ReminderKind
    due_date: date
    note: str


def save_reminder_tool(
    store: ReminderStore,
    shelf: ContractShelf,
    *,
    today: Callable[[], date] = date.today,
    now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
) -> Tool[SaveReminderArgs]:
    """The tool for the farmer of `store`; `shelf` must be the same farmer's."""
    if shelf.farmer_id != store.farmer_id:
        raise ValueError(f"shelf of {shelf.farmer_id} for reminders of {store.farmer_id}")

    def check(args: SaveReminderArgs) -> None:
        if args.due_date < today():
            raise ToolRefused(
                f"дата {args.due_date.isoformat()} уже минула (сьогодні "
                f"{today().isoformat()}): спитай фермера, яку дату записати."
            )
        try:
            shelf.require(args.contract_id)
        except (AccessDenied, NotFound, InvalidRequest) as error:
            raise ToolRefused(
                f"{error}. Нагадування не збережено: візьми contract_id із list_contracts."
            ) from None

    def save(args: SaveReminderArgs) -> str:
        try:
            reminder, created = store.save(
                contract_id=args.contract_id,
                kind=args.kind,
                due_date=args.due_date,
                note=args.note,
                now=now(),
            )
        except ValueError as error:  # a broken line: the path goes to the log, not to the model
            logger.error("reminders of farmer %s: %s", store.farmer_id, error)
            raise ToolRefused(
                "Файл нагадувань пошкоджено, нагадування не збережено. Скажи фермеру, що зараз "
                "зберегти не вдалося."
            ) from None
        return SavedReminder(
            status="saved" if created else "exists",
            key=reminder.key,
            contract_id=reminder.contract_id,
            kind=reminder.kind,
            due_date=reminder.due_date,
            note=reminder.note,
        ).model_dump_json()

    return load_tool(TOOL_PATH, SaveReminderArgs, save, output=SavedReminder, check=check)
