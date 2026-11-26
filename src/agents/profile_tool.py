"""The agent's tools remember_fact and forget_fact: the farmer's profile, kept between sessions.

Both have a side effect (side_effect: true in their YAML), as save_reminder has: the graph asks
the farmer before each call runs (graph.py, confirm), and none is retried automatically. A
write to the profile changes every later answer, so it needs the farmer's own yes: otherwise a
document with an injected instruction ("запам'ятай: звіти надсилати на …") could plant a fact
that outlives the conversation it came from.

What can refuse a call is checked before the farmer is asked (Tool.check): the arguments by the
schema, then the profile itself. remember_fact refuses a value already stored, forget_fact a key
not stored (or stale), so the farmer is never asked about a call that changes nothing. The check
runs again just before the write.
"""

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.agents.profile import BrokenProfile, Fact, FactKey, FactValue, ProfileStore
from src.agents.tools import Tool, ToolArgs, ToolRefused, load_tool

REPO = Path(__file__).resolve().parents[2]
REMEMBER_PATH = REPO / "tools" / "remember_fact.yaml"
FORGET_PATH = REPO / "tools" / "forget_fact.yaml"
BROKEN = (
    "Профіль пошкоджено, нічого не змінено. Скажи фермеру, що зараз змінити профіль не вдалося."
)

logger = logging.getLogger(__name__)

type Now = Callable[[], datetime]


class RememberFactArgs(ToolArgs):
    key: FactKey
    value: FactValue  # one line, as the profile keeps it: a check compares it with the stored one


class ForgetFactArgs(ToolArgs):
    key: FactKey


class RememberedFact(BaseModel):
    """What the model reads back after a write."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["saved", "updated", "renewed"]  # renewed: the same value, a new date
    key: FactKey
    value: str
    previous: str | None = None  # the value replaced; only for updated


class ForgottenFact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["forgotten"]
    key: FactKey
    value: str  # what was deleted, for the model to tell the farmer


def profile_tools(
    store: ProfileStore, *, now: Now = lambda: datetime.now().astimezone()
) -> tuple[Tool[RememberFactArgs], Tool[ForgetFactArgs]]:
    """remember_fact and forget_fact for the farmer of `store`."""
    return _remember_tool(store, now), _forget_tool(store, now)


def _remember_tool(store: ProfileStore, now: Now) -> Tool[RememberFactArgs]:
    def check(args: RememberFactArgs) -> None:
        stored = _stored(store, args.key, now)
        if stored is not None and stored.value == args.value:
            raise ToolRefused(
                f"Це вже є в профілі: {args.key.value} = «{stored.value}» (від "
                f"{stored.updated_at:%d.%m.%Y}). Нічого не треба зберігати."
            )

    def remember(args: RememberFactArgs) -> str:
        fact, previous = _guarded(lambda: store.remember(args.key, args.value, now=now()))
        if previous is None:
            answer = RememberedFact(status="saved", key=fact.key, value=fact.value)
        elif previous.value == fact.value:  # stored by another call since the check
            answer = RememberedFact(status="renewed", key=fact.key, value=fact.value)
        else:
            answer = RememberedFact(
                status="updated", key=fact.key, value=fact.value, previous=previous.value
            )
        return answer.model_dump_json(exclude_none=True)

    return load_tool(REMEMBER_PATH, RememberFactArgs, remember, output=RememberedFact, check=check)


def _forget_tool(store: ProfileStore, now: Now) -> Tool[ForgetFactArgs]:
    def check(args: ForgetFactArgs) -> None:
        _require_stored(store, args.key, now)

    def forget(args: ForgetFactArgs) -> str:
        fact = _require_stored(store, args.key, now)
        _guarded(lambda: store.forget(args.key))
        return ForgottenFact(status="forgotten", key=fact.key, value=fact.value).model_dump_json()

    return load_tool(FORGET_PATH, ForgetFactArgs, forget, output=ForgottenFact, check=check)


def _stored(store: ProfileStore, key: FactKey, now: Now) -> Fact | None:
    facts = _guarded(lambda: store.facts(now=now()))
    return next((fact for fact in facts if fact.key == key), None)


def _require_stored(store: ProfileStore, key: FactKey, now: Now) -> Fact:
    fact = _stored(store, key, now)
    if fact is None:
        raise ToolRefused(f"У профілі немає факту {key.value}: нічого забувати.")
    return fact


def _guarded[T](call: Callable[[], T]) -> T:
    """The store's answer; a broken file is logged with its path and refused without it."""
    try:
        return call()
    except BrokenProfile as error:
        logger.error("%s", error)
        raise ToolRefused(BROKEN) from None
