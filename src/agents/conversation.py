"""A conversation with the farmer: the agent with both memories, one question at a time.

The agent (graph.py) answers one request and remembers nothing after it. This module gives each
request what the agent remembers:
- the dialogue of this conversation (memory.py): its summary and last turns go before the
  question as messages; the turns over the cap are summarized first, by one more model call;
- the farmer's profile (profile.py): the facts the farmer said yes to in earlier sessions, read
  anew for each question, so a fact remembered a moment ago is already there.
Only the profile goes after the system prompt (`context`): the farmer confirmed every fact of
it, one line each. The summary is the model's own text and stays among the messages, as data.

The functions take a Dialogue and return a new one; nothing is kept between calls but the
profile file. A session is a new Dialogue; a new agent may serve it, only the profile carries
over. The agent must have the profile tools of the same store (profile_tool.profile_tools).

A turn enters the dialogue when the request ends: a request waiting for the farmer's yes adds
nothing until confirm(). A partial answer enters as its first line only: the tool results it
quotes stay in the trace, not in every later call.

A profile that cannot be read (a broken row, a file locked past the wait, a folder that cannot
be made) is logged and the question is answered without it: a farmer asking about a contract
should not wait for a repair of the memory.
"""

import logging
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.agents.graph import PARTIAL_HEADERS, Agent
from src.agents.limits import StopReason
from src.agents.memory import Dialogue, Summarizer
from src.agents.profile import BrokenProfile, Fact, ProfileStore
from src.agents.trace import Trace

PROFILE_HEADER = "# Farmer profile"
PROFILE_NOTE = (
    "Facts the farmer confirmed in earlier conversations, with the day each was written. "
    "They are background: what a contract or the farmer says now wins."
)
NO_ANSWER = "Відповіді немає: {reason}."

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Memory:
    """What a conversation of one farmer remembers with: the summarizer and the profile."""

    summarizer: Summarizer
    profile: ProfileStore
    now: Callable[[], datetime] = field(default=lambda: datetime.now().astimezone())


def ask(
    agent: Agent,
    memory: Memory,
    dialogue: Dialogue,
    question: str,
    trace_file: Path | None = None,
) -> tuple[Trace, Dialogue]:
    """One question of the conversation: its trace, and the dialogue to give the next one."""
    compacted, summary = memory.summarizer.compact(dialogue)
    trace = agent.run(
        question,
        trace_file,
        history=compacted.messages(),
        context=context(_facts(memory)),
        summary=summary,
    )
    return trace, _after(trace, compacted)


def confirm(
    agent: Agent,
    dialogue: Dialogue,
    waiting: Trace,
    *,
    approved: bool,
    trace_file: Path | None = None,
) -> tuple[Trace, Dialogue]:
    """The farmer's answer to the question `waiting` asked; `dialogue` is what ask() returned."""
    trace = agent.resume(waiting.thread_id, approved=approved, trace_file=trace_file)
    return trace, _after(trace, dialogue)


def context(facts: Sequence[Fact]) -> str | None:
    """What goes after the system prompt: the profile, a fact a line; None without facts."""
    if not facts:
        return None
    lines = (f"- {f.key.value}: {f.value} (записано {f.updated_at:%d.%m.%Y})" for f in facts)
    return "\n".join((PROFILE_HEADER, PROFILE_NOTE, *lines))


def _facts(memory: Memory) -> tuple[Fact, ...]:
    try:
        return memory.profile.facts(now=memory.now())
    except (BrokenProfile, sqlite3.Error, OSError) as error:
        logger.error("profile not used: %s: %s", type(error).__name__, error)
        return ()


def _after(trace: Trace, dialogue: Dialogue) -> Dialogue:
    """The dialogue once the request ends; unchanged while it waits for the farmer."""
    if trace.stop_reason == StopReason.AWAITING_CONFIRMATION:
        return dialogue
    return dialogue.with_turn(trace.question, _remembered_answer(trace))


def _remembered_answer(trace: Trace) -> str:
    """The answer as later questions read it: whole, or the line a partial answer opens with."""
    if trace.stop_reason == StopReason.ANSWERED and trace.answer:
        return trace.answer
    return PARTIAL_HEADERS.get(trace.stop_reason, NO_ANSWER.format(reason=trace.stop_reason))
