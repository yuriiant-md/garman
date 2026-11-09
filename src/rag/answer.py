"""The stage 2 answer: claims, each with its sources, and how Python turns them into citations.

The model sees every context chunk under a short label ("S1", "S2", …) and cites only labels.
Python checks that each cited label was in the context and writes the citation itself from the
chunk's metadata, so a citation cannot name a clause the chunk does not have; the text of a claim
is still the model's. It also adds the warnings the law needs: the Commercial Code has not been
in force since 28.08.2025, and court decisions resting on it are practice under a repealed code.
A wrong answer from the model cannot drop them.
"""

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.rag.chunking import Chunk
from src.rag.sources import DocType

NOT_IN_DOCUMENTS_TEXT = "У документах цього немає."
COMMERCIAL_CODE_REPEALED = "28.08.2025"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    NOT_IN_DOCUMENTS = "not_in_documents"


class Claim(BaseModel):
    """One statement of the answer and the labels of the chunks it rests on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    sources: tuple[str, ...] = Field(min_length=1)


class Answer(BaseModel):
    """What the model returns: claims with sources, or that the documents do not say."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: AnswerStatus
    claims: tuple[Claim, ...] = ()

    @model_validator(mode="after")
    def _claims_match_status(self) -> Self:
        if self.status is AnswerStatus.ANSWERED and not self.claims:
            raise ValueError("an answer needs at least one claim")
        if self.status is AnswerStatus.NOT_IN_DOCUMENTS and self.claims:
            raise ValueError("not_in_documents carries no claims")
        return self


def answer_problem(answer: Answer, context: Mapping[str, Chunk]) -> str | None:
    """Why the answer cannot be accepted for this context; None when it can."""
    cited = (label for claim in answer.claims for label in claim.sources)
    unknown = sorted({label for label in cited if label not in context})
    if unknown:
        return (
            f"the answer cites {', '.join(unknown)}, which are not in the context; "
            f"cite only {', '.join(context)}"
        )
    return None


def citation(chunk: Chunk) -> str:
    """[short title, ref], or the section when the chunk has no ref, or the title alone."""
    if chunk.ref:
        return f"[{chunk.short_title}, {chunk.ref}]"
    if chunk.section:
        return f"[{chunk.short_title}, розділ «{chunk.section.rstrip(':')}»]"
    return f"[{chunk.short_title}]"


def warning(chunk: Chunk) -> str | None:
    """What a reader must know before relying on the chunk; None for a norm in force."""
    if chunk.doc_type is DocType.LAW and chunk.in_force is False:
        return f"{chunk.short_title} втратив чинність: це не чинна норма, а стара редакція."
    if chunk.doc_type is DocType.COURT_DECISION and chunk.cites_commercial_code:
        return f"Це практика за ГК, що втратив чинність {COMMERCIAL_CODE_REPEALED}."
    return None


def render(answer: Answer, context: Mapping[str, Chunk]) -> str:
    """Each claim with its citations, then the warnings of the cited chunks, each once.

    Two chunks may cite alike (two parts of one decision's section have no ref), so citations
    and warnings are made unique by their text, not by their label.
    The answer must have passed answer_problem: every cited label is in the context.
    """
    if answer.status is AnswerStatus.NOT_IN_DOCUMENTS:
        return NOT_IN_DOCUMENTS_TEXT
    lines = [
        " ".join([claim.text, *_unique(citation(context[label]) for label in claim.sources)])
        for claim in answer.claims
    ]
    cited = (context[label] for claim in answer.claims for label in claim.sources)
    warnings = _unique(
        f"⚠ {citation(chunk)}: {text}" for chunk in cited if (text := warning(chunk)) is not None
    )
    return "\n".join([*lines, *([""] + warnings if warnings else [])])


def _unique(texts: Iterable[str]) -> list[str]:
    """The texts in their first order, each once."""
    return list(dict.fromkeys(texts))
