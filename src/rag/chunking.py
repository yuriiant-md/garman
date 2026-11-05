"""Chunks of the stage 2 index: citable units from structure.py packed to a size an embedder takes.

A unit longer than MAX_CHARS is split between lines, a line between sentences, a sentence between
words; every piece keeps the unit's ref, so an answer still cites [own_003, п. 7.3]. A unit shorter
than MIN_CHARS joins its neighbour in the same section, and the ref becomes a range «п. 1.1–1.3».
fixed_size_chunks is the naive baseline that structural chunks are compared with."""

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from datetime import date
from itertools import groupby
from typing import assert_never

from pydantic import BaseModel, ConfigDict, Field

from src.rag.documents import Document, ManifestEntry
from src.rag.sources import DocType, Source
from src.rag.structure import Line, Unit, contract_units, decision_units, law_units

# 94% of contract clauses and nearly every law article fit whole; about 400-500 tokens
MAX_CHARS = 1500
MIN_CHARS = 200  # shorter units (a lone "1.1. Товар.") say too little to be found on their own
FIXED_OVERLAP = 200
SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+")
RANGE_DASH = "–"


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str  # "own_001#007": the document and the chunk's place in it
    doc_id: str
    doc_type: DocType
    title: str
    short_title: str
    source_url: str | None = None
    version_date: date | None = None
    in_force: bool | None = None
    cites_commercial_code: bool | None = None
    text_sha256: str  # of the whole document: a changed document means its chunks are stale
    section: str | None = None
    heading: str | None = None
    ref: str | None = None
    pages: tuple[int, ...] = ()
    text: str = Field(min_length=1)


def chunk_document(
    document: Document,
    source: Source,
    entry: ManifestEntry,
    *,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
) -> tuple[Chunk, ...]:
    """Chunks of one document with its metadata from rag_sources.csv and the manifest."""
    for other in (source.doc_id, entry.doc_id):
        if other != document.doc_id:
            raise ValueError(f"{document.doc_id}: metadata belongs to {other}")
    if entry.doc_type is not source.doc_type:
        raise ValueError(
            f"{document.doc_id}: {source.doc_type} in sources, {entry.doc_type} in manifest"
        )
    units = document_units(document, source.doc_type, max_chars=max_chars)
    packed = pack_units(units, max_chars=max_chars, min_chars=min_chars)
    return tuple(
        _chunk(source, entry, number, unit) for number, unit in enumerate(packed, start=1)
    )


def pack_units(units: Sequence[Unit], *, max_chars: int, min_chars: int) -> tuple[Unit, ...]:
    """Long units split to max_chars, then short ones merged with the unit before them."""
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    pieces = tuple(piece for unit in units for piece in _split_unit(unit, max_chars))
    groups = _pack(pieces, lambda group, unit: _can_merge(group, unit, max_chars, min_chars))
    return tuple(_merged(group) for group in groups)


def fixed_size_chunks(text: str, *, size: int, overlap: int) -> tuple[str, ...]:
    """Windows of `size` characters, each starting `size - overlap` after the one before."""
    spans = fixed_size_spans(len(text), size=size, overlap=overlap)
    return tuple(text[start:end] for start, end in spans)


def fixed_size_spans(length: int, *, size: int, overlap: int) -> tuple[tuple[int, int], ...]:
    """(start, end) of each fixed-size window over a text of `length` characters."""
    if size < 1 or not 0 <= overlap < size:
        raise ValueError(f"fixed-size chunks need size > overlap >= 0, got {size=}, {overlap=}")
    if length == 0:
        return ()
    starts = range(0, max(length - overlap, 1), size - overlap)
    return tuple((start, min(start + size, length)) for start in starts)


def document_units(document: Document, doc_type: DocType, *, max_chars: int) -> tuple[Unit, ...]:
    """The citable units of a document, found by the rules of its type."""
    match doc_type:
        case DocType.CONTRACT:
            return contract_units(document, max_chars=max_chars)
        case DocType.LAW:
            return law_units(document, max_chars=max_chars)
        case DocType.COURT_DECISION:
            return decision_units(document)
        case _:
            assert_never(doc_type)


def _chunk(source: Source, entry: ManifestEntry, number: int, unit: Unit) -> Chunk:
    return Chunk(
        chunk_id=f"{source.doc_id}#{number:03d}",
        doc_id=source.doc_id,
        doc_type=source.doc_type,
        title=source.title,
        short_title=source.short_title,
        source_url=source.source_url,
        version_date=source.version_date,
        in_force=source.in_force,
        cites_commercial_code=source.cites_commercial_code,
        text_sha256=entry.text_sha256,
        section=unit.section,
        heading=unit.heading,
        ref=unit.ref,
        pages=unit.pages,
        text=unit.text,
    )


def _pack[T](
    items: Iterable[T], can_join: Callable[[tuple[T, ...], T], bool]
) -> tuple[tuple[T, ...], ...]:
    """Greedy grouping: an item joins the last group while can_join allows it."""
    groups: list[tuple[T, ...]] = []
    for item in items:
        if groups and can_join(groups[-1], item):
            groups[-1] = (*groups[-1], item)
        else:
            groups.append((item,))
    return tuple(groups)


def _joined_length(texts: Iterable[str]) -> int:
    """Length of the texts joined by one character, a newline between lines or a space."""
    return len("\n".join(texts))


def _split_unit(unit: Unit, max_chars: int) -> tuple[Unit, ...]:
    if len(unit.text) <= max_chars:
        return (unit,)
    lines = tuple(piece for line in unit.lines for piece in _split_line(line, max_chars))
    groups = _pack(
        lines,
        lambda group, line: _joined_length(item.text for item in (*group, line)) <= max_chars,
    )
    return tuple(replace(unit, lines=group) for group in groups)


def _split_line(line: Line, max_chars: int) -> tuple[Line, ...]:
    """A line longer than max_chars, split between sentences; a sentence too long, between words.

    A single word longer than max_chars (a long URL) stays whole: cutting it helps no one."""
    if len(line.text) <= max_chars:
        return (line,)
    atoms = tuple(
        atom
        for sentence in SENTENCE_END.split(line.text)
        for atom in ((sentence,) if len(sentence) <= max_chars else sentence.split())
    )
    groups = _pack(atoms, lambda group, atom: _joined_length((*group, atom)) <= max_chars)
    return tuple(Line(line.page, " ".join(group)) for group in groups)


def _can_merge(group: tuple[Unit, ...], unit: Unit, max_chars: int, min_chars: int) -> bool:
    """Same section and heading, fits max_chars, and one side is too short to stand alone."""
    last = group[-1]
    group_length = _joined_length(item.text for item in group)
    return (
        (last.section, last.heading) == (unit.section, unit.heading)
        and group_length + 1 + len(unit.text) <= max_chars
        and min(group_length, len(unit.text)) < min_chars
    )


def _merged(group: tuple[Unit, ...]) -> Unit:
    if len(group) == 1:
        return group[0]
    return replace(
        group[0],
        lines=tuple(line for unit in group for line in unit.lines),
        ref=_merged_ref(unit.ref for unit in group),
    )


def _merged_ref(refs: Iterable[str | None]) -> str | None:
    """«п. 1.1», «п. 1.2», «п. 1.3» → «п. 1.1–1.3»; other kinds are listed: «преамбула, п. 1.1»."""
    distinct = tuple(dict.fromkeys(ref for ref in refs if ref))
    if not distinct:
        return None
    runs = groupby(distinct, key=_ref_prefix)
    return ", ".join(_ref_range(tuple(run)) for _, run in runs)


def _ref_prefix(ref: str) -> str:
    """Everything before the last number: "ст. 30, ч." of "ст. 30, ч. 2"."""
    return ref.rpartition(" ")[0]


def _ref_range(refs: tuple[str, ...]) -> str:
    if len(refs) == 1:
        return refs[0]
    return f"{refs[0]}{RANGE_DASH}{refs[-1].rpartition(' ')[2]}"
