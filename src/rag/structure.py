"""Citable units of an ingested document: what an answer can point to as [own_003, п. 7.3].

A contract splits into its preamble, clauses (7.3 with its subclauses 7.3.1) and requisites; a law
into articles, or the parts of an article longer than a chunk; a decision into its paragraphs.
Each unit keeps its section and the page of every line. Sizes are chunking.py's job: units are
packed into chunks there, so the rules for finding clauses stay apart from the size limits."""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

from src.rag.documents import Document, DocumentPart

PREAMBLE_REF = "преамбула"
REQUISITES_SECTION = "Реквізити та додатки"
MAX_HEADING_CHARS = 100
MIN_HEADING_LETTERS = 4  # "ВСТАНОВИВ:" is a heading, a judge's "О. [ПІБ]" is not
FINAL_PUNCTUATION = (".", ":", ";", ",", "!", "?")

# "7.3. Текст", "7.3 .Текст" (OCR) or "1.1." alone; "7.3.1." is a subclause and stays inside 7.3
CLAUSE = re.compile(r"(\d{1,2})\.(\d{1,2})(?:\s*\.)?(?=\s|$|[^\d.])")
SUBCLAUSE = re.compile(r"(\d{1,2}\.\d{1,2})\.(\d{1,2})(?:\s*\.)?(?=\s|$|[^\d.])")
CLAUSE_REF = "п. "
# "7. Відповідальність сторін" or "IV. Якість товару"; OCR gives Cyrillic І and Х as well
SECTION = re.compile(r"(\d{1,2}|[IVXІХ]{1,6})\.\s*[^\d\s.].*")
ROMAN = {"I": 1, "І": 1, "V": 5, "X": 10, "Х": 10}
PARTY = re.compile(r"(покупець|постачальник|продавець|замовник|учасник|виконавець)\s*:?")
REQUISITES_WORDS = re.compile(r"реквізит|адрес|підпис|специфікац|додаток")
# "1. " is a part of a law article, "201.10. " a point of a Tax Code article
LAW_PART = re.compile(r"(\d{1,3}(?:\.\d{1,3})?)\.\s")
DECISION_PARAGRAPH = re.compile(r"(\d{1,3}(?:\.\d{1,3})*)\.\s")
NUMBERED_HEADING = re.compile(r"\d{1,2}\.\s+(.+)")
TAG = re.compile(r"\[[^\]]*\]")  # a masked value such as [ПІБ]


class Line(NamedTuple):
    page: int | None  # a contract page; laws and decisions have none
    text: str


@dataclass(frozen=True)
class Unit:
    lines: tuple[Line, ...]
    section: str | None = None  # "7. Відповідальність сторін", "ПОЗИЦІЯ ВЕРХОВНОГО СУДУ"
    heading: str | None = None  # "Стаття 625. Відповідальність за порушення грошового зобов'язання"
    ref: str | None = None  # "п. 7.3", "ст. 625, ч. 2"; None when the text has no number to cite

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def pages(self) -> tuple[int, ...]:
        return tuple(sorted({line.page for line in self.lines if line.page is not None}))


class _Heading(NamedTuple):
    """The line names the section of the units after it and is not part of their text."""

    section: str


class _Start(NamedTuple):
    """The line opens a new unit."""

    ref: str | None
    section: str | None = None  # a new section the unit opens, as the requisites do


type _Mark = _Heading | _Start | None  # None: the line goes on with the open unit


def contract_units(document: Document, *, max_chars: int) -> tuple[Unit, ...]:
    """Clauses; a clause longer than max_chars is split into its subclauses 10.6.1, 10.6.2…"""
    lines = _lines(document.parts)
    clauses = _group(lines, tuple(_contract_marks(lines)))
    return tuple(unit for clause in clauses for unit in _clause_units(clause, max_chars))


def law_units(document: Document, *, max_chars: int) -> tuple[Unit, ...]:
    """Whole articles; an article longer than max_chars is split into its parts."""
    return tuple(unit for part in document.parts for unit in _article_units(document, part, max_chars))


def decision_units(document: Document) -> tuple[Unit, ...]:
    """One unit per line: a decision comes from HTML, where a line is a paragraph."""
    lines = _lines(document.parts)
    return _group(lines, tuple(_decision_mark(line.text) for line in lines))


def _lines(parts: Sequence[DocumentPart]) -> tuple[Line, ...]:
    return tuple(Line(part.page, text) for part in parts for text in part.text.splitlines() if text)


def _group(
    lines: Sequence[Line], marks: Sequence[_Mark], *, orphan_ref: str | None = None
) -> tuple[Unit, ...]:
    units: list[Unit] = []
    section: str | None = None
    is_open = False  # a heading closes the unit before it
    for line, mark in zip(lines, marks, strict=True):
        if isinstance(mark, _Heading):
            section, is_open = mark.section, False
        elif mark is None and is_open:
            units[-1] = replace(units[-1], lines=(*units[-1].lines, line))
        else:
            ref = mark.ref if mark else orphan_ref
            section = (mark.section if mark else None) or section
            units.append(Unit(lines=(line,), section=section, ref=ref))
            is_open = True
    return tuple(units)


def _regroup(unit: Unit, marks: Sequence[_Mark]) -> tuple[Unit, ...]:
    """A long unit split by its own numbering; the pieces keep its section, heading and ref."""
    pieces = _group(unit.lines, marks, orphan_ref=unit.ref)
    return tuple(replace(piece, section=unit.section, heading=unit.heading) for piece in pieces)


def _contract_marks(lines: Sequence[Line]) -> Iterator[_Mark]:
    next_majors = _next_clause_majors(lines)
    last = (0, 0)  # clause numbers only grow: "1.1 цього Договору" after 3.1 is a wrapped reference
    for index, line in enumerate(lines):
        clause = _clause_number(line.text)
        if clause and clause > last:
            last = clause
            yield _Start(f"{CLAUSE_REF}{clause[0]}.{clause[1]}")
        elif _section_number(line.text) in {next_majors[index]} - {None}:
            yield _Heading(line.text)  # a real section "5." goes on with clause 5.1
        elif last > (0, 0) and _starts_requisites(line.text):
            yield _Start(None, REQUISITES_SECTION)
        elif index == 0:
            yield _Start(PREAMBLE_REF)
        else:
            yield None


def _clause_number(text: str) -> tuple[int, int] | None:
    """None for "5.1. цього Договору.": a clause never starts with a lowercase word."""
    match = CLAUSE.match(text)
    if match is None or text[match.end() :].lstrip(" .")[:1].islower():
        return None
    return int(match[1]), int(match[2])


def _clause_units(clause: Unit, max_chars: int) -> tuple[Unit, ...]:
    if len(clause.text) <= max_chars or not (clause.ref or "").startswith(CLAUSE_REF):
        return (clause,)
    number = (clause.ref or "").removeprefix(CLAUSE_REF)
    return _regroup(clause, tuple(_subclause_mark(line.text, number) for line in clause.lines))


def _subclause_mark(text: str, clause: str) -> _Mark:
    match = SUBCLAUSE.match(text)
    return _Start(f"{CLAUSE_REF}{clause}.{match[2]}") if match and match[1] == clause else None


def _next_clause_majors(lines: Sequence[Line]) -> tuple[int | None, ...]:
    """For each line, the section number of the first clause after it."""
    majors: list[int | None] = []
    upcoming: int | None = None
    for line in reversed(lines):
        majors.append(upcoming)
        clause = _clause_number(line.text)
        upcoming = clause[0] if clause else upcoming
    return tuple(reversed(majors))


def _section_number(text: str) -> int | None:
    match = SECTION.fullmatch(text)
    if not match or len(text) > MAX_HEADING_CHARS:
        return None
    number = match[1]
    return int(number) if number.isdigit() else _roman(number)


def _roman(numeral: str) -> int:
    values = [ROMAN[letter] for letter in numeral]
    return sum(-value if value < after else value for value, after in zip(values, [*values[1:], 0]))


def _starts_requisites(text: str) -> bool:
    """A party label or a requisites heading: «ПОКУПЕЦЬ:», «ПІДПИСИ СТОРІН», «XI. Юридичні адреси»."""
    lowered = text.lower()
    is_heading = _is_capitals(text) or (
        SECTION.fullmatch(text) is not None
        and len(text) <= MAX_HEADING_CHARS
        and not text.endswith(FINAL_PUNCTUATION)
    )
    return is_heading and bool(PARTY.fullmatch(lowered) or REQUISITES_WORDS.search(lowered))


def _is_capitals(text: str) -> bool:
    letters = [char for char in TAG.sub("", text) if char.isalpha()]
    return (
        len(text) <= MAX_HEADING_CHARS
        and len(letters) >= MIN_HEADING_LETTERS
        and all(letter.isupper() for letter in letters)
    )


def _article_units(document: Document, part: DocumentPart, max_chars: int) -> tuple[Unit, ...]:
    if part.article is None:
        raise ValueError(f"{document.doc_id}: a law part without an article number")
    article = Unit(lines=_lines((part,)), heading=part.heading, ref=f"ст. {part.article}")
    if len(article.text) <= max_chars:
        return (article,)
    ref = article.ref or ""
    return _regroup(article, tuple(_law_mark(line.text, part.article, ref) for line in article.lines))


def _law_mark(text: str, article: str, article_ref: str) -> _Mark:
    match = LAW_PART.match(text)
    if not match:
        return None
    label = match[1]
    if "." not in label:
        return _Start(f"{article_ref}, ч. {label}")
    # a point "201.10" belongs to article 201; "14.1.1" lines stay inside their point
    return _Start(f"{article_ref}, п. {label}") if label.split(".")[0] == article else None


def _decision_mark(text: str) -> _Mark:
    if _is_capitals(text) or _is_numbered_heading(text):
        return _Heading(text)
    match = DECISION_PARAGRAPH.match(text)
    return _Start(f"п. {match[1]}" if match else None)


def _is_numbered_heading(text: str) -> bool:
    """«1. Короткий зміст позовних вимог»; a paragraph has digits or ends like a sentence."""
    match = NUMBERED_HEADING.fullmatch(text)
    if not match or len(text) > MAX_HEADING_CHARS:
        return False
    title = match[1]
    return (
        title[0].isupper()
        and not any(char.isdigit() for char in title)
        and not title.endswith(FINAL_PUNCTUATION)
    )
