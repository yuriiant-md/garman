"""Read the text layer of a PDF and clean it: one text for stage 1, page by page for stage 2.

PyMuPDF reads the text as it is: stamp and signature noise stays, only layout artefacts go —
soft hyphens, trailing spaces, page numbers, and headers or footers repeated on most pages."""

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pymupdf

MIN_PAGES_FOR_HEADERS = 3  # in a shorter document a repeated line is as likely to be content
HEADER_PAGE_SHARE = 0.5  # a header or footer repeats on at least half of the pages
EDGE_LINES = 2  # headers and footers are among the first and last lines of a page
# own_013-018 reached us with names, stamps and contacts replaced by these labels, one per line
REDACTION_LABELS = frozenset({"COMPANY", "EMAIL", "NAME", "PHONE"})

HYPHEN_BREAK = re.compile(r"(?<=[^\W\d_])-\n(?=[^\W\d_])")  # "будь-\nяких": the hyphen is real


def extract_text(path: Path) -> tuple[str, int]:
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc), doc.page_count


def pdf_pages(path: Path) -> tuple[str, ...]:
    with pymupdf.open(path) as doc:
        return tuple(page.get_text() for page in doc)


def clean_text(text: str) -> str:
    text = re.sub(r"\xad\s*\n\s*", "", text).replace("\xad", "")  # soft hyphen splits a word
    lines = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", lines).strip("\n") + "\n"


def normalize_text(text: str) -> str:
    """One form of every character (NFC), single spaces, words joined back across line breaks."""
    lines = (" ".join(line.split()) for line in unicodedata.normalize("NFC", text).splitlines())
    joined = HYPHEN_BREAK.sub("-", "\n".join(lines))
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


def clean_pages(pages: Sequence[str]) -> tuple[str, ...]:
    """Normalised pages without their page numbers, headers and footers; a page may end up empty."""
    page_lines = [normalize_text(page).splitlines() for page in pages]
    headers = _headers(page_lines)
    return tuple(
        _clean_page(number, lines, headers) for number, lines in enumerate(page_lines, start=1)
    )


def _edges(lines: list[str]) -> list[int]:
    filled = [index for index, line in enumerate(lines) if line]
    return filled[:EDGE_LINES] + filled[-EDGE_LINES:]


def _headers(page_lines: list[list[str]]) -> frozenset[str]:
    if len(page_lines) < MIN_PAGES_FOR_HEADERS:
        return frozenset()
    counts = Counter(
        line for lines in page_lines for line in {lines[index] for index in _edges(lines)}
    )
    return frozenset(
        line for line, count in counts.items() if count >= HEADER_PAGE_SHARE * len(page_lines)
    )


def _clean_page(number: int, lines: list[str], headers: frozenset[str]) -> str:
    edges = _edges(lines)
    ends = {edges[0], edges[-1]} if edges else set()  # a page number is the first or last line
    kept = (
        line
        for index, line in enumerate(lines)
        if not (index in edges and line in headers)
        and not (index in ends and line == str(number))
        and line not in REDACTION_LABELS
    )
    return normalize_text("\n".join(kept))
