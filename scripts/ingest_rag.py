"""Turn the stage 2 corpus in data/raw/ into clean, anonymised documents in data/rag/.

Contracts go through the stage 1 masking of scripts/pdf_to_text.py with their pages kept, and the
masked text must match the reviewed data/texts/<doc_id>.txt: only masking a person has checked
reaches git. A law keeps the article block of each kodeksy.com.ua page, a decision the text inside
the reyestr.court.gov.ua page; menus, banners and the "copy may be outdated" note stay behind.
Names in decisions are masked like in contracts, judges included: answers cite the case number.
What was masked in laws and decisions goes to data/raw/rag/mask_review.txt (outside git) for a
person to check. Exact duplicates are dropped, near-duplicates (one template, another contract)
are only flagged in the manifest. Nothing is written while any document has a problem."""

import argparse
import hashlib
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Sequence
from html.parser import HTMLParser
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import pymupdf

from scripts.pdf_to_text import (
    EXTRA_ADDRESSES_NAME,
    EXTRA_HIDDEN_NAME,
    EXTRA_NAMES_NAME,
    Found,
    clean_text,
    mask_pii,
    read_reviewer_list,
    suspicious_words,
)
from src.rag.documents import Document, DocumentPart, ManifestEntry, NearDuplicate, write_jsonl
from src.rag.sources import (
    MIN_DOCUMENTS,
    DocType,
    Source,
    article_key,
    corpus_problems,
    load_sources,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "data" / "rag_sources.csv"
RAW_ROOT = ROOT / "data" / "raw"
TEXTS_DIR = ROOT / "data" / "texts"
OUT_DIR = ROOT / "data" / "rag"
REVIEW_PATH = RAW_ROOT / "rag" / "mask_review.txt"
DOCUMENTS_NAME = "documents.jsonl"
MANIFEST_NAME = "manifest.jsonl"
OUTPUT_NAMES = (DOCUMENTS_NAME, MANIFEST_NAME)
OWN_PREFIX = "own_"  # the author's private contracts: company codes are hidden too

PAGE_BREAK = "⟦page⟧"  # a line no masking rule touches: parts are masked as one text
MIN_PAGES_FOR_HEADERS = 3  # in a shorter document a repeated line is as likely to be content
HEADER_PAGE_SHARE = 0.5  # a header or footer repeats on at least half of the pages
EDGE_LINES = 2  # headers and footers are among the first and last lines of a page
# own_013-018 reached us with names, stamps and contacts replaced by these labels, one per line
REDACTION_LABELS = frozenset({"COMPANY", "EMAIL", "NAME", "PHONE"})
SHINGLE_WORDS = 5
NEAR_DUPLICATE_SIMILARITY = 0.8  # own contracts made from one template score 0.85-0.92

HYPHEN_BREAK = re.compile(r"(?<=[^\W\d_])-\n(?=[^\W\d_])")  # "будь-\nяких": the hyphen is real
ARTICLE_BLOCK = re.compile(r'<div id="statya">(.*?)</div>\s*<!-- statya -->', re.DOTALL)
HEADING = re.compile(r"<h1>(.*?)</h1>", re.DOTALL)
DECISION_BLOCK = re.compile(r'<textarea id="txtdepository">(.*?)</textarea>', re.DOTALL)
BLOCK_TAGS = frozenset({"br", "div", "h1", "h2", "h3", "h4", "li", "ol", "p", "table", "tr", "ul"})
HIDDEN_TAGS = frozenset({"head", "script", "style", "title"})
REVIEW_HEADER = (
    "# Masked values of laws and decisions; contracts are checked against data/texts/ instead.\n"
    "# A missed name goes to data/raw/rag/extra_names.txt, then run the script again.\n"
)


class Ingested(NamedTuple):
    documents: tuple[Document, ...]  # without exact duplicates
    manifest: tuple[ManifestEntry, ...]  # one entry per source, duplicates included
    review: tuple[str, ...]
    problems: tuple[str, ...]


class _Read(NamedTuple):
    document: Document | None
    found: Found
    problem: str | None = None


# --- cleaning ---


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


# --- reading the raw formats ---


def pdf_pages(path: Path) -> tuple[str, ...]:
    with pymupdf.open(path) as doc:
        return tuple(page.get_text() for page in doc)


class _VisibleText(HTMLParser):
    """The text a browser shows: a line break in the source is a space, blocks and <br> are lines."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in HIDDEN_TAGS:
            self._hidden += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in HIDDEN_TAGS:
            self._hidden = max(0, self._hidden - 1)
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._hidden:
            self.parts.append(re.sub(r"\s+", " ", data))


def visible_text(fragment: str) -> str:
    """Text of an HTML fragment, one line per block; empty paragraphs (&nbsp;) are dropped."""
    parser = _VisibleText()
    parser.feed(fragment)
    parser.close()
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return normalize_text("\n".join(line for line in lines if line))


def law_article(page_html: str, article: str) -> DocumentPart:
    """One kodeksy.com.ua article page: the heading "Стаття 549. …" and the text below it."""
    block = ARTICLE_BLOCK.search(page_html)
    heading = HEADING.search(block.group(1)) if block else None
    if block is None or heading is None:
        raise ValueError("no article block (div#statya with a heading) on the page")
    title = visible_text(heading.group(1)).splitlines()[-1]  # "Цивільний кодекс України" above
    if not title.startswith(f"Стаття {article}."):
        raise ValueError(f"the page holds {title!r}, not article {article}")
    body = visible_text(block.group(1)[heading.end() :])
    return DocumentPart(article=article, heading=title, text=body)


def decision_text(page_html: str) -> str:
    """A reyestr.court.gov.ua page keeps the decision as an HTML document inside a textarea."""
    block = DECISION_BLOCK.search(page_html)
    if block is None:
        raise ValueError("no decision text (textarea#txtdepository) on the page")
    return visible_text(block.group(1))


# --- masking ---


def mask_parts(
    texts: Sequence[str], raw_dir: Path, *, company_codes: bool = False
) -> tuple[tuple[str, ...], Found]:
    """Mask all parts as one text, as stage 1 did: a surname found on page 1 is hidden on page 5.
    Lists a reviewer added (extra_names.txt and others) are read from the folder of the raw file."""
    joined = f"\n{PAGE_BREAK}\n".join(texts)
    masked, found = mask_pii(
        clean_text(joined),
        read_reviewer_list(raw_dir / EXTRA_NAMES_NAME),
        read_reviewer_list(raw_dir / EXTRA_ADDRESSES_NAME),
        read_reviewer_list(raw_dir / EXTRA_HIDDEN_NAME),
        company_codes=company_codes,
    )
    parts = tuple(part.strip("\n") for part in masked.split(PAGE_BREAK))
    if len(parts) != len(texts):
        raise ValueError(f"masking merged parts: {len(texts)} in, {len(parts)} out")
    return parts, found


def _reviewed_problem(doc_id: str, masked_pages: Sequence[str], texts_dir: Path) -> str | None:
    path = texts_dir / f"{doc_id}.txt"
    if not path.is_file():
        return f"{doc_id}: no reviewed stage 1 text {path}"
    if path.read_text(encoding="utf-8").split() != "\n".join(masked_pages).split():
        return f"{doc_id}: the masked text differs from the reviewed {path}; check the masking"
    return None


# --- one document ---


def _read(source: Source, raw_root: Path, texts_dir: Path) -> _Read:
    path = raw_root / source.raw_path
    try:
        if source.doc_type is DocType.CONTRACT:
            return _read_contract(source.doc_id, path, texts_dir)
        if source.doc_type is DocType.LAW:
            return _read_law(source.doc_id, path)
        return _read_decision(source.doc_id, path)
    except ValueError as error:  # a broken page or an empty document; pydantic errors included
        return _Read(None, [], f"{source.doc_id}: {error}")


def _read_contract(doc_id: str, path: Path, texts_dir: Path) -> _Read:
    private = doc_id.startswith(OWN_PREFIX)
    masked, found = mask_parts(pdf_pages(path), path.parent, company_codes=private)
    if problem := _reviewed_problem(doc_id, masked, texts_dir):
        return _Read(None, found, problem)
    pages = tuple(
        DocumentPart(page=number, text=text)
        for number, text in enumerate(clean_pages(masked), start=1)
        if text  # a page of only a stamp or a footer
    )
    return _Read(Document(doc_id=doc_id, parts=pages), found)


def _read_law(doc_id: str, folder: Path) -> _Read:
    pages = sorted(folder.glob("*.html"), key=lambda page: article_key(page.stem))
    articles = tuple(_article(page) for page in pages)
    masked, found = mask_parts([article.text for article in articles], folder.parent)
    parts = tuple(
        DocumentPart(article=article.article, heading=article.heading, text=normalize_text(text))
        for article, text in zip(articles, masked, strict=True)
    )
    return _Read(Document(doc_id=doc_id, parts=parts), found)


def _article(page: Path) -> DocumentPart:
    try:
        return law_article(page.read_text(encoding="utf-8"), page.stem)
    except ValueError as error:
        raise ValueError(f"{page}: {error}") from None


def _read_decision(doc_id: str, path: Path) -> _Read:
    (masked,), found = mask_parts([decision_text(path.read_text(encoding="utf-8"))], path.parent)
    return _Read(Document(doc_id=doc_id, parts=(DocumentPart(text=normalize_text(masked)),)), found)


# --- the whole corpus ---


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def raw_sha256(path: Path) -> str:
    """Hash of a raw file, or of every article page in a law folder together with its name."""
    if path.is_file():
        return sha256_hex(path.read_bytes())
    listing = "".join(
        f"{page.name} {sha256_hex(page.read_bytes())}\n" for page in sorted(path.glob("*.html"))
    )
    return sha256_hex(listing.encode())


def _shingles(text: str) -> frozenset[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold())
    return frozenset(
        tuple(words[index : index + SHINGLE_WORDS])
        for index in range(len(words) - SHINGLE_WORDS + 1)
    )


def near_duplicates(documents: Sequence[Document]) -> dict[str, tuple[NearDuplicate, ...]]:
    """Pairs of documents sharing most of their five-word runs, each flagged on both sides."""
    shingles = {document.doc_id: _shingles(document.text) for document in documents}
    pairs = [
        (first, second, len(a & b) / len(a | b))
        for (first, a), (second, b) in combinations(shingles.items(), 2)
        if a and b
    ]
    similar = [
        (doc_id, NearDuplicate(doc_id=other, similarity=round(score, 2)))
        for first, second, score in pairs
        if score >= NEAR_DUPLICATE_SIMILARITY
        for doc_id, other in ((first, second), (second, first))
    ]
    return {
        doc_id: tuple(
            sorted(
                (near for owner, near in similar if owner == doc_id),
                key=lambda near: (-near.similarity, near.doc_id),
            )
        )
        for doc_id in shingles
    }


def ingest(sources: Sequence[Source], raw_root: Path, texts_dir: Path) -> Ingested:
    reads = tuple(_read(source, raw_root, texts_dir) for source in sources)
    problems = tuple(read.problem for read in reads if read.problem)
    if problems:
        return Ingested((), (), (), problems)
    documents = tuple(read.document for read in reads if read.document)
    hashes = tuple(sha256_hex(document.text.encode()) for document in documents)
    originals: dict[str, str] = {}
    for document, text_hash in zip(documents, hashes, strict=True):
        originals.setdefault(text_hash, document.doc_id)  # the first one in the list stays
    kept = tuple(
        document
        for document, text_hash in zip(documents, hashes, strict=True)
        if originals[text_hash] == document.doc_id
    )
    similar = near_duplicates(kept)
    manifest = tuple(
        _manifest_entry(source, document, read.found, raw_root, text_hash, originals, similar)
        for source, document, read, text_hash in zip(sources, documents, reads, hashes, strict=True)
    )
    review = tuple(
        line
        for source, document, read in zip(sources, documents, reads, strict=True)
        if source.doc_type is not DocType.CONTRACT
        for line in _review_lines(source, read.found, document.text)
    )
    return Ingested(kept, manifest, review, ())


def _manifest_entry(
    source: Source,
    document: Document,
    found: Found,
    raw_root: Path,
    text_hash: str,
    originals: dict[str, str],
    similar: dict[str, tuple[NearDuplicate, ...]],
) -> ManifestEntry:
    original = originals[text_hash]
    return ManifestEntry(
        doc_id=source.doc_id,
        doc_type=source.doc_type,
        format=source.format,
        raw_sha256=raw_sha256(raw_root / source.raw_path),
        text_sha256=text_hash,
        parts=len(document.parts),
        chars=len(document.text),
        masked=dict(sorted(Counter(kind for kind, _ in found).items())),
        duplicate_of=None if original == source.doc_id else original,
        near_duplicates=similar.get(source.doc_id, ()),
    )


def _review_lines(source: Source, found: Found, text: str) -> list[str]:
    return [
        f"== {source.doc_id} ({source.short_title})",
        *(f"{kind}: {value!r}" for kind, value in found),
        *(f"CHECK: {word!r}" for word in suspicious_words(text)),
    ]


# --- the command ---


def run(
    sources_path: Path,
    raw_root: Path,
    texts_dir: Path,
    out_dir: Path,
    review_path: Path,
    min_documents: int = MIN_DOCUMENTS,
) -> int:
    sources = load_sources(sources_path)
    if problems := corpus_problems(sources, raw_root, min_documents):
        return _fail(problems)
    result = ingest(sources, raw_root, texts_dir)
    if result.problems:
        return _fail(result.problems)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / DOCUMENTS_NAME, result.documents)
    write_jsonl(out_dir / MANIFEST_NAME, result.manifest)
    review_path.write_text(REVIEW_HEADER + "".join(f"{line}\n" for line in result.review))
    _print_summary(result, out_dir, review_path)
    return 0


def _fail(problems: Sequence[str]) -> int:
    for problem in problems:
        print(f"PROBLEM {problem}")
    print("nothing written: fix the problems above and run again")
    return 1


def _print_summary(result: Ingested, out_dir: Path, review_path: Path) -> None:
    parts = sum(len(document.parts) for document in result.documents)
    masked = sum((Counter(entry.masked) for entry in result.manifest), Counter[str]())
    duplicates = [entry.doc_id for entry in result.manifest if entry.duplicate_of]
    pairs = {
        tuple(sorted((entry.doc_id, near.doc_id)))
        for entry in result.manifest
        for near in entry.near_duplicates
    }
    print(f"{len(result.documents)} documents, {parts} parts -> {out_dir / DOCUMENTS_NAME}")
    print(f"masked: {', '.join(f'{count} {kind}' for kind, count in sorted(masked.items()))}")
    print(f"exact duplicates dropped: {', '.join(duplicates) or 'none'}")
    print(f"near-duplicate pairs flagged: {len(pairs)}")
    print(f"check what was masked in laws and decisions: {review_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", type=Path, default=SOURCES_PATH)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--texts", type=Path, default=TEXTS_DIR, help="reviewed stage 1 texts")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--review", type=Path, default=REVIEW_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args.sources, args.raw_root, args.texts, args.out, args.review)


if __name__ == "__main__":
    sys.exit(main())
