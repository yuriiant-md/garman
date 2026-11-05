"""Split the ingested stage 2 documents into chunks for the index: data/rag/chunks.jsonl.

Each document is joined with its row of data/rag_sources.csv and its manifest entry, cut into
citable units (src/rag/structure.py) and packed to MAX_CHARS (src/rag/chunking.py). The run also
compares these structural chunks with fixed-size windows of the same size over the same text: how
many clauses a chunk boundary cuts, how many chunks run across a section or an article, and how
many chunks a single ref can cite. Nothing is written while any document has a problem."""

import argparse
import statistics
import sys
from collections.abc import Iterable, Sequence
from math import ceil
from pathlib import Path
from typing import NamedTuple

from src.rag.chunking import (
    FIXED_OVERLAP,
    MAX_CHARS,
    MIN_CHARS,
    Chunk,
    document_units,
    fixed_size_spans,
    metadata_problem,
    pack_units,
    packed_chunks,
)
from src.rag.documents import (
    Document,
    ManifestEntry,
    print_problems,
    read_documents,
    read_manifest,
    write_jsonl,
)
from src.rag.sources import Source, load_sources
from src.rag.structure import Unit

ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "data" / "rag"
DOCUMENTS_PATH = RAG_DIR / "documents.jsonl"
MANIFEST_PATH = RAG_DIR / "manifest.jsonl"
SOURCES_PATH = ROOT / "data" / "rag_sources.csv"
CHUNKS_PATH = RAG_DIR / "chunks.jsonl"
P90 = 0.9
TABLE_COLUMNS = (
    "chunking",
    "chunks",
    "median chars",
    "p90 chars",
    "clauses cut",
    "cross a boundary",
    "one ref",
)


class Joined(NamedTuple):
    document: Document
    source: Source
    entry: ManifestEntry


class Packed(NamedTuple):
    """One document cut into units and packed once; both the chunks and the comparison use it."""

    joined: Joined
    units: tuple[Unit, ...]
    packed: tuple[Unit, ...]


class Span(NamedTuple):
    start: int
    end: int


class Stats(NamedTuple):
    chunks: int
    median_chars: float
    p90_chars: int
    cut_units: float  # share of citable units that no single chunk holds whole
    cross_boundary: float  # share of chunks touching two sections or two articles
    single_ref: float  # share of chunks one ref cites: [own_003, п. 7.3]


class Comparison(NamedTuple):
    structural: Stats
    fixed: Stats


class _Measured(NamedTuple):
    """One chunking of one document: a flag per unit and per chunk, summed up by _stats."""

    lengths: tuple[int, ...]
    cut: tuple[bool, ...]
    cross: tuple[bool, ...]
    single: tuple[bool, ...]


# --- joining a document with its metadata ---


def join_corpus(
    documents: Sequence[Document], sources: Sequence[Source], manifest: Sequence[ManifestEntry]
) -> tuple[tuple[Joined, ...], tuple[str, ...]]:
    """Each document with its source row and manifest entry, and what is missing or disagrees."""
    by_source = {source.doc_id: source for source in sources}
    by_entry = {entry.doc_id: entry for entry in manifest}
    results = tuple(_joined(document, by_source, by_entry) for document in documents)
    joined = tuple(result for result in results if isinstance(result, Joined))
    problems = tuple(result for result in results if isinstance(result, str))
    return joined, problems


def _joined(
    document: Document, sources: dict[str, Source], entries: dict[str, ManifestEntry]
) -> Joined | str:
    doc_id = document.doc_id
    source = sources.get(doc_id)
    entry = entries.get(doc_id)
    if source is None:
        return f"{doc_id}: no row in rag_sources.csv"
    if entry is None:
        return f"{doc_id}: no entry in the manifest"
    return metadata_problem(document, source, entry) or Joined(document, source, entry)


def pack_corpus(joined: Iterable[Joined], *, max_chars: int, min_chars: int) -> tuple[Packed, ...]:
    """Each document's citable units and the chunks they pack into, found once per run."""
    return tuple(_pack_one(item, max_chars=max_chars, min_chars=min_chars) for item in joined)


def _pack_one(item: Joined, *, max_chars: int, min_chars: int) -> Packed:
    units = document_units(item.document, item.source.doc_type, max_chars=max_chars)
    return Packed(item, units, pack_units(units, max_chars=max_chars, min_chars=min_chars))


def corpus_chunks(packed: Iterable[Packed]) -> tuple[Chunk, ...]:
    return tuple(chunk for item in packed for chunk in packed_chunks(*item.joined, item.packed))


# --- structural chunks against fixed-size windows ---


def compare(packed: Iterable[Packed], *, fixed_size: int, fixed_overlap: int) -> Comparison:
    """Both chunkings measured on the same text, against the same citable units."""
    structural: list[_Measured] = []
    fixed: list[_Measured] = []
    for (document, _, _), units, chunks in packed:
        text = document.text
        unit_spans = _locate(text, units)
        structural.append(
            _measure(units, unit_spans, _locate(text, chunks), [len(c.text) for c in chunks])
        )
        windows = tuple(
            Span(*window)
            for window in fixed_size_spans(len(text), size=fixed_size, overlap=fixed_overlap)
        )
        fixed.append(_measure(units, unit_spans, windows, [end - start for start, end in windows]))
    return Comparison(_stats(structural), _stats(fixed))


def _locate(text: str, units: Sequence[Unit]) -> tuple[Span, ...]:
    """Where each unit lies in the document text: from its first line to the end of its last."""
    spans: list[Span] = []
    cursor = 0
    for unit in units:
        start = cursor
        for index, line in enumerate(unit.lines):
            position = text.find(line.text, cursor)
            if position < 0:
                raise ValueError(f"a chunk line is not in the document text: {line.text[:60]!r}")
            start = position if index == 0 else start
            cursor = position + len(line.text)
        spans.append(Span(start, cursor))
    return tuple(spans)


def _measure(
    units: Sequence[Unit],
    unit_spans: Sequence[Span],
    chunk_spans: Sequence[Span],
    lengths: Sequence[int],
) -> _Measured:
    cited = [span for unit, span in zip(units, unit_spans, strict=True) if unit.ref]
    touched = [_touched(chunk, units, unit_spans) for chunk in chunk_spans]
    return _Measured(
        lengths=tuple(lengths),
        cut=tuple(not any(_holds(chunk, span) for chunk in chunk_spans) for span in cited),
        cross=tuple(len({(u.section, u.heading) for u in group}) > 1 for group in touched),
        single=tuple(len({u.ref for u in group if u.ref}) == 1 for group in touched),
    )


def _holds(chunk: Span, unit: Span) -> bool:
    return chunk.start <= unit.start and unit.end <= chunk.end


def _touched(chunk: Span, units: Sequence[Unit], unit_spans: Sequence[Span]) -> list[Unit]:
    """The units a chunk has at least one character of."""
    return [
        unit
        for unit, span in zip(units, unit_spans, strict=True)
        if span.start < chunk.end and chunk.start < span.end
    ]


def _stats(measured: Sequence[_Measured]) -> Stats:
    lengths = sorted(length for item in measured for length in item.lengths)
    return Stats(
        chunks=len(lengths),
        median_chars=statistics.median(lengths) if lengths else 0,
        p90_chars=lengths[ceil(P90 * len(lengths)) - 1] if lengths else 0,
        cut_units=_share(flag for item in measured for flag in item.cut),
        cross_boundary=_share(flag for item in measured for flag in item.cross),
        single_ref=_share(flag for item in measured for flag in item.single),
    )


def _share(flags: Iterable[bool]) -> float:
    values = tuple(flags)
    return sum(values) / len(values) if values else 0.0


# --- the command ---


def run(
    documents_path: Path,
    manifest_path: Path,
    sources_path: Path,
    out_path: Path,
    *,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    fixed_overlap: int = FIXED_OVERLAP,
) -> int:
    documents = read_documents(documents_path)
    joined, problems = join_corpus(
        documents, load_sources(sources_path), read_manifest(manifest_path)
    )
    if problems:
        return print_problems(problems)
    packed = pack_corpus(joined, max_chars=max_chars, min_chars=min_chars)
    chunks = corpus_chunks(packed)
    comparison = compare(
        packed,
        fixed_size=max_chars,  # the same size, so only where the boundaries fall differs
        fixed_overlap=fixed_overlap,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, chunks)
    print(f"{len(chunks)} chunks from {len(documents)} documents -> {out_path}")
    _print_comparison(
        comparison,
        structural=f"structural ≤ {max_chars}, merge < {min_chars}",
        fixed=f"fixed {max_chars}, overlap {fixed_overlap}",
    )
    return 0


def _print_comparison(comparison: Comparison, *, structural: str, fixed: str) -> None:
    print(f"| {' | '.join(TABLE_COLUMNS)} |")
    print(f"|{' --- |' * len(TABLE_COLUMNS)}")
    for label, stats in ((structural, comparison.structural), (fixed, comparison.fixed)):
        print(
            f"| {label} | {stats.chunks} | {stats.median_chars:.0f} | {stats.p90_chars} "
            f"| {stats.cut_units:.1%} | {stats.cross_boundary:.1%} | {stats.single_ref:.1%} |"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--documents", type=Path, default=DOCUMENTS_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--sources", type=Path, default=SOURCES_PATH)
    parser.add_argument("--out", type=Path, default=CHUNKS_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args.documents, args.manifest, args.sources, args.out)


if __name__ == "__main__":
    sys.exit(main())
