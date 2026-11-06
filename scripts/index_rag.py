"""Build the stage 2 search index in Qdrant: structural chunks and fixed-size windows.

The `chunks` collection gets data/rag/chunks.jsonl, the structural chunks scripts/chunk_rag.py
wrote. `chunks_fixed` gets fixed-size windows of the same size over the same documents: the
baseline that search quality is compared with. Both are embedded by bge-m3 on this machine's CPU,
about seven minutes for the corpus; the first run also downloads the 2.3 GB model. A collection
already built from the same chunks by the same model is skipped; --force rebuilds it anyway.
Qdrant must be running first: docker compose up -d qdrant. Nothing is indexed while chunks.jsonl
was cut from other documents than documents.jsonl holds now.

A Mac that falls asleep pauses the run, so keep it awake: caffeinate -i uv run python -m ...
"""

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from scripts.chunk_rag import (
    CHUNKS_PATH,
    DOCUMENTS_PATH,
    MANIFEST_PATH,
    SOURCES_PATH,
    Joined,
    join_corpus,
)
from src.rag.chunking import FIXED_OVERLAP, MAX_CHARS, Chunk, fixed_document_chunks
from src.rag.documents import print_problems, read_documents, read_jsonl, read_manifest
from src.rag.embedding import Embedder, bge_m3
from src.rag.index import build_index, index_stamp, stale_reason
from src.rag.sources import load_sources

QDRANT_URL = "http://127.0.0.1:6333"  # docker-compose.yml; QDRANT_URL in the environment wins
STRUCTURAL_COLLECTION = "chunks"
FIXED_COLLECTION = "chunks_fixed"
RECHUNK = "rechunk first: uv run python -m scripts.chunk_rag"
START_QDRANT = "docker compose up -d qdrant"
PRINT_EVERY = 256  # chunks between two progress lines: one line about every 20 s
SECONDS_IN_MINUTE = 60


def stale_chunks(chunks: Sequence[Chunk], joined: Sequence[Joined]) -> list[str]:
    """Documents whose chunks were cut from another text than theirs now, or are missing."""
    current = {item.document.doc_id: item.entry.text_sha256 for item in joined}
    chunked = {chunk.doc_id: chunk.text_sha256 for chunk in chunks}
    changed = [
        f"{doc_id}: chunks.jsonl was cut from another text; {RECHUNK}"
        for doc_id, text_sha256 in chunked.items()
        if current.get(doc_id) != text_sha256
    ]
    missing = [
        f"{doc_id}: no chunks in chunks.jsonl; {RECHUNK}"
        for doc_id in current
        if doc_id not in chunked
    ]
    return changed + missing


def progress_line(collection: str, done: int, total: int, elapsed: float) -> str:
    rate = done / elapsed
    line = f"{collection}: {done}/{total} embedded, {rate:.1f} per s"
    if done == total:
        return line
    left = (total - done) / rate
    if left < SECONDS_IN_MINUTE:
        return f"{line}, about {left:.0f} s left"
    return f"{line}, about {left / SECONDS_IN_MINUTE:.0f} min left"


@dataclass
class ProgressPrinter:
    """Prints a progress line every PRINT_EVERY chunks and at the end, as the embedder reports."""

    collection: str
    every: int = PRINT_EVERY
    started: float = field(default_factory=time.perf_counter)
    printed: int = 0

    def __call__(self, done: int, total: int) -> None:
        if done - self.printed < self.every and done != total:
            return
        self.printed = done
        elapsed = time.perf_counter() - self.started
        # flushed: a run logged to a file shows how far it got while it runs
        print(progress_line(self.collection, done, total, elapsed), flush=True)


def run(
    client: QdrantClient,
    embedder_factory: Callable[[], Embedder],
    *,
    chunks_path: Path,
    documents_path: Path,
    manifest_path: Path,
    sources_path: Path,
    force: bool = False,
) -> int:
    chunks = read_jsonl(chunks_path, Chunk)
    joined, problems = join_corpus(
        read_documents(documents_path), load_sources(sources_path), read_manifest(manifest_path)
    )
    if not problems:  # chunks are compared with documents once every document has its metadata
        problems = tuple(stale_chunks(chunks, joined))
    if problems:
        return print_problems(problems)
    embedder = embedder_factory()  # loaded only once every input has been checked
    collections = ((STRUCTURAL_COLLECTION, chunks), (FIXED_COLLECTION, fixed_chunks(joined)))
    for collection, corpus in collections:
        index_collection(client, collection, corpus, embedder, force=force)
    return 0


def fixed_chunks(joined: Sequence[Joined]) -> tuple[Chunk, ...]:
    """Fixed-size windows over every document: the baseline for the structural chunks."""
    return tuple(
        chunk
        for item in joined
        for chunk in fixed_document_chunks(*item, size=MAX_CHARS, overlap=FIXED_OVERLAP)
    )


def reachable_client(url: str) -> QdrantClient | None:
    """A client of the Qdrant server at url, or None (and how to start it) when none answers."""
    client = QdrantClient(url=url, check_compatibility=False)
    try:
        client.get_collections()
    except ResponseHandlingException:
        print(f"Qdrant is not reachable at {url}: start it with {START_QDRANT}")
        return None
    return client


def index_collection(
    client: QdrantClient,
    collection: str,
    chunks: Sequence[Chunk],
    embedder: Embedder,
    *,
    force: bool,
) -> None:
    """Build the collection unless it already holds these chunks embedded by this model."""
    reason = stale_reason(client, collection, index_stamp(chunks, embedder))
    if reason is None and not force:
        print(f"{collection}: up to date, {len(chunks)} chunks, skipped")
        return
    print(f"{collection}: building: {reason or 'forced'}", flush=True)
    started = time.perf_counter()
    build_index(client, collection, chunks, embedder, on_progress=ProgressPrinter(collection))
    print(f"{collection}: {len(chunks)} chunks indexed in {time.perf_counter() - started:.0f} s")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", QDRANT_URL))
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--documents", type=Path, default=DOCUMENTS_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--sources", type=Path, default=SOURCES_PATH)
    parser.add_argument(
        "--force", action="store_true", help="rebuild collections that are up to date"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    client = reachable_client(args.qdrant_url)
    if client is None:
        return 1
    return run(
        client,
        bge_m3,
        chunks_path=args.chunks,
        documents_path=args.documents,
        manifest_path=args.manifest,
        sources_path=args.sources,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
