from collections.abc import Sequence
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from scripts import chunk_rag, index_rag
from src.rag.chunking import Chunk
from src.rag.documents import (
    Document,
    DocumentPart,
    ManifestEntry,
    read_jsonl,
    sha256_hex,
    write_jsonl,
)
from src.rag.embedding import Progress, Vector
from src.rag.index import check_fresh, index_stamp
from src.rag.sources import DocFormat, DocType

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")

HEADER = "doc_id,doc_type,format,title,short_title,raw_path\n"
LAW_ROW = "law_civil_code,law,html,Цивільний кодекс України,ЦК,rag/law_civil_code\n"
LAW = Document(
    doc_id="law_civil_code",
    parts=(
        DocumentPart(
            article="549",
            heading="Стаття 549. Поняття неустойки",
            text="1. Неустойкою (штрафом, пенею) є грошова сума.",
        ),
        DocumentPart(
            article="625",
            heading="Стаття 625. Відповідальність за порушення грошового зобов'язання",
            text="1. Боржник не звільняється від відповідальності.",
        ),
    ),
)


class FakeEmbedder:
    model_name = "fake-embedder"
    dimension = 2

    def __init__(self) -> None:
        self.embedded = 0  # texts embedded so far

    def embed(
        self, texts: Sequence[str], on_progress: Progress | None = None
    ) -> tuple[Vector, ...]:
        self.embedded += len(texts)
        if on_progress:
            on_progress(len(texts), len(texts))
        return tuple((1.0, 0.0) for _ in texts)


class Factory:
    """Hands out one fake embedder and remembers whether anyone asked for it."""

    def __init__(self) -> None:
        self.calls = 0
        self.embedder = FakeEmbedder()

    def __call__(self) -> FakeEmbedder:
        self.calls += 1
        return self.embedder


def manifest_entry(document: Document) -> ManifestEntry:
    return ManifestEntry(
        doc_id=document.doc_id,
        doc_type=DocType.LAW,
        format=DocFormat.HTML,
        raw_sha256="r" * 64,
        text_sha256=sha256_hex(document.text.encode()),
        parts=len(document.parts),
        chars=len(document.text),
    )


def chunked_corpus(tmp_path: Path) -> None:
    """The corpus files and the chunks.jsonl scripts/chunk_rag.py cuts from them."""
    write_jsonl(tmp_path / "documents.jsonl", (LAW,))
    write_jsonl(tmp_path / "manifest.jsonl", (manifest_entry(LAW),))
    (tmp_path / "sources.csv").write_text(HEADER + LAW_ROW, encoding="utf-8")
    assert (
        chunk_rag.run(
            tmp_path / "documents.jsonl",
            tmp_path / "manifest.jsonl",
            tmp_path / "sources.csv",
            tmp_path / "chunks.jsonl",
            min_chars=10,
        )
        == 0
    )


def run_index(
    tmp_path: Path, client: QdrantClient, factory: Factory, *, force: bool = False
) -> int:
    return index_rag.run(
        client,
        factory,
        chunks_path=tmp_path / "chunks.jsonl",
        documents_path=tmp_path / "documents.jsonl",
        manifest_path=tmp_path / "manifest.jsonl",
        sources_path=tmp_path / "sources.csv",
        force=force,
    )


def test_run_builds_the_structural_and_the_fixed_size_index(tmp_path: Path) -> None:
    chunked_corpus(tmp_path)
    client = QdrantClient(":memory:")
    structural = read_jsonl(tmp_path / "chunks.jsonl", Chunk)

    assert run_index(tmp_path, client, Factory()) == 0

    assert len(structural) == 2  # one chunk per article
    check_fresh(client, index_rag.STRUCTURAL_COLLECTION, index_stamp(structural, FakeEmbedder()))
    # the whole law is shorter than one window
    assert client.count(index_rag.FIXED_COLLECTION).count == 1


def test_run_indexes_nothing_when_chunks_are_older_than_the_documents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chunked_corpus(tmp_path)
    edited = LAW.model_copy(update={"parts": LAW.parts[:1]})
    write_jsonl(tmp_path / "documents.jsonl", (edited,))
    write_jsonl(tmp_path / "manifest.jsonl", (manifest_entry(edited),))
    client = QdrantClient(":memory:")
    factory = Factory()

    assert run_index(tmp_path, client, factory) == 1

    assert "law_civil_code: chunks.jsonl was cut from another text" in capsys.readouterr().out
    assert client.get_collections().collections == []
    assert factory.calls == 0  # the 2.3 GB model is not even loaded


def test_main_explains_how_to_start_qdrant_when_it_is_not_running(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert index_rag.main(["--qdrant-url", "http://127.0.0.1:1"]) == 1

    assert "docker compose up -d qdrant" in capsys.readouterr().out


def test_a_second_run_skips_collections_built_from_the_same_chunks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chunked_corpus(tmp_path)
    client = QdrantClient(":memory:")
    assert run_index(tmp_path, client, Factory()) == 0
    capsys.readouterr()
    factory = Factory()

    assert run_index(tmp_path, client, factory) == 0

    assert factory.embedder.embedded == 0
    printed = capsys.readouterr().out
    assert "chunks: up to date, 2 chunks, skipped" in printed
    assert "chunks_fixed: up to date, 1 chunks, skipped" in printed


def test_force_rebuilds_a_fresh_collection(tmp_path: Path) -> None:
    chunked_corpus(tmp_path)
    client = QdrantClient(":memory:")
    assert run_index(tmp_path, client, Factory()) == 0
    factory = Factory()

    assert run_index(tmp_path, client, factory, force=True) == 0

    assert factory.embedder.embedded == 3  # 2 structural chunks and 1 window


def test_run_says_why_it_rebuilds_a_collection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chunked_corpus(tmp_path)

    assert run_index(tmp_path, QdrantClient(":memory:"), Factory()) == 0

    assert "chunks: building: no collection 'chunks' in Qdrant" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("done", "elapsed", "line"),
    [
        (512, 40.0, "chunks: 512/2836 embedded, 12.8 per s, about 3 min left"),
        (2700, 200.0, "chunks: 2700/2836 embedded, 13.5 per s, about 10 s left"),
        (2836, 195.0, "chunks: 2836/2836 embedded, 14.5 per s"),
    ],
)
def test_progress_line_says_how_fast_and_how_long_is_left(
    done: int, elapsed: float, line: str
) -> None:
    assert index_rag.progress_line("chunks", done, 2836, elapsed) == line
