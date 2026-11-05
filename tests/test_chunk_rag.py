from pathlib import Path

import pytest

from scripts import chunk_rag
from scripts.chunk_rag import Joined, Stats, compare, join_corpus, pack_corpus, run
from src.rag import chunking
from src.rag.chunking import Chunk, document_units
from src.rag.documents import Document, DocumentPart, ManifestEntry, sha256_hex, write_jsonl
from src.rag.sources import DocFormat, DocType, Source
from src.rag.structure import Unit

HEADER = "doc_id,doc_type,format,title,short_title,raw_path\n"
CONTRACT_ROW = "own_001,contract,pdf,Договір own_001,Договір own_001,own/own_001.pdf\n"
DUPLICATE_ROW = "own_002,contract,pdf,Договір own_002,Договір own_002,own/own_002.pdf\n"
LAW_ROW = "law_civil_code,law,html,Цивільний кодекс України,ЦК,rag/law_civil_code\n"

# three clauses of 35 characters in two sections; a clause starts with a capital letter
CONTRACT = Document(
    doc_id="own_001",
    parts=(
        DocumentPart(page=1, text=f"1. Предмет\n1.1. А{'а' * 29}\n1.2. Б{'б' * 29}"),
        DocumentPart(page=2, text=f"2. Ціна\n2.1. В{'в' * 29}"),
    ),
)
LAW = Document(
    doc_id="law_civil_code",
    parts=(
        DocumentPart(
            article="625",
            heading="Стаття 625. Відповідальність за порушення грошового зобов'язання",
            text="1. Боржник не звільняється від відповідальності.",
        ),
    ),
)


def entry(doc_id: str, doc_type: DocType, **fields: object) -> ManifestEntry:
    return ManifestEntry.model_validate(
        {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "format": DocFormat.PDF if doc_type is DocType.CONTRACT else DocFormat.HTML,
            "raw_sha256": "r" * 64,
            "text_sha256": "t" * 64,
            "parts": 1,
            "chars": 100,
            **fields,
        }
    )


def text_hash(document: Document) -> str:
    return sha256_hex(document.text.encode())


MANIFEST = (
    entry("own_001", DocType.CONTRACT, text_sha256=text_hash(CONTRACT)),
    # an exact duplicate is listed in the manifest but has no document to chunk
    entry("own_002", DocType.CONTRACT, duplicate_of="own_001"),
    entry("law_civil_code", DocType.LAW, text_sha256=text_hash(LAW)),
)


def corpus(
    tmp_path: Path,
    *,
    rows: str = CONTRACT_ROW + DUPLICATE_ROW + LAW_ROW,
    manifest: tuple[ManifestEntry, ...] = MANIFEST,
) -> None:
    write_jsonl(tmp_path / "documents.jsonl", (CONTRACT, LAW))
    write_jsonl(tmp_path / "manifest.jsonl", manifest)
    (tmp_path / "sources.csv").write_text(HEADER + rows, encoding="utf-8")


def run_corpus(tmp_path: Path) -> int:
    return run(
        tmp_path / "documents.jsonl",
        tmp_path / "manifest.jsonl",
        tmp_path / "sources.csv",
        tmp_path / "out" / "chunks.jsonl",
    )


def read_chunks(path: Path) -> list[Chunk]:
    return [Chunk.model_validate_json(line) for line in path.read_text().splitlines()]


def test_run_writes_the_same_chunks_on_every_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus(tmp_path)
    chunks_path = tmp_path / "out" / "chunks.jsonl"

    assert run_corpus(tmp_path) == 0
    first = chunks_path.read_bytes()
    assert run_corpus(tmp_path) == 0

    assert chunks_path.read_bytes() == first
    assert [(chunk.chunk_id, chunk.ref) for chunk in read_chunks(chunks_path)] == [
        ("own_001#001", "п. 1.1–1.2"),
        ("own_001#002", "п. 2.1"),
        ("law_civil_code#001", "ст. 625"),
    ]
    printed = capsys.readouterr().out
    assert "structural" in printed
    assert "fixed" in printed


def test_run_writes_nothing_when_a_document_has_no_source_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus(tmp_path, rows=CONTRACT_ROW + DUPLICATE_ROW)

    assert run_corpus(tmp_path) == 1

    assert not (tmp_path / "out").exists()
    assert "law_civil_code" in capsys.readouterr().out


def test_run_writes_nothing_when_a_document_has_no_manifest_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus(tmp_path, manifest=MANIFEST[:2])

    assert run_corpus(tmp_path) == 1

    assert not (tmp_path / "out").exists()
    assert "law_civil_code" in capsys.readouterr().out


def test_run_writes_nothing_when_the_manifest_is_from_another_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    older = entry("own_001", DocType.CONTRACT, text_sha256=sha256_hex("старий текст".encode()))
    corpus(tmp_path, manifest=(older, *MANIFEST[1:]))

    assert run_corpus(tmp_path) == 1

    assert not (tmp_path / "out").exists()
    assert "own_001: the text differs from the manifest" in capsys.readouterr().out


def test_run_cuts_each_document_into_units_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus(tmp_path)
    cut: list[str] = []

    def counted(document: Document, doc_type: DocType, *, max_chars: int) -> tuple[Unit, ...]:
        cut.append(document.doc_id)
        return document_units(document, doc_type, max_chars=max_chars)

    monkeypatch.setattr(chunking, "document_units", counted)
    monkeypatch.setattr(chunk_rag, "document_units", counted)

    assert run_corpus(tmp_path) == 0
    assert sorted(cut) == ["law_civil_code", "own_001"]


def test_join_corpus_reports_a_document_whose_types_disagree() -> None:
    law_source = Source(
        doc_id="law_civil_code",
        doc_type=DocType.LAW,
        format=DocFormat.HTML,
        title="ЦК",
        short_title="ЦК",
        raw_path="rag/law_civil_code",
    )

    joined, problems = join_corpus(
        (LAW,), (law_source,), (entry("law_civil_code", DocType.CONTRACT),)
    )

    assert joined == ()
    assert len(problems) == 1
    assert "law_civil_code" in problems[0]


def contract_joined() -> Joined:
    contract_source = Source(
        doc_id="own_001",
        doc_type=DocType.CONTRACT,
        format=DocFormat.PDF,
        title="Договір own_001",
        short_title="Договір own_001",
        raw_path="own/own_001.pdf",
    )
    return Joined(CONTRACT, contract_source, MANIFEST[0])


def test_structural_chunks_keep_clauses_whole_and_fixed_windows_cut_them() -> None:
    packed = pack_corpus((contract_joined(),), max_chars=60, min_chars=10)

    result = compare(packed, fixed_size=60, fixed_overlap=10)

    assert result.structural == pytest.approx(
        Stats(
            chunks=3, median_chars=35, p90_chars=35, cut_units=0, cross_boundary=0, single_ref=1
        )
    )
    # windows [0, 60), [50, 110), [100, 127): the second one cuts п. 1.2 and п. 2.1
    # and runs from section 1 into section 2; only the last one stays inside one clause
    assert result.fixed == pytest.approx(
        Stats(
            chunks=3,
            median_chars=60,
            p90_chars=60,
            cut_units=2 / 3,
            cross_boundary=1 / 3,
            single_ref=1 / 3,
        )
    )
