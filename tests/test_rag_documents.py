from pathlib import Path

import pytest
from pydantic import ValidationError

from src.rag.documents import (
    Document,
    DocumentPart,
    ManifestEntry,
    NearDuplicate,
    print_problems,
    read_documents,
    read_manifest,
    sha256_hex,
    write_jsonl,
)
from src.rag.sources import DocFormat, DocType

LAW = Document(
    doc_id="law_x",
    parts=(
        DocumentPart(article="549", heading="Стаття 549. Поняття неустойки", text="1. Неустойкою…"),
        DocumentPart(article="550", heading="Стаття 550. Підстави", text="1. Право на неустойку…"),
    ),
)
CONTRACT = Document(
    doc_id="own_001",
    parts=(DocumentPart(page=1, text="ДОГОВІР № 1"), DocumentPart(page=2, text="7.3. Пеня")),
)


def test_document_text_joins_parts_with_their_headings() -> None:
    assert LAW.text == (
        "Стаття 549. Поняття неустойки\n1. Неустойкою…\n\nСтаття 550. Підстави\n1. Право на неустойку…"
    )
    assert CONTRACT.text == "ДОГОВІР № 1\n\n7.3. Пеня"


def test_documents_survive_a_write_and_read(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"

    write_jsonl(path, (LAW, CONTRACT))

    assert read_documents(path) == (LAW, CONTRACT)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "Поняття неустойки" in lines[0]  # Cyrillic stays readable in git diffs


def test_write_jsonl_gives_the_same_bytes_every_time(tmp_path: Path) -> None:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"

    write_jsonl(first, (LAW, CONTRACT))
    write_jsonl(second, (LAW, CONTRACT))

    assert first.read_bytes() == second.read_bytes()


def test_write_jsonl_writes_a_manifest_entry_with_near_duplicates(tmp_path: Path) -> None:
    entry = ManifestEntry(
        doc_id="own_005",
        doc_type=DocType.CONTRACT,
        format=DocFormat.PDF,
        raw_sha256="a" * 64,
        text_sha256="b" * 64,
        parts=11,
        chars=30_000,
        masked={"ПІБ": 2},
        near_duplicates=(NearDuplicate(doc_id="own_007", similarity=0.92),),
    )
    path = tmp_path / "manifest.jsonl"

    write_jsonl(path, (entry,))

    assert ManifestEntry.model_validate_json(path.read_text(encoding="utf-8")) == entry


def test_read_documents_names_the_broken_line(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text(LAW.model_dump_json() + "\n" + '{"doc_id": "x"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"documents\.jsonl:2"):
        read_documents(path)


def test_read_manifest_reads_entries_back_and_names_the_broken_line(tmp_path: Path) -> None:
    entry = ManifestEntry(
        doc_id="law_x",
        doc_type=DocType.LAW,
        format=DocFormat.HTML,
        raw_sha256="a" * 64,
        text_sha256=sha256_hex(LAW.text.encode()),
        parts=2,
        chars=len(LAW.text),
    )
    path = tmp_path / "manifest.jsonl"
    write_jsonl(path, (entry,))
    assert read_manifest(path) == (entry,)

    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"manifest\.jsonl:2"):
        read_manifest(path)


def test_sha256_hex_is_the_hex_digest() -> None:
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_print_problems_lists_each_problem_and_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert print_problems(["law_x: no row", "court_1: broken page"]) == 1

    printed = capsys.readouterr().out.splitlines()
    assert printed[:2] == ["PROBLEM law_x: no row", "PROBLEM court_1: broken page"]
    assert "nothing written" in printed[2]


@pytest.mark.parametrize(
    "fields",
    [
        {"text": ""},  # an empty part would become an empty chunk
        {"text": "x", "page": 0},  # pages count from 1, as a reader sees them
    ],
)
def test_document_part_rejects_bad_values(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DocumentPart.model_validate(fields)


def test_document_needs_at_least_one_part() -> None:
    with pytest.raises(ValidationError):
        Document(doc_id="court_1", parts=())
