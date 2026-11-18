from pathlib import Path

import pymupdf
import pytest

from src.documents import contract
from src.documents.contract import contract_clauses, read_contract
from src.rag.documents import Document, DocumentPart

PAGES = (
    "ДОГОВІР ПОСТАВКИ № 1\nПостачальник ТОВ «Агро», код 12345678\nCOMPANY\n",
    "2\n7.3. Пеня 0,1% за кожен день прострочення.\nCOMPANY\n",
    "3\n12.1. Договір діє до 31.12.2026.\nCOMPANY\n",
)


def reads_as(monkeypatch: pytest.MonkeyPatch, pages: tuple[str, ...]) -> None:
    """The PDF reads as these pages: Cyrillic PDFs are awkward to build in a test."""
    monkeypatch.setattr(contract, "pdf_pages", lambda path: pages)


def test_a_new_contract_is_masked_cleaned_and_keeps_its_page_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads_as(monkeypatch, PAGES)

    document = read_contract(tmp_path / "farmer_upload.pdf")

    assert document.doc_id == "farmer_upload"
    assert [part.page for part in document.parts] == [1, 2, 3]
    assert document.parts[1].text == "7.3. Пеня 0,1% за кожен день прострочення."
    assert "код [ЄДРПОУ]" in document.text  # a farmer's contract is private
    assert "COMPANY" not in document.text  # a footer on every page is cleaned away


def test_a_public_contract_may_keep_its_company_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads_as(monkeypatch, PAGES)

    document = read_contract(tmp_path / "a.pdf", doc_id="prozorro_001", company_codes=False)

    assert document.doc_id == "prozorro_001"
    assert "код 12345678" in document.text


def test_a_page_left_empty_by_cleaning_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads_as(monkeypatch, ("ДОГОВІР № 1\nCOMPANY\n", "2\nCOMPANY\n", "3\n1.1. Ціна.\nCOMPANY\n"))

    document = read_contract(tmp_path / "a.pdf")

    assert [part.page for part in document.parts] == [1, 3]


def test_names_a_reviewer_listed_next_to_the_file_are_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads_as(monkeypatch, ("ДОГОВІР № 1\n1.1. Підписав Гнатюк.\n",))
    (tmp_path / "extra_names.txt").write_text("Гнатюк\n", encoding="utf-8")

    document = read_contract(tmp_path / "a.pdf")

    assert document.parts[0].text == "ДОГОВІР № 1\n1.1. Підписав [ПІБ]."


def test_a_pdf_without_text_is_an_error_that_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads_as(monkeypatch, ("", "\n"))  # a scan with no text layer

    with pytest.raises(ValueError, match="scan.*no text"):
        read_contract(tmp_path / "scan.pdf")


def test_a_real_pdf_is_read_page_by_page(tmp_path: Path) -> None:
    pdf = tmp_path / "contract.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "1.1. Price is fixed.")
        doc.new_page().insert_text((72, 72), "1.2. Delivery by truck.")
        doc.save(pdf)

    document = read_contract(pdf)

    assert [(part.page, part.text) for part in document.parts] == [
        (1, "1.1. Price is fixed."),
        (2, "1.2. Delivery by truck."),
    ]


def test_contract_clauses_are_the_citable_clauses_with_their_pages() -> None:
    document = Document(
        doc_id="farmer_upload",
        parts=(
            DocumentPart(page=1, text="ДОГОВІР № 1\n1. Предмет\n1.1. Продавець передає товар."),
            DocumentPart(page=2, text="1.2. Покупець оплачує товар."),
        ),
    )

    clauses = contract_clauses(document)

    assert [(c.ref, c.section, c.pages) for c in clauses] == [
        ("преамбула", None, (1,)),
        ("п. 1.1", "1. Предмет", (1,)),
        ("п. 1.2", "1. Предмет", (2,)),
    ]


def test_a_clause_longer_than_a_chunk_is_split_into_its_subclauses() -> None:
    long_text = "Покупець сплачує пеню за кожен день прострочення. " * 20  # ~1000 chars
    text = f"7.3. Відповідальність:\n7.3.1. {long_text}\n7.3.2. {long_text}"
    document = Document(doc_id="a", parts=(DocumentPart(page=1, text=text),))

    clauses = contract_clauses(document)

    assert [c.ref for c in clauses] == ["п. 7.3", "п. 7.3.1", "п. 7.3.2"]
