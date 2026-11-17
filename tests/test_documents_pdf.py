from pathlib import Path

import pymupdf
import pytest

from src.documents.pdf import clean_pages, clean_text, extract_text, normalize_text, pdf_pages

CONTRACT_PAGES = (
    "ДОГОВІР ПОСТАВКИ № 1\nПостачальник ТОВ «Агро», код 12345678\nCOMPANY\n",
    "2\n7.3. Пеня 0,1% за кожен день прострочення.\nCOMPANY\n",
    "3\n12.1. Договір діє до 31.12.2026.\nCOMPANY\n",
)


def test_clean_text_trims_line_ends_and_long_gaps_but_keeps_line_structure():
    raw = "1.1. Товар  \n\n\n\n1.2. Ціна\t\n"

    assert clean_text(raw) == "1.1. Товар\n\n1.2. Ціна\n"


def test_clean_text_joins_word_split_by_soft_hyphen():
    assert (
        clean_text("директора Тараса Анатолійо\xad\nвича, що")
        == "директора Тараса Анатолійовича, що\n"
    )


def test_extract_text_reads_pdf_text_layer(tmp_path):
    pdf = tmp_path / "contract.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "Contract 1.1 price 6.00")
        doc.save(pdf)

    text, pages = extract_text(pdf)

    assert "Contract 1.1 price 6.00" in text
    assert pages == 1





@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("будь-\nяких оплат", "будь-яких оплат"),  # a line break inside a hyphenated word
        ("контрагентів-\nпостачальників", "контрагентів-постачальників"),
        ("Сторони:\n- Покупець", "Сторони:\n- Покупець"),  # a list dash is not a break
        ("ціна\xa0\xa0100   грн\t за  тонну", "ціна 100 грн за тонну"),
        ("пункт 1\n\n\n\nпункт 2", "пункт 1\n\nпункт 2"),
        ("  відступ  \n", "відступ"),
        ("його", "його"),  # a decomposed "й" from a PDF becomes one character (NFC)
    ],
)
def test_normalize_text(raw: str, clean: str) -> None:
    assert normalize_text(raw) == clean


def test_clean_pages_drops_page_numbers_and_lines_repeated_on_most_pages() -> None:
    assert clean_pages(CONTRACT_PAGES) == (
        "ДОГОВІР ПОСТАВКИ № 1\nПостачальник ТОВ «Агро», код 12345678",
        "7.3. Пеня 0,1% за кожен день прострочення.",
        "12.1. Договір діє до 31.12.2026.",
    )


def test_clean_pages_drops_redaction_labels_left_in_the_pdf() -> None:
    # own_013-018 came with names, stamps and contacts replaced by labels, several per page
    pages = ("пункт 9.1\n\nNAME\nNAME\nCOMPANY\nCOMPANY\nCOMPANY\nEMAIL\nPHONE\nIBAN\n[IBAN]",)

    assert clean_pages(pages) == ("пункт 9.1\n\nIBAN\n[IBAN]",)  # "IBAN" is a field label


def test_clean_pages_keeps_numbers_that_are_not_the_page_number() -> None:
    pages = (
        "Специфікація\nКількість\n1\nтонна",  # "1" in a table, not a page number
        "5\nпункт 9.1",  # page 2 cannot be numbered "5"
    )

    assert clean_pages(pages) == pages


def test_clean_pages_keeps_repeated_lines_of_a_short_document() -> None:
    pages = ("Покупець\nпункт 1", "Покупець\nпункт 2")  # two pages: no header to speak of

    assert clean_pages(pages) == pages


def test_pdf_pages_reads_every_page(tmp_path: Path) -> None:
    path = tmp_path / "contract.pdf"
    with pymupdf.open() as doc:
        for text in ("Page one", "Page two"):
            doc.new_page().insert_text((72, 72), text)
        doc.save(path)

    assert [page.strip() for page in pdf_pages(path)] == ["Page one", "Page two"]
