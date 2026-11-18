"""A contract PDF as a Document: for the corpus and for a contract a farmer sends.

The order matters: all pages are masked as one text first (a surname found on page 1 is hidden
on page 5), then cleaned of page numbers, headers and footers; a page left empty is dropped."""

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from src.documents.masking import Found, mask_parts
from src.documents.pdf import clean_pages, pdf_pages
from src.rag.chunking import MAX_CHARS
from src.rag.documents import Document, DocumentPart
from src.rag.structure import Unit, contract_units


class MaskedContract(NamedTuple):
    pages: tuple[str, ...]  # masked but not cleaned: the text stage 1 reviewed in data/texts/
    found: Found  # the personal data put behind tags such as [ПІБ]


def mask_contract(path: Path, *, company_codes: bool) -> MaskedContract:
    """The PDF's pages masked together; lists a reviewer added are read from the PDF's folder."""
    pages, found = mask_parts(pdf_pages(path), path.parent, company_codes=company_codes)
    return MaskedContract(pages, found)


def contract_document(doc_id: str, masked_pages: Sequence[str]) -> Document:
    """Cleaned pages under their PDF page numbers; a page of only a stamp or a footer is dropped."""
    parts = tuple(
        DocumentPart(page=number, text=text)
        for number, text in enumerate(clean_pages(masked_pages), start=1)
        if text
    )
    if not parts:
        raise ValueError(f"{doc_id}: no text on any page; a scanned PDF needs OCR first")
    return Document(doc_id=doc_id, parts=parts)


def read_contract(path: Path, *, doc_id: str | None = None, company_codes: bool = True) -> Document:
    """A new contract, with no Qdrant and no sources row; company codes are hidden unless asked."""
    masked = mask_contract(path, company_codes=company_codes)
    return contract_document(doc_id or path.stem, masked.pages)


def contract_clauses(document: Document, *, max_chars: int = MAX_CHARS) -> tuple[Unit, ...]:
    """The clauses an answer can cite; a clause over max_chars is split into its subclauses."""
    return contract_units(document, max_chars=max_chars)
