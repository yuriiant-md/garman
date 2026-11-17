"""Turn the Prozorro contract PDFs from data/raw/prozorro/ into anonymized texts in data/texts/.

PyMuPDF reads the text layer as it is: stamp and signature noise stays, the texts must be messy.
Phones, IBANs, bank accounts, personal tax numbers (РНОКПП), emails and people's names become
tags. Company names and their 8-digit ЄДРПОУ codes are public and stay. Every masked value goes
to a review file inside data/raw/ (outside git), so a person checks the result before labeling;
a missed surname goes to data/raw/prozorro/extra_names.txt and the script is run again.

`--prefix own_` does the same for the author's own contracts in data/raw/own/: texts own_NNN,
sources data/own_sources.csv with the contract number. These are private contracts, so every
8-digit company code is hidden too, and strings listed in extra_hidden.txt (a name left in a
page header) become [ПРИХОВАНО].

Reading and masking live in src/documents/; this script runs them over a folder of PDFs and
writes the texts, the sources CSV and the review file."""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

from src.documents.masking import (
    EXTRA_ADDRESSES_NAME,
    EXTRA_HIDDEN_NAME,
    EXTRA_NAMES_NAME,
    mask_pii,
    read_reviewer_list,
    suspicious_words,
)
from src.documents.pdf import clean_text, extract_text

ROOT = Path(__file__).resolve().parents[1]
TEXTS_DIR = ROOT / "data" / "texts"
REVIEW_NAME = "mask_review.txt"
FILE_PREFIX = "prozorro_"  # Prozorro PDFs are named prozorro_<tenderID>.pdf
TENDER_URL = "https://prozorro.gov.ua/tender/{tender_id}"
# The title may break across lines in the PDF; a dot that ends the sentence is not part of it
CONTRACT_NO = re.compile(r"договір[^№]{0,60}№\s*([\w/-]+(?:\.[\w/-]+)*)", re.IGNORECASE)


def raw_dir_for(prefix: str) -> Path:
    """prozorro_ -> data/raw/prozorro, own_ -> data/raw/own."""
    return ROOT / "data" / "raw" / prefix.rstrip("_")


def sources_path_for(prefix: str) -> Path:
    return ROOT / "data" / f"{prefix}sources.csv"


def convert_all(
    raw_dir: Path,
    texts_dir: Path,
    sources_path: Path,
    review_path: Path,
    overwrite: bool = False,
    prefix: str = FILE_PREFIX,
) -> list[dict[str, str]]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"no folder with contract PDFs: {raw_dir}")
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDFs in {raw_dir}")
    ids = [f"{prefix}{index:03d}" for index in range(1, len(pdfs) + 1)]
    existing = [
        texts_dir / f"{doc_id}.txt" for doc_id in ids if (texts_dir / f"{doc_id}.txt").exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"{existing[0]} already exists; pass --overwrite to replace the texts"
        )

    extra_names = read_reviewer_list(raw_dir / EXTRA_NAMES_NAME)
    extra_addresses = read_reviewer_list(raw_dir / EXTRA_ADDRESSES_NAME)
    extra_hidden = read_reviewer_list(raw_dir / EXTRA_HIDDEN_NAME)
    private = prefix != FILE_PREFIX  # not a public procurement: company codes are hidden too
    texts_dir.mkdir(parents=True, exist_ok=True)
    rows, review = [], []
    for doc_id, pdf in zip(ids, pdfs, strict=True):
        raw_text, pages = extract_text(pdf)
        text, found = mask_pii(
            clean_text(raw_text), extra_names, extra_addresses, extra_hidden, company_codes=private
        )
        (texts_dir / f"{doc_id}.txt").write_text(text)
        source = _source_columns(pdf, prefix, text)
        counts = Counter(kind for kind, _ in found)
        rows.append(
            {
                "id": doc_id,
                **source,
                "pages": str(pages),
                "chars": str(len(text)),
                "masked": " ".join(f"{kind}:{count}" for kind, count in sorted(counts.items())),
            }
        )
        review.append(f"== {doc_id} ({source_label(source)})")
        review.extend(f"{kind}: {value!r}" for kind, value in found)
        review.extend(f"CHECK: {word!r}" for word in suspicious_words(text))
    _write_sources(sources_path, rows)
    review_path.write_text("\n".join(review) + "\n")
    return rows


def _source_columns(pdf: Path, prefix: str, text: str) -> dict[str, str]:
    """Where a text came from: the tender for Prozorro, the PDF and contract number otherwise."""
    if prefix != FILE_PREFIX:
        number = CONTRACT_NO.search(text)
        return {"source_file": pdf.name, "contract_no": number.group(1) if number else ""}
    tender_id = pdf.stem.removeprefix(FILE_PREFIX)
    return {"tender_id": tender_id, "tender_url": TENDER_URL.format(tender_id=tender_id)}


def source_label(row: dict[str, str]) -> str:
    return row.get("tender_id") or row["source_file"]


def _write_sources(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and anonymize contract texts.")
    parser.add_argument(
        "--prefix", default=FILE_PREFIX, help="text id prefix: prozorro_ (default) or own_"
    )
    parser.add_argument("--raw-dir", type=Path, help="default: data/raw/<prefix without _>")
    parser.add_argument("--texts-dir", type=Path, default=TEXTS_DIR)
    parser.add_argument("--sources", type=Path, help="default: data/<prefix>sources.csv")
    parser.add_argument("--overwrite", action="store_true", help="replace existing texts")
    args = parser.parse_args(argv)
    return argparse.Namespace(**{
        **vars(args),
        "raw_dir": args.raw_dir or raw_dir_for(args.prefix),
        "sources": args.sources or sources_path_for(args.prefix),
    })


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    review_path = args.raw_dir / REVIEW_NAME
    try:
        rows = convert_all(
            args.raw_dir, args.texts_dir, args.sources, review_path, args.overwrite, args.prefix
        )
    except (OSError, pymupdf.FileDataError) as error:
        print(f"Conversion failed: {error}", file=sys.stderr)
        return 1
    for row in rows:
        masked = row["masked"] or "-"
        print(f"{row['id']}  {source_label(row)}  {row['pages']} p.  masked: {masked}")
    print(f"Check every masked value and the texts themselves: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
