"""Find grain supply contracts on Prozorro and download them as candidates for the golden set.

The search covers completed tenders for wheat, corn, barley and grain in general. For each one
the script downloads the signed contract PDFs into data/raw/prozorro_candidates/ (outside git)
and writes candidates.csv with the text-layer size and whether the text mentions grain and
penalties. A person then opens the best candidates, keeps 10 in data/raw/prozorro/, and runs
scripts/pdf_to_text.py. Prozorro limits request rate, so the script pauses between calls."""

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "prozorro_candidates"
SEARCH_URL = "https://prozorro.gov.ua/api/search/tenders"
DETAILS_URL = "https://prozorro.gov.ua/api/tenders/{tender_id}/details"  # gives the internal id
TENDER_URL = "https://public-api.prozorro.gov.ua/api/2.5/tenders/{internal_id}"
HEADERS = {"User-Agent": "garman course research", "Accept": "application/json"}

CPVS = ("03211100-4", "03211200-5", "03211400-7", "03210000-6")  # wheat, corn, barley, grain
MIN_AMOUNT_UAH = 50_000  # smaller purchases are mostly a few bags of feed
GRAIN_WORDS = re.compile(r"пшениц|кукурудз|ячмін|зерн|овес|жито", re.IGNORECASE)
PENALTY_WORDS = re.compile(r"пен[яію]|штраф|неустойк", re.IGNORECASE)
CONTRACT_DOC_TYPES = {"contractSigned", "contractAnnexe", None, ""}
MIN_CHARS_PER_PAGE = 500  # below this the PDF is a scan without a text layer
PAUSE_S = 1.0
RATE_LIMIT_PAUSE_S = 30.0
ATTEMPTS = 3


def is_candidate(hit: dict[str, Any]) -> bool:
    amount = hit.get("value", {}).get("amount", 0)
    return amount >= MIN_AMOUNT_UAH and bool(GRAIN_WORDS.search(hit.get("title") or ""))


def contract_pdfs(tender: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        doc
        for contract in tender.get("contracts", [])
        for doc in contract.get("documents", [])
        if (doc.get("title") or "").lower().endswith(".pdf")
        and doc.get("documentType") in CONTRACT_DOC_TYPES
    ]


def text_flags(text: str) -> dict[str, bool]:
    return {"grain": bool(GRAIN_WORDS.search(text)), "penalty": bool(PENALTY_WORDS.search(text))}


def text_stats(path: Path) -> dict[str, Any]:
    with pymupdf.open(path) as doc:
        text = "".join(page.get_text() for page in doc)
        pages = doc.page_count
    chars_per_page = len(text) // max(pages, 1)
    return {
        "pages": pages,
        "chars_per_page": chars_per_page,
        "text_layer": chars_per_page >= MIN_CHARS_PER_PAGE,
        **text_flags(text),
    }


def _request(client: httpx.Client, method: str, url: str, **kwargs: Any) -> httpx.Response:
    for attempt in range(1, ATTEMPTS + 1):
        response = client.request(method, url, **kwargs)
        if response.status_code not in (429, 503) or attempt == ATTEMPTS:
            response.raise_for_status()
            return response
        time.sleep(RATE_LIMIT_PAUSE_S * attempt)
    raise AssertionError("unreachable")


def search(client: httpx.Client, cpv: str, page: int) -> list[dict[str, Any]]:
    form = {"cpv[0]": cpv, "status[0]": "complete", "page": page}  # pages start at 1
    return _request(client, "POST", SEARCH_URL, data=form).json()["data"]


def load_tender(client: httpx.Client, tender_id: str) -> dict[str, Any]:
    internal_id = _request(client, "GET", DETAILS_URL.format(tender_id=tender_id)).json()["id"]
    time.sleep(PAUSE_S)
    return _request(client, "GET", TENDER_URL.format(internal_id=internal_id)).json()["data"]


def fetch_candidates(client: httpx.Client, out_dir: Path, pages: int) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for cpv in CPVS:
        for page in range(1, pages + 1):
            hits = search(client, cpv, page)
            for hit in hits:
                if hit["tenderID"] in seen or not is_candidate(hit):
                    continue
                seen.add(hit["tenderID"])
                rows.extend(_download_contracts(client, hit, out_dir))
            if not hits:
                break
    return rows


def _download_contracts(
    client: httpx.Client, hit: dict[str, Any], out_dir: Path
) -> list[dict[str, Any]]:
    tender = load_tender(client, hit["tenderID"])
    rows = []
    for index, doc in enumerate(contract_pdfs(tender), start=1):
        path = out_dir / f"prozorro_{hit['tenderID']}_{index}.pdf"
        if not path.exists():
            path.write_bytes(_request(client, "GET", doc["url"]).content)
            time.sleep(PAUSE_S)
        rows.append(
            {
                "file": path.name,
                "tender_url": f"https://prozorro.gov.ua/tender/{hit['tenderID']}",
                "amount_uah": hit["value"]["amount"],
                "buyer": tender["procuringEntity"]["name"],
                "document": doc["title"],
                **text_stats(path),
            }
        )
    return rows


def write_candidates(path: Path, rows: list[dict[str, Any]]) -> None:
    best_first = sorted(
        rows, key=lambda row: (not row["text_layer"], not row["penalty"], -row["amount_uah"])
    )
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(best_first)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Prozorro grain contracts as golden candidates."
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--pages", type=int, default=20, help="search pages per CPV code, 20 tenders each"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as client:
            rows = fetch_candidates(client, args.out_dir, args.pages)
    except (httpx.HTTPError, KeyError, ValueError) as error:
        print(f"Prozorro search failed: {error!r}", file=sys.stderr)
        return 1
    if not rows:
        print("No contract PDFs found.", file=sys.stderr)
        return 1
    write_candidates(args.out_dir / "candidates.csv", rows)
    with_text = sum(row["text_layer"] for row in rows)
    print(
        f"{len(rows)} contract PDFs, {with_text} with a text layer: {args.out_dir}/candidates.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
