"""Download the HTML part of the stage 2 corpus listed in data/rag_sources.csv into data/raw/rag/.

Laws come from the kodeksy.com.ua mirror (zakon.rada.gov.ua does not open from here): the script
reads the law's table of contents and saves one page per wanted article as rag/<doc_id>/<article>.html.
Supreme Court decisions come from reyestr.court.gov.ua as rag/<doc_id>.html. Contracts are the PDFs
already in data/raw/, so they need no download. A page already on disk is never fetched again; a
page that fails is reported with its URL and the path to save it by hand. At the end the script
checks the whole corpus (raw files and course minimums) and exits with 1 while anything is missing."""

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from src.rag.sources import (
    ARTICLE,
    MIN_DOCUMENTS,
    DocType,
    Source,
    corpus_problems,
    load_sources,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "data" / "rag_sources.csv"
RAW_ROOT = ROOT / "data" / "raw"
HEADERS = {"User-Agent": "garman course research"}
PAUSE_S = 1.0  # both sites are small public services, so one request a second
# The block scripts/ingest_rag.py reads: a 200 page without it is cut off or a captcha
ARTICLE_MARK = ("article", b'<div id="statya">')
DECISION_MARK = ("decision", b'<textarea id="txtdepository">')


def article_links(index_html: str, index_url: str) -> dict[str, str]:
    """Article number → page URL, from the law's table of contents ("statja-549.htm" or "14-1.htm")."""
    slug = Path(urlparse(index_url).path).stem
    pattern = re.compile(rf'href="(/{re.escape(slug)}/(?:statja-)?({ARTICLE})\.htm)"')
    return {
        match.group(2): urljoin(index_url, match.group(1)) for match in pattern.finditer(index_html)
    }


def _save(
    client: httpx.Client, url: str, path: Path, mark: tuple[str, bytes], pause_s: float
) -> str | None:
    if path.exists():
        return None
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as error:
        return f"{url}: {error}; save the page by hand as {path}"
    finally:
        time.sleep(pause_s)
    kind, block = mark
    if block not in response.content:  # never kept, so the next run fetches it again
        return f"{url}: the page has no {kind} text (cut off or a captcha?); save it as {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return None


def _fetch_law(
    client: httpx.Client, source: Source, index_url: str, raw_root: Path, pause_s: float
) -> list[str]:
    folder = raw_root / source.raw_path
    try:
        index = client.get(index_url)
        index.raise_for_status()
    except httpx.HTTPError as error:
        return [f"{index_url}: {error}; save the article pages by hand in {folder}"]
    finally:
        time.sleep(pause_s)
    links = article_links(index.text, index_url)
    wanted = {article: url for article, url in links.items() if source.wants_article(article)}
    saves = (
        _save(client, url, folder / f"{article}.html", ARTICLE_MARK, pause_s)
        for article, url in wanted.items()
    )
    failures = [failure for failure in saves if failure]
    # the mirror sometimes sends a cut-off table of contents, so a short page must not pass silently
    saved = {path.stem for path in folder.glob("*.html")}
    lacking = [a for a in source.named_articles() if a not in links and a not in saved]
    retry = f"run the script again or save the pages by hand in {folder}"
    if not wanted and not saved:
        failures.append(f"{index_url}: the table of contents lists no wanted article; {retry}")
    elif lacking:
        failures.append(f"{index_url}: the table of contents lacks {', '.join(lacking)}; {retry}")
    return failures


def fetch_source(
    client: httpx.Client, source: Source, raw_root: Path, pause_s: float = PAUSE_S
) -> list[str]:
    """Download one document's HTML; returns what failed, with the URL and where to save it by hand."""
    if source.doc_type is DocType.CONTRACT or source.source_url is None:
        return []
    if source.doc_type is DocType.LAW:
        return _fetch_law(client, source, source.source_url, raw_root, pause_s)
    path = raw_root / source.raw_path
    failure = _save(client, source.source_url, path, DECISION_MARK, pause_s)
    return [failure] if failure else []


def run(
    sources_path: Path,
    raw_root: Path,
    client: httpx.Client,
    pause_s: float = PAUSE_S,
    min_documents: int = MIN_DOCUMENTS,
) -> int:
    sources = load_sources(sources_path)
    failures = [
        failure for source in sources for failure in fetch_source(client, source, raw_root, pause_s)
    ]
    kinds = Counter(f"{source.doc_type} {source.format}" for source in sources)
    counts = ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))
    print(f"{len(sources)} documents: {counts}")
    for source in sources:
        folder = raw_root / source.raw_path
        if source.doc_type is DocType.LAW and folder.is_dir():
            print(f"  {source.doc_id}: {len(list(folder.glob('*.html')))} article pages")
    problems = failures + corpus_problems(sources, raw_root, min_documents)
    for problem in problems:
        print(f"MISSING {problem}")
    return 1 if problems else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", type=Path, default=SOURCES_PATH)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--pause", type=float, default=PAUSE_S, help="seconds between requests")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as client:
        return run(args.sources, args.raw_root, client, args.pause)


if __name__ == "__main__":
    sys.exit(main())
