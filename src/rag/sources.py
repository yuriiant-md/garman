"""The stage 2 corpus list: one row of data/rag_sources.csv per document, checked before ingestion.

Contracts are the PDFs already in data/raw/ (Prozorro and the author's own). Laws and Supreme Court
decisions are HTML pages that scripts/fetch_rag_sources.py saves next to them. A law keeps only the
articles named in `articles`, so a whole code does not flood the index with unrelated chunks."""

import csv
import re
from datetime import date
from enum import StrEnum
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

MIN_DOCUMENTS = 30  # course minimum for the stage 2 corpus
MIN_FORMATS = 2
DOC_ID_PATTERN = re.compile(r"[a-z0-9_]+")  # the id names raw files and appears in citations
ARTICLE = r"\d+(?:-\d+)?"  # "14-1" is article 14¹, inserted after article 14
ARTICLES_ITEM = re.compile(rf"({ARTICLE})(?:\.\.({ARTICLE}))?")  # "625" or a range "549..552"


class DocType(StrEnum):
    CONTRACT = "contract"
    LAW = "law"
    COURT_DECISION = "court_decision"


class DocFormat(StrEnum):
    PDF = "pdf"
    HTML = "html"


class Source(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    doc_type: DocType
    format: DocFormat
    title: str
    short_title: str  # how an answer cites the document: [ЦК, ст. 549]
    source_url: str | None = None
    raw_path: str  # relative to data/raw/; for a law, the folder of its article pages
    version_date: date | None = None  # edition of a law or date of a decision, when known
    in_force: bool | None = None
    cites_commercial_code: bool | None = None  # a decision that leans on the repealed ГК
    articles: str | None = None  # "549..552,625,14-1"; empty means the whole law

    @field_validator("doc_id")
    @classmethod
    def _doc_id_is_a_file_name(cls, value: str) -> str:
        if not DOC_ID_PATTERN.fullmatch(value):
            raise ValueError("doc_id may use only lowercase Latin letters, digits and '_'")
        return value

    @field_validator("raw_path")
    @classmethod
    def _raw_path_stays_inside_data_raw(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("raw_path must be relative to data/raw/ without '..'")
        return value

    @field_validator("source_url")
    @classmethod
    def _source_url_is_a_web_address(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("https://", "http://")):
            raise ValueError("source_url must start with https:// or http://")
        return value

    @field_validator("articles")
    @classmethod
    def _articles_are_numbers_or_ranges(cls, value: str | None) -> str | None:
        if value is not None and not all(
            ARTICLES_ITEM.fullmatch(item.strip()) for item in value.split(",")
        ):
            raise ValueError("articles must look like '549..552,625,14-1'")
        return value

    def wants_article(self, article: str) -> bool:
        """Whether the article page belongs to this document; inserted articles follow their range."""
        if self.articles is None:
            return True
        return any(_in_item(article, item.strip()) for item in self.articles.split(","))

    def named_articles(self) -> tuple[str, ...]:
        """The articles `articles` spells out: single ones and both ends of each range."""
        if self.articles is None:
            return ()
        return tuple(
            article
            for item in self.articles.split(",")
            for article in item.strip().split("..")
        )


def _article_key(article: str) -> tuple[int, int]:
    main, _, inserted = article.partition("-")
    return int(main), int(inserted or 0)


def _in_item(article: str, item: str) -> bool:
    match = ARTICLES_ITEM.fullmatch(item)
    if match is None or not re.fullmatch(ARTICLE, article):
        return False
    first, last = match.groups()
    if last is None:
        return article == first
    key = _article_key(article)
    if key < _article_key(first):
        return False
    # "14..15" also takes 15-1, 15-2: a range ends after the inserted articles of its last number
    return key <= _article_key(last) if "-" in last else key[0] <= _article_key(last)[0]


def load_sources(path: Path) -> tuple[Source, ...]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    sources = tuple(_parse_row(path, line, row) for line, row in enumerate(rows, start=2))
    seen: set[str] = set()
    for source in sources:
        if source.doc_id in seen:
            raise ValueError(f"{path}: doc_id {source.doc_id} appears more than once")
        seen.add(source.doc_id)
    return sources


def _parse_row(path: Path, line: int, row: dict[str, str | None]) -> Source:
    cells = {key: (value or "").strip() or None for key, value in row.items()}
    try:
        return Source.model_validate(cells)
    except ValidationError as error:
        raise ValueError(f"{path}:{line}: {error}") from None


def corpus_problems(
    sources: tuple[Source, ...], raw_root: Path, min_documents: int = MIN_DOCUMENTS
) -> list[str]:
    """What keeps the corpus from ingestion: missing raw files and the course minimums."""
    problems = [problem for source in sources if (problem := _raw_problem(source, raw_root))]
    if len(sources) < min_documents:
        problems.append(
            f"the corpus has {len(sources)} documents, the course asks for at least {min_documents}"
        )
    formats = {source.format for source in sources}
    if len(formats) < MIN_FORMATS:
        problems.append(
            f"the corpus has {len(formats)} format(s), the course asks for at least {MIN_FORMATS}"
        )
    return problems


def _raw_problem(source: Source, raw_root: Path) -> str | None:
    path = raw_root / source.raw_path
    if source.doc_type is DocType.LAW:
        if not path.is_dir() or not any(path.iterdir()):
            return f"{source.doc_id}: no article pages in {path}"
        return None
    if not path.is_file():
        return f"{source.doc_id}: raw file {path} is missing"
    return None
