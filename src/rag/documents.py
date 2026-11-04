"""Ingested documents of the stage 2 corpus: clean, anonymised text split the way citations need it.

scripts/ingest_rag.py writes data/rag/documents.jsonl (one Document per line) and manifest.jsonl
(one ManifestEntry per line); chunking reads the documents back. A contract is split into pages,
a law into articles, a decision stays whole: its sections are found when it is chunked. Titles,
dates and in-force flags stay in data/rag_sources.csv and are joined by doc_id, not copied here."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.rag.sources import DocFormat, DocType


class DocumentPart(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)  # a contract page, as a reader numbers it
    article: str | None = None  # a law article: "549", "14-1"
    heading: str | None = None  # "Стаття 549. Поняття неустойки"


class Document(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    parts: tuple[DocumentPart, ...] = Field(min_length=1)

    @property
    def text(self) -> str:
        return "\n\n".join(
            f"{part.heading}\n{part.text}" if part.heading else part.text for part in self.parts
        )


class NearDuplicate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    similarity: float  # Jaccard similarity of word shingles, 0..1


class ManifestEntry(BaseModel):
    """What was ingested from one source. No run timestamps: the file changes only with the data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    doc_type: DocType
    format: DocFormat
    raw_sha256: str  # of the raw file, or of all article pages of a law
    text_sha256: str  # of Document.text; equal hashes mean an exact duplicate
    parts: int
    chars: int
    masked: dict[str, int] = {}  # personal data tags put in, by kind
    duplicate_of: str | None = None  # an exact duplicate is not written to documents.jsonl
    near_duplicates: tuple[NearDuplicate, ...] = ()  # similar documents, kept but flagged


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    lines = (record.model_dump_json(exclude_defaults=True) for record in records)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def read_documents(path: Path) -> tuple[Document, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return tuple(_parse_line(path, number, line) for number, line in enumerate(lines, start=1))


def _parse_line(path: Path, number: int, line: str) -> Document:
    try:
        return Document.model_validate_json(line)
    except ValidationError as error:
        raise ValueError(f"{path}:{number}: {error}") from None
