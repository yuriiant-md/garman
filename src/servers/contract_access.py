"""A farmer's contracts, read-only: the core of the MCP server `contracts` (contracts_server.py).

Who asks is fixed when the server starts (farmer_id), never by the model: a tool argument
`farmer_id` could be set to anyone's. The folder is the whitelist and the list of owners at once:

    <root>/<farmer_id>/<contract_id>.pdf

The model names a contract only by its id. The id is checked against ID_PATTERN before a path is
built, so "../", an absolute path or a file name never reach the file system. The file must also
resolve inside the farmer's own folder: a symlink out of it is denied like another farmer's file.

403 for another farmer's contract, 404 for one that is nowhere. The two differ so that the agent
can tell the farmer why it cannot answer (#24); the cost is that a 403 says such a contract exists.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from src.documents.contract import NoTextError, contract_clauses, read_contract
from src.rag.structure import CLAUSE_REF, Unit

ID_PATTERN = r"[a-z0-9][a-z0-9_-]{0,63}"  # a farmer or a contract: no dots, slashes or spaces
# "7.3", "10.6.1": the number only, no "п."; [0-9], not \d, which takes Arabic and other digits
CLAUSE_PATTERN = r"[0-9]{1,3}(?:\.[0-9]{1,3}){1,3}"
PREVIEW_CHARS = 80  # of a clause in the outline: enough to choose, too little to answer from
# Contracts kept read: a farmer has a handful. list_contracts reads every one of them, so a shelf
# of more than CACHE_SIZE is read again in full on each list (deferred: review M2 of #22).
CACHE_SIZE = 32


class InvalidRequest(ValueError):
    """An argument in a form the server does not accept; checked before any file is touched."""


class AccessDenied(PermissionError):
    """403: the contract is not in the farmer's own folder."""


class NotFound(LookupError):
    """404: no such contract anywhere, or no such clause in it."""


class UnreadableContract(ValueError):
    """The PDF is broken, or a scan with no text."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ContractSummary(_Model):
    contract_id: str
    clauses: int
    note: str | None = None  # why a contract has no clauses: a scan, a broken file


class OutlineItem(_Model):
    number: str
    section: str | None
    preview: str


class Clause(_Model):
    contract_id: str
    number: str
    section: str | None
    pages: tuple[int, ...]
    text: str


class _Numbered(NamedTuple):
    number: str  # "7.3", from the ref "п. 7.3"
    unit: Unit


@dataclass(frozen=True)
class ContractShelf:
    root: Path
    farmer_id: str

    def __post_init__(self) -> None:
        if not re.fullmatch(ID_PATTERN, self.farmer_id):
            raise ValueError(f"invalid farmer id {self.farmer_id!r}: expected {ID_PATTERN}")
        if not self.folder.is_dir():
            raise ValueError(f"no folder for farmer {self.farmer_id} in {self.root}")
        # a folder linked to another farmer's would hand out that farmer's contracts
        if self.folder.resolve() != self.root.resolve() / self.farmer_id:
            raise ValueError(f"folder of farmer {self.farmer_id} leads outside its own")

    @property
    def folder(self) -> Path:
        return self.root / self.farmer_id

    def contracts(self) -> tuple[ContractSummary, ...]:
        """The farmer's own contracts; one that cannot be read is listed with a note."""
        return tuple(_summary(path) for path in self._own_files())

    def outline(self, contract_id: str) -> tuple[OutlineItem, ...]:
        """The numbered clauses of one contract, each with its section and a short preview."""
        return tuple(
            OutlineItem(number=item.number, section=item.unit.section, preview=_preview(item.unit))
            for item in _clauses(self._path(contract_id))
        )

    def clause(self, contract_id: str, number: str) -> Clause:
        if not re.fullmatch(CLAUSE_PATTERN, number):
            raise InvalidRequest(f"invalid clause {number!r}: expected a number such as 7.3")
        for item in _clauses(self._path(contract_id)):
            if item.number == number:
                return Clause(
                    contract_id=contract_id,
                    number=number,
                    section=item.unit.section,
                    pages=item.unit.pages,
                    text=item.unit.text,
                )
        raise NotFound(f"404 not found: no clause {number} in contract {contract_id}")

    def require(self, contract_id: str) -> None:
        """Pass the farmer's own contract, without reading it; refuse any other as _path does."""
        self._path(contract_id)

    # --- the whitelist

    def _own_files(self) -> list[Path]:
        return sorted(
            path
            for path in self.folder.glob("*.pdf")
            if re.fullmatch(ID_PATTERN, path.stem) and self._is_own(path)
        )

    def _is_own(self, path: Path) -> bool:
        """A regular file right in the farmer's folder, after any symlink is followed."""
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            return False
        return resolved.is_file() and resolved.parent == self.folder.resolve()

    def _path(self, contract_id: str) -> Path:
        if not re.fullmatch(ID_PATTERN, contract_id):
            raise InvalidRequest(f"invalid contract_id {contract_id!r}: expected {ID_PATTERN}")
        path = self.folder / f"{contract_id}.pdf"
        if self._is_own(path):
            return path
        if path.is_symlink() or self._held_by_another(contract_id):
            raise AccessDenied(f"403 access denied: contract {contract_id} is not yours")
        raise NotFound(f"404 not found: you have no contract {contract_id}")

    def _held_by_another(self, contract_id: str) -> bool:
        return any(
            (folder / f"{contract_id}.pdf").is_file()
            for folder in self.root.iterdir()
            if folder.is_dir() and folder.name != self.farmer_id
        )


def _summary(path: Path) -> ContractSummary:
    try:
        return ContractSummary(contract_id=path.stem, clauses=len(_clauses(path)))
    except UnreadableContract as error:
        return ContractSummary(contract_id=path.stem, clauses=0, note=str(error))


def _preview(unit: Unit) -> str:
    text = " ".join(unit.text.split())
    return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1] + "…"


def _clauses(path: Path) -> tuple[_Numbered, ...]:
    return _read(path, path.stat().st_mtime_ns)


@lru_cache(maxsize=CACHE_SIZE)
def _read(path: Path, mtime_ns: int) -> tuple[_Numbered, ...]:
    """The numbered clauses, read once; mtime_ns is in the key: a changed file is read again."""
    try:
        document = read_contract(path, doc_id=path.stem)
    except NoTextError as error:  # ours: names the id, not the path, and says what to do
        raise UnreadableContract(str(error)) from None
    except (ValueError, RuntimeError):  # pymupdf's own text may name the path on the server
        raise UnreadableContract(f"{path.stem}: the PDF cannot be read") from None
    return tuple(
        _Numbered(unit.ref.removeprefix(CLAUSE_REF), unit)
        for unit in contract_clauses(document)
        if unit.ref is not None and unit.ref.startswith(CLAUSE_REF)
    )


def clear_cache() -> None:
    _read.cache_clear()
