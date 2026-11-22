"""Lay out the farmers' contracts for the MCP server `contracts` (stage 3) from data/farmers.csv.

    uv run python -m scripts.setup_farmers

Each row of data/farmers.csv gives a contract of the stage 2 corpus (data/rag_sources.csv) to a
farmer. The PDF is copied, not linked, to data/raw/farmers/<farmer_id>/<contract_id>.pdf: the
server denies a symlink out of the farmer's folder. The reviewer's masking lists next to the PDF
(extra_names.txt and others) are copied too, so the server masks the text as stage 2 did. Every
row is checked before anything is copied. A PDF in a farmer's folder that the list does not name
stops the run with exit code 2, so `make agent` does not start on a stale layout; it is never
deleted, that is the user's to do. Nothing laid out here goes to git: data/raw/ is ignored.
"""

import argparse
import csv
import re
import shutil
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from src.documents.masking import EXTRA_ADDRESSES_NAME, EXTRA_HIDDEN_NAME, EXTRA_NAMES_NAME
from src.rag.sources import DocFormat, DocType, Source, load_sources
from src.servers.contract_access import ID_PATTERN
from src.servers.contracts_server import DEFAULT_CONTRACTS_DIR

ROOT = Path(__file__).resolve().parents[1]
FARMERS_PATH = ROOT / "data" / "farmers.csv"
SOURCES_PATH = ROOT / "data" / "rag_sources.csv"
RAW_ROOT = ROOT / "data" / "raw"
MASKING_LISTS = (EXTRA_NAMES_NAME, EXTRA_ADDRESSES_NAME, EXTRA_HIDDEN_NAME)
PDF_SUFFIX = ".pdf"


class FarmerContract(BaseModel):
    """One row of data/farmers.csv: whose contract it is."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    farmer_id: str
    contract_id: str

    @field_validator("farmer_id", "contract_id")
    @classmethod
    def _an_id_the_server_accepts(cls, value: str) -> str:
        if not re.fullmatch(ID_PATTERN, value):
            raise ValueError(f"{value!r} does not match {ID_PATTERN}")
        return value


def load_farmers(path: Path) -> tuple[FarmerContract, ...]:
    """The rows of the list; a contract given to two farmers is an error."""
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    contracts = tuple(_parse_row(path, line, row) for line, row in enumerate(rows, start=2))
    owners: dict[str, str] = {}
    for contract in contracts:
        owner = owners.setdefault(contract.contract_id, contract.farmer_id)
        if owner != contract.farmer_id:
            raise ValueError(
                f"{path}: {contract.contract_id} is listed for {owner} and {contract.farmer_id}"
            )
    return contracts


def lay_out(
    contracts: Sequence[FarmerContract],
    sources: Iterable[Source],
    raw_root: Path,
    out: Path,
) -> list[Path]:
    """Copy each contract and its folder's masking lists; returns the PDFs copied anew."""
    by_id = {source.doc_id: source for source in sources}
    pdfs = {contract: _pdf(contract.contract_id, by_id, raw_root) for contract in contracts}
    lists = _masking_lists(pdfs)
    copied = []
    for contract, pdf in pdfs.items():
        target = out / contract.farmer_id / f"{contract.contract_id}{PDF_SUFFIX}"
        if _copy_if_changed(pdf, target):
            copied.append(target)
    for (farmer_id, name), source in lists.items():
        _copy_if_changed(source, out / farmer_id / name)
    return copied


def unlisted(contracts: Iterable[FarmerContract], out: Path) -> list[Path]:
    """PDFs in the folders of the listed farmers that the list does not name."""
    listed = {(c.farmer_id, f"{c.contract_id}{PDF_SUFFIX}") for c in contracts}
    folders = sorted({out / farmer_id for farmer_id, _ in listed})
    return [
        path
        for folder in folders
        if folder.is_dir()
        for path in sorted(folder.glob(f"*{PDF_SUFFIX}"))
        if (folder.name, path.name) not in listed
    ]


def _parse_row(path: Path, line: int, row: dict[str | None, str | None]) -> FarmerContract:
    if None in row:  # DictReader puts the cells beyond the header under None
        raise ValueError(f"{path}:{line}: more cells than columns")
    cells = {key: (value or "").strip() for key, value in row.items()}
    try:
        return FarmerContract.model_validate(cells)
    except ValidationError as error:
        raise ValueError(f"{path}:{line}: {error}") from None


def _pdf(contract_id: str, by_id: dict[str, Source], raw_root: Path) -> Path:
    source = by_id.get(contract_id)
    if source is None:
        raise ValueError(f"{contract_id} is not in the corpus list (rag_sources.csv)")
    if source.doc_type is not DocType.CONTRACT or source.format is not DocFormat.PDF:
        kind = f"{source.doc_type} ({source.format})"
        raise ValueError(f"{contract_id} is a {kind}, not a PDF contract")
    pdf = raw_root / source.raw_path
    if not pdf.is_file():
        raise FileNotFoundError(f"{contract_id}: no PDF at {pdf}")
    return pdf


def _masking_lists(pdfs: dict[FarmerContract, Path]) -> dict[tuple[str, str], Path]:
    """(farmer, list name) → the list to copy; two different lists for one farmer are an error."""
    chosen: dict[tuple[str, str], Path] = {}
    for contract, pdf in pdfs.items():
        for name in MASKING_LISTS:
            source = pdf.parent / name
            if not source.is_file():
                continue
            key = (contract.farmer_id, name)
            first = chosen.setdefault(key, source)
            if first.read_bytes() != source.read_bytes():
                raise ValueError(
                    f"{contract.farmer_id}: {name} of {first.parent} and {source.parent} differ; "
                    "merge them into one list or give the contracts to different farmers"
                )
    return chosen


def _copy_if_changed(source: Path, target: Path) -> bool:
    """Copy through a temporary file, so a crash never leaves half a PDF the server would read."""
    if target.is_file() and target.read_bytes() == source.read_bytes():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    return True


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--farmers", type=Path, default=FARMERS_PATH)
    parser.add_argument("--sources", type=Path, default=SOURCES_PATH)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_CONTRACTS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        contracts = load_farmers(args.farmers)
        copied = lay_out(contracts, load_sources(args.sources), args.raw_root, args.out)
    except (OSError, ValueError) as error:
        print(f"cannot lay out the contracts: {error}", file=sys.stderr)
        return 2
    farmers = len({contract.farmer_id for contract in contracts})
    print(f"{len(contracts)} contracts of {farmers} farmers, {len(copied)} copied anew: {args.out}")
    left = unlisted(contracts, args.out)
    for path in left:
        print(
            f"{path.parent.name}/{path.name} is not in {args.farmers.name}, yet the server serves "
            "it: delete it by hand, or give it to the farmer in the list",
            file=sys.stderr,
        )
    # a contract moved to another farmer but left here would answer instead of the 403
    return 2 if left else 0


if __name__ == "__main__":
    sys.exit(main())
