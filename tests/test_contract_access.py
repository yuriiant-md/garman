import os
from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest

from src.documents import contract
from src.servers import contract_access
from src.servers.contract_access import (
    PREVIEW_CHARS,
    AccessDenied,
    Clause,
    ContractShelf,
    ContractSummary,
    InvalidRequest,
    NotFound,
    OutlineItem,
    UnreadableContract,
)

PAGES = (
    "ДОГОВІР ПОСТАВКИ № 1\nПостачальник ТОВ «Агро», код 12345678\n",
    "1. Предмет\n1.1. Продавець передає пшеницю 3 класу.\n1.2. Покупець приймає товар.\n",
    "2. Ціна\n2.1. Ціна 9 000 грн за тонну.\n2.2. Пеня 0,1% за кожен день прострочення.\n",
)
SHORT = ("ДОГОВІР № 2\n1.1. Ціна договірна.\n1.2. Оплата за 5 днів.\n",)


@pytest.fixture(autouse=True)
def fresh_cache() -> Iterator[None]:
    contract_access.clear_cache()
    yield
    contract_access.clear_cache()


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every PDF reads as PAGES, k_002 as SHORT; returns the names of the files read."""
    read: list[str] = []

    def pages(path: Path) -> tuple[str, ...]:
        read.append(path.name)
        return SHORT if path.resolve().stem == "k_002" else PAGES  # an alias reads as its target

    monkeypatch.setattr(contract, "pdf_pages", pages)
    return read


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """farmer_a owns k_001 and k_002, farmer_b owns k_900."""
    for farmer, contract_ids in {"farmer_a": ("k_001", "k_002"), "farmer_b": ("k_900",)}.items():
        folder = tmp_path / farmer
        folder.mkdir()
        for contract_id in contract_ids:
            (folder / f"{contract_id}.pdf").write_bytes(b"%PDF-1.7")
    return tmp_path


@pytest.fixture
def shelf(root: Path, reads: list[str]) -> ContractShelf:
    return ContractShelf(root=root, farmer_id="farmer_a")


# --- the shelf itself


@pytest.mark.parametrize("farmer_id", ["", "../farmer_b", "Farmer A", "a" * 65])
def test_an_invalid_farmer_id_is_refused_at_start(root: Path, farmer_id: str) -> None:
    with pytest.raises(ValueError, match="farmer id"):
        ContractShelf(root=root, farmer_id=farmer_id)


def test_a_farmer_without_a_folder_is_refused_at_start(root: Path) -> None:
    with pytest.raises(ValueError, match="no folder"):
        ContractShelf(root=root, farmer_id="farmer_c")


def test_a_farmer_folder_that_links_to_another_farmers_is_refused_at_start(root: Path) -> None:
    (root / "farmer_s").symlink_to(root / "farmer_b")

    with pytest.raises(ValueError, match="farmer_s.*outside"):
        ContractShelf(root=root, farmer_id="farmer_s")


def test_a_farmer_folder_that_links_out_of_the_root_is_refused_at_start(
    root: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    (root / "farmer_x").symlink_to(tmp_path_factory.mktemp("elsewhere"))

    with pytest.raises(ValueError, match="farmer_x.*outside"):
        ContractShelf(root=root, farmer_id="farmer_x")


def test_a_root_reached_through_a_symlink_works(
    root: Path, tmp_path_factory: pytest.TempPathFactory, reads: list[str]
) -> None:
    link = tmp_path_factory.mktemp("link") / "contracts"
    link.symlink_to(root)

    shelf = ContractShelf(root=link, farmer_id="farmer_a")

    assert [summary.contract_id for summary in shelf.contracts()] == ["k_001", "k_002"]
    with pytest.raises(AccessDenied, match="403"):
        shelf.outline("k_900")


# --- list_contracts


def test_lists_only_the_farmers_own_contracts_with_their_clause_counts(
    shelf: ContractShelf,
) -> None:
    assert shelf.contracts() == (
        ContractSummary(contract_id="k_001", clauses=4),
        ContractSummary(contract_id="k_002", clauses=2),
    )


def test_a_file_that_is_not_a_pdf_and_a_subfolder_are_not_listed(
    root: Path, shelf: ContractShelf
) -> None:
    (root / "farmer_a" / "notes.txt").write_text("x", encoding="utf-8")
    (root / "farmer_a" / "old.pdf").mkdir()
    (root / "farmer_a" / "Bad Name.pdf").write_bytes(b"%PDF-1.7")

    assert [summary.contract_id for summary in shelf.contracts()] == ["k_001", "k_002"]


def test_a_symlink_out_of_the_farmers_folder_is_not_listed(
    root: Path, shelf: ContractShelf
) -> None:
    (root / "farmer_a" / "k_link.pdf").symlink_to(root / "farmer_b" / "k_900.pdf")

    assert [summary.contract_id for summary in shelf.contracts()] == ["k_001", "k_002"]


def test_a_scan_is_listed_with_a_note_and_does_not_hide_the_other_contracts(
    shelf: ContractShelf, monkeypatch: pytest.MonkeyPatch
) -> None:
    def pages(path: Path) -> tuple[str, ...]:
        return SHORT if path.stem == "k_002" else ("",)  # k_001 is a scan with no text layer

    monkeypatch.setattr(contract, "pdf_pages", pages)

    scan, short = shelf.contracts()

    assert (scan.contract_id, scan.clauses) == ("k_001", 0)
    assert scan.note is not None and "needs OCR" in scan.note
    assert short == ContractSummary(contract_id="k_002", clauses=2, note=None)


def test_a_broken_pdf_is_unreadable_and_the_server_path_is_not_shown(
    root: Path, shelf: ContractShelf, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: Path) -> tuple[str, ...]:
        raise pymupdf.FileDataError(f"Failed to open file '{path}' as type pdf.")  # as pymupdf does

    monkeypatch.setattr(contract, "pdf_pages", broken)

    with pytest.raises(UnreadableContract, match="k_001") as error:
        shelf.outline("k_001")
    assert str(root) not in str(error.value)


def test_any_other_reading_error_reaches_the_model_only_as_a_plain_message(
    shelf: ContractShelf, monkeypatch: pytest.MonkeyPatch
) -> None:
    def encrypted(path: Path) -> tuple[str, ...]:
        raise ValueError("document closed or encrypted")  # what pymupdf says of an encrypted PDF

    monkeypatch.setattr(contract, "pdf_pages", encrypted)

    with pytest.raises(UnreadableContract) as error:
        shelf.outline("k_001")
    assert str(error.value) == "k_001: the PDF cannot be read"


# --- access to one contract


def test_a_symlink_to_a_contract_in_the_farmers_own_folder_is_allowed(
    root: Path, shelf: ContractShelf
) -> None:
    (root / "farmer_a" / "k_alias.pdf").symlink_to(root / "farmer_a" / "k_002.pdf")

    assert shelf.clause("k_alias", "1.2").text == "1.2. Оплата за 5 днів."


def test_another_farmers_contract_is_denied_with_403(shelf: ContractShelf) -> None:
    with pytest.raises(AccessDenied, match="403"):
        shelf.outline("k_900")


def test_a_symlink_to_another_farmers_contract_is_denied_with_403(
    root: Path, shelf: ContractShelf, reads: list[str]
) -> None:
    (root / "farmer_a" / "k_link.pdf").symlink_to(root / "farmer_b" / "k_900.pdf")

    with pytest.raises(AccessDenied, match="403"):
        shelf.clause("k_link", "1.1")
    assert reads == []  # refused before the file is read


def test_an_unknown_contract_is_not_found_with_404(shelf: ContractShelf) -> None:
    with pytest.raises(NotFound, match="404"):
        shelf.outline("k_404")


@pytest.mark.parametrize(
    "contract_id",
    [
        "../farmer_b/k_900",
        "/etc/passwd",
        "k_001.pdf",
        "",
        "K 001",
        "a" * 65,
        "k_001\x00",
        "k_001\n",  # Python's $ would let a trailing newline through; fullmatch does not
    ],
)
def test_a_contract_id_that_could_be_a_path_is_invalid(
    shelf: ContractShelf, reads: list[str], contract_id: str
) -> None:
    with pytest.raises(InvalidRequest, match="contract_id"):
        shelf.outline(contract_id)
    assert reads == []


# --- contract_outline


def test_the_outline_lists_numbered_clauses_with_their_section_and_a_preview(
    shelf: ContractShelf,
) -> None:
    outline = shelf.outline("k_001")

    assert [(item.number, item.section) for item in outline] == [
        ("1.1", "1. Предмет"),
        ("1.2", "1. Предмет"),
        ("2.1", "2. Ціна"),
        ("2.2", "2. Ціна"),
    ]
    assert outline[0] == OutlineItem(
        number="1.1", section="1. Предмет", preview="1.1. Продавець передає пшеницю 3 класу."
    )


def test_a_long_clause_is_cut_in_the_outline(
    shelf: ContractShelf, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_clause = "1.1. " + "Покупець сплачує пеню " * 20
    monkeypatch.setattr(contract, "pdf_pages", lambda path: (f"{long_clause}\n1.2. Інше.\n",))

    preview = shelf.outline("k_001")[0].preview

    assert len(preview) == PREVIEW_CHARS
    assert preview.endswith("…")


# --- get_clause


def test_a_clause_comes_with_its_section_and_pages(shelf: ContractShelf) -> None:
    assert shelf.clause("k_001", "2.2") == Clause(
        contract_id="k_001",
        number="2.2",
        section="2. Ціна",
        pages=(3,),
        text="2.2. Пеня 0,1% за кожен день прострочення.",
    )


def test_the_clause_text_is_masked(
    shelf: ContractShelf, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        contract, "pdf_pages", lambda path: ("1.1. Продавець ТОВ «Агро», код 12345678.\n",)
    )

    assert shelf.clause("k_001", "1.1").text == "1.1. Продавець ТОВ «Агро», код [ЄДРПОУ]."


def test_an_unknown_clause_number_is_not_found_with_404(shelf: ContractShelf) -> None:
    with pytest.raises(NotFound, match="404.*9.9"):
        shelf.clause("k_001", "9.9")


@pytest.mark.parametrize(
    "number", ["7", "п. 7.3", "7.3.", "7..3", "abc", "", "1.1.1.1.1", "١.٣"]  # last: Arabic digits
)
def test_a_clause_number_in_another_form_is_invalid(shelf: ContractShelf, number: str) -> None:
    with pytest.raises(InvalidRequest, match="clause"):
        shelf.clause("k_001", number)


# --- reading once


def test_a_contract_is_read_once_for_several_calls(
    shelf: ContractShelf, reads: list[str]
) -> None:
    shelf.outline("k_001")
    shelf.clause("k_001", "1.1")
    shelf.clause("k_001", "2.2")

    assert reads == ["k_001.pdf"]


def test_a_changed_file_is_read_again(root: Path, shelf: ContractShelf, reads: list[str]) -> None:
    pdf = root / "farmer_a" / "k_001.pdf"
    shelf.outline("k_001")
    stat = pdf.stat()
    os.utime(pdf, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    shelf.outline("k_001")

    assert reads == ["k_001.pdf", "k_001.pdf"]
