from pathlib import Path

import pytest

from scripts.setup_farmers import (
    FARMERS_PATH,
    SOURCES_PATH,
    FarmerContract,
    lay_out,
    load_farmers,
    main,
)
from src.documents.masking import EXTRA_ADDRESSES_NAME, EXTRA_HIDDEN_NAME, EXTRA_NAMES_NAME
from src.rag.sources import DocType, load_sources
from src.tools.playbook import PLAYBOOKS_DIR

SOURCES_HEADER = (
    "doc_id,doc_type,format,title,short_title,source_url,raw_path,version_date,in_force,"
    "cites_commercial_code,articles,retrieved_on\n"
)
MASKING_LISTS = (EXTRA_NAMES_NAME, EXTRA_ADDRESSES_NAME, EXTRA_HIDDEN_NAME)


def write_sources(path: Path, *rows: str) -> Path:
    path.write_text(SOURCES_HEADER + "".join(f"{row}\n" for row in rows), encoding="utf-8")
    return path


def contract_row(doc_id: str, raw_path: str) -> str:
    return f"{doc_id},contract,pdf,Договір {doc_id},Договір {doc_id},,{raw_path},2026-06-30,,,,"


def write_farmers(path: Path, *rows: str) -> Path:
    path.write_text("farmer_id,contract_id\n" + "".join(f"{row}\n" for row in rows), "utf-8")
    return path


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    (root / "own").mkdir(parents=True)
    (root / "own" / "a.pdf").write_bytes(b"%PDF-a")
    (root / "own" / "b.pdf").write_bytes(b"%PDF-b")
    return root


@pytest.fixture
def sources(tmp_path: Path) -> Path:
    return write_sources(
        tmp_path / "sources.csv",
        contract_row("own_001", "own/a.pdf"),
        contract_row("own_002", "own/b.pdf"),
        "law_1,law,html,ЦК,ЦК,,laws/civil,,true,,,",
    )


def test_each_contract_is_copied_into_its_farmers_folder_under_its_id(
    tmp_path: Path, raw: Path, sources: Path
) -> None:
    # Arrange
    farmers = write_farmers(tmp_path / "farmers.csv", "farmer_a,own_001", "farmer_b,own_002")
    out = tmp_path / "farmers"

    # Act
    lay_out(load_farmers(farmers), load_sources(sources), raw, out)

    # Assert
    assert (out / "farmer_a" / "own_001.pdf").read_bytes() == b"%PDF-a"
    assert (out / "farmer_b" / "own_002.pdf").read_bytes() == b"%PDF-b"
    assert not (out / "farmer_a" / "own_001.pdf").is_symlink()  # the server denies a symlink


def test_the_reviewers_masking_lists_go_with_the_pdf(
    tmp_path: Path, raw: Path, sources: Path
) -> None:
    for name in MASKING_LISTS:
        (raw / "own" / name).write_text(f"{name} lines\n", encoding="utf-8")
    farmers = write_farmers(tmp_path / "farmers.csv", "farmer_a,own_001")
    out = tmp_path / "farmers"

    lay_out(load_farmers(farmers), load_sources(sources), raw, out)

    for name in MASKING_LISTS:
        assert (out / "farmer_a" / name).read_text(encoding="utf-8") == f"{name} lines\n"


def test_a_second_run_changes_nothing_and_a_changed_pdf_is_copied_again(
    tmp_path: Path, raw: Path, sources: Path
) -> None:
    farmers = load_farmers(write_farmers(tmp_path / "farmers.csv", "farmer_a,own_001"))
    out = tmp_path / "farmers"
    copied = out / "farmer_a" / "own_001.pdf"
    assert lay_out(farmers, load_sources(sources), raw, out) == [copied]

    assert lay_out(farmers, load_sources(sources), raw, out) == []

    (raw / "own" / "a.pdf").write_bytes(b"%PDF-a2")
    assert lay_out(farmers, load_sources(sources), raw, out) == [copied]
    assert copied.read_bytes() == b"%PDF-a2"


@pytest.mark.parametrize(
    ("row", "problem"),
    [
        ("farmer_a,own_404", "own_404 is not in"),
        ("farmer_a,law_1", "law_1 is a law"),
        ("Farmer A,own_001", "farmer_id"),
        ("farmer_a,../own_001", "contract_id"),
    ],
)
def test_a_row_the_server_could_not_serve_stops_the_layout(
    tmp_path: Path, raw: Path, sources: Path, row: str, problem: str
) -> None:
    farmers = write_farmers(tmp_path / "farmers.csv", row)

    with pytest.raises(ValueError, match=problem):
        lay_out(load_farmers(farmers), load_sources(sources), raw, tmp_path / "farmers")

    assert not (tmp_path / "farmers").exists()  # checked before anything is copied


def test_a_contract_has_one_owner(tmp_path: Path) -> None:
    farmers = write_farmers(tmp_path / "farmers.csv", "farmer_a,own_001", "farmer_b,own_001")

    with pytest.raises(ValueError, match="own_001 is listed for farmer_a and farmer_b"):
        load_farmers(farmers)


def test_a_missing_pdf_is_named_before_anything_is_copied(
    tmp_path: Path, raw: Path, sources: Path
) -> None:
    (raw / "own" / "b.pdf").unlink()
    farmers = write_farmers(tmp_path / "farmers.csv", "farmer_a,own_001", "farmer_a,own_002")

    with pytest.raises(FileNotFoundError, match="own_002"):
        lay_out(load_farmers(farmers), load_sources(sources), raw, tmp_path / "farmers")

    assert not (tmp_path / "farmers").exists()


def test_two_source_folders_with_different_lists_for_one_farmer_are_refused(
    tmp_path: Path, raw: Path
) -> None:
    (raw / "other").mkdir()
    (raw / "other" / "c.pdf").write_bytes(b"%PDF-c")
    (raw / "own" / EXTRA_NAMES_NAME).write_text("Іваненко\n", encoding="utf-8")
    (raw / "other" / EXTRA_NAMES_NAME).write_text("Петренко\n", encoding="utf-8")
    sources = write_sources(
        tmp_path / "sources.csv",
        contract_row("own_001", "own/a.pdf"),
        contract_row("own_003", "other/c.pdf"),
    )
    farmers = write_farmers(tmp_path / "farmers.csv", "farmer_a,own_001", "farmer_a,own_003")

    with pytest.raises(ValueError, match=f"{EXTRA_NAMES_NAME}.*differ"):
        lay_out(load_farmers(farmers), load_sources(sources), raw, tmp_path / "farmers")


def test_a_pdf_left_in_a_farmers_folder_but_not_listed_stops_it_and_is_kept(
    tmp_path: Path, raw: Path, sources: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    farmers = write_farmers(tmp_path / "farmers.csv", "farmer_a,own_001")
    out = tmp_path / "farmers"
    (out / "farmer_a").mkdir(parents=True)
    (out / "farmer_a" / "old_009.pdf").write_bytes(b"%PDF-old")
    args = ["--farmers", str(farmers), "--sources", str(sources), "--raw-root", str(raw)]

    code = main([*args, "--out", str(out)])

    # the server would still serve it: a contract moved to another farmer would break the 403 task
    assert code == 2
    assert "farmer_a/old_009.pdf is not in" in capsys.readouterr().err
    assert (out / "farmer_a" / "old_009.pdf").exists()  # deleting a farmer's file is the user's


def test_a_bad_list_is_a_clear_message_and_exit_code_2(
    tmp_path: Path, raw: Path, sources: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    farmers = write_farmers(tmp_path / "farmers.csv", "farmer_a,own_404")
    args = ["--farmers", str(farmers), "--sources", str(sources), "--raw-root", str(raw)]

    code = main([*args, "--out", str(tmp_path / "farmers")])

    assert code == 2
    assert "own_404 is not in" in capsys.readouterr().err


def test_the_repos_list_names_contracts_of_the_corpus_and_farmers_with_a_playbook() -> None:
    farmers = load_farmers(FARMERS_PATH)
    contracts = {s.doc_id for s in load_sources(SOURCES_PATH) if s.doc_type is DocType.CONTRACT}

    assert {row.contract_id for row in farmers} <= contracts
    # the task set asks as farmer_a; farmer_b only owns the contract of the 403 task
    assert FarmerContract(farmer_id="farmer_b", contract_id="own_004") in farmers
    assert (PLAYBOOKS_DIR / "farmer_a.yaml").exists()
