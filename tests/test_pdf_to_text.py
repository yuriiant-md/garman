import csv
from pathlib import Path

import pytest

from scripts import pdf_to_text
from scripts.pdf_to_text import convert_all, main, parse_args


def make_raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    for tender in ("UA-2024-03-06-009978-a", "UA-2016-11-02-000170-c"):
        (raw / f"prozorro_{tender}.pdf").write_bytes(b"%PDF stub")
    return raw


def fake_extract(path: Path) -> tuple[str, int]:
    return f"Договір {path.stem}\nТел. 0966378279  \n", 2


def test_convert_all_writes_numbered_masked_texts_sources_and_review(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_to_text, "extract_text", fake_extract)
    raw, texts = make_raw_dir(tmp_path), tmp_path / "texts"
    sources, review = tmp_path / "sources.csv", raw / "mask_review.txt"

    rows = convert_all(raw, texts, sources, review)

    assert [row["id"] for row in rows] == ["prozorro_001", "prozorro_002"]
    first = (texts / "prozorro_001.txt").read_text()
    assert first == "Договір prozorro_UA-2016-11-02-000170-c\nТел. [ТЕЛЕФОН]\n"
    saved = list(csv.DictReader(sources.open()))
    assert saved[0]["tender_id"] == "UA-2016-11-02-000170-c"
    assert saved[0]["tender_url"] == "https://prozorro.gov.ua/tender/UA-2016-11-02-000170-c"
    assert "contract_no" not in saved[0]
    assert saved[0]["masked"] == "ТЕЛЕФОН:1"
    assert "0966378279" in review.read_text()


def test_convert_all_refuses_to_overwrite_texts_without_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_to_text, "extract_text", fake_extract)
    raw, texts = make_raw_dir(tmp_path), tmp_path / "texts"
    texts.mkdir()
    (texts / "prozorro_001.txt").write_text("розмічений текст")

    with pytest.raises(FileExistsError):
        convert_all(raw, texts, tmp_path / "sources.csv", raw / "review.txt")

    assert (texts / "prozorro_001.txt").read_text() == "розмічений текст"


def test_convert_all_reads_extra_names_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_to_text, "extract_text", lambda path: ("підпис Гнатюка\n", 1))
    raw, texts = make_raw_dir(tmp_path), tmp_path / "texts"
    (raw / "extra_names.txt").write_text("# прізвища, які пропустило маскування\nГнатюк\n")

    convert_all(raw, texts, tmp_path / "sources.csv", raw / "review.txt")

    assert (texts / "prozorro_001.txt").read_text() == "підпис [ПІБ]\n"


def test_convert_all_with_own_prefix_records_the_source_file_instead_of_a_tender(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pdf_to_text, "extract_text", fake_extract)
    raw, texts = tmp_path / "raw", tmp_path / "texts"
    raw.mkdir()
    for name in ("Training_set_02.pdf", "Training_set_01.pdf"):
        (raw / name).write_bytes(b"%PDF stub")
    review = raw / "mask_review.txt"

    rows = convert_all(raw, texts, tmp_path / "own_sources.csv", review, prefix="own_")

    assert [row["id"] for row in rows] == ["own_001", "own_002"]
    assert (texts / "own_001.txt").read_text().startswith("Договір Training_set_01")
    saved = list(csv.DictReader((tmp_path / "own_sources.csv").open()))
    assert saved[0]["source_file"] == "Training_set_01.pdf"
    assert "tender_id" not in saved[0]
    assert "== own_001 (Training_set_01.pdf)" in review.read_text()


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("ДОГОВІР ПОСТАВКИ № K2606-0484\n", "K2606-0484"),
        ("ДОГОВІР ПОСТАВКИ\n№ K2606-0484\n", "K2606-0484"),  # the PDF broke the title
        ("Договір поставки сільськогосподарської продукції № 12/06.\n", "12/06"),
    ],
)
def test_contract_number_survives_line_breaks_and_drops_the_final_dot(text, number):
    assert pdf_to_text.CONTRACT_NO.search(text).group(1) == number


def test_own_contracts_record_the_contract_number_and_hide_codes(tmp_path, monkeypatch):
    text = "ДОГОВІР ПОСТАВКИ № K2606-0484\nм. Київ\nкод ЄДРПОУ 30307207\n"
    monkeypatch.setattr(pdf_to_text, "extract_text", lambda path: (text, 9))
    raw, texts = tmp_path / "raw", tmp_path / "texts"
    raw.mkdir()
    (raw / "Training_set_01.pdf").write_bytes(b"%PDF stub")
    (raw / "extra_hidden.txt").write_text("# назви, які лишились у колонтитулі\nм. Київ\n")

    rows = convert_all(raw, texts, tmp_path / "own_sources.csv", raw / "review.txt", prefix="own_")

    assert rows[0]["contract_no"] == "K2606-0484"
    assert (texts / "own_001.txt").read_text() == (
        "ДОГОВІР ПОСТАВКИ № K2606-0484\n[ПРИХОВАНО]\nкод ЄДРПОУ [ЄДРПОУ]\n"
    )


def test_prefix_picks_the_raw_folder_and_sources_file():
    own = parse_args(["--prefix", "own_"])
    prozorro = parse_args([])

    assert (own.raw_dir, own.sources) == (
        pdf_to_text.ROOT / "data" / "raw" / "own", pdf_to_text.ROOT / "data" / "own_sources.csv"
    )
    assert (prozorro.raw_dir, prozorro.sources) == (
        pdf_to_text.ROOT / "data" / "raw" / "prozorro",
        pdf_to_text.ROOT / "data" / "prozorro_sources.csv",
    )


def test_main_reports_missing_raw_dir_without_traceback(tmp_path, capsys):
    code = main(["--raw-dir", str(tmp_path / "absent"), "--texts-dir", str(tmp_path / "t")])

    assert code == 1
    assert "absent" in capsys.readouterr().err


def test_review_lists_suspicious_words_left_next_to_masked_names(tmp_path, monkeypatch):
    text = "[ПІБ] IOpia Олександровича, що діє\n"
    monkeypatch.setattr(pdf_to_text, "extract_text", lambda path: (text, 1))
    raw = make_raw_dir(tmp_path)
    review = raw / "review.txt"

    convert_all(raw, tmp_path / "texts", tmp_path / "sources.csv", review)

    assert "CHECK: 'IOpia'" in review.read_text()


def test_convert_all_reads_extra_addresses_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_to_text, "extract_text", lambda path: ("вул. Садова 7А\n", 1))
    raw, texts = make_raw_dir(tmp_path), tmp_path / "texts"
    (raw / "extra_addresses.txt").write_text("# адреси ФОП\nвул. Садова 7А\n")

    convert_all(raw, texts, tmp_path / "sources.csv", raw / "review.txt")

    assert (texts / "prozorro_001.txt").read_text() == "[АДРЕСА]\n"
