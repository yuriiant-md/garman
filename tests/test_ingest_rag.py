import hashlib
from pathlib import Path

import pymupdf
import pytest

from scripts import ingest_rag
from scripts.ingest_rag import (
    clean_pages,
    decision_text,
    ingest,
    law_article,
    near_duplicates,
    normalize_text,
    pdf_pages,
    run,
    visible_text,
)
from src.rag.documents import Document, DocumentPart, ManifestEntry, read_documents
from src.rag.sources import DocFormat, DocType, Source

HEADER = (
    "doc_id,doc_type,format,title,short_title,source_url,raw_path,"
    "version_date,in_force,cites_commercial_code,articles\n"
)
COURT_ROW = "court_1,court_decision,html,Постанова,ВС № 910/1/18,,rag/court_1.html,2021-12-03,,,\n"
CONTRACT_ROW = "prozorro_001,contract,pdf,Договір,Договір prozorro_001,,prozorro/a.pdf,2016-10-27,,,\n"

# A kodeksy.com.ua article page cut down to the parts that matter: the warning and the
# neighbour links sit outside the article block and must not reach the index.
KODEKSY_PAGE = """<html><head><title>ЦК</title><script>var ad = 1;</script></head><body>
<div class="addnav"><p>Ст. {article} ЦКУ від 16.01.2003 № 435-IV</p>
<div id="law_status"><div>Ця копія може бути застарілою. Актуальну редакцію перевіряйте
в офіційному джерелі.</div></div></div>
<div id="statya">
<!-- <div id="smartrotator_ad_4728"></div> -->
<h1>Цивільний кодекс України<br /> Стаття {article}. Поняття неустойки</h1>
<p>1. Неустойкою є грошова сума, яку боржник передає кредиторові у разі порушення
зобов&#39;язання.<br />
<br />
2. Штрафом є неустойка, що обчислюється у відсотках.</p>
</div>   <!-- statya -->
<div class="breadstat">Стаття <a href="/x/statja-548.htm">548</a> <b>{article}</b></div>
</body></html>"""

# A reyestr.court.gov.ua page: the decision itself is an HTML document inside a textarea,
# the menu, the print button and the "test mode" banner are site chrome.
REYESTR_PAGE = """<html><head><script>$('a').css({"color": "black"});</script></head><body>
<div id="menu"><a id="aSearch">Пошук</a></div>
<div id="divdocument"><table id="info"><tr><td>Дата набрання законної сили:
<b>03.12.2021</b></td></tr></table>
<div id="divprint"><button id="btnPractice" title="База працює у тестовому режимі">Релевантна
практика ВС</button><input type="button" id="btnPrint" value="Версія для друку" /></div>
<textarea id="txtdepository"><html><head><title>Єдиний державний реєстр судових рішень</title>
</head><body>
<p><b><span>ПОСТАНОВА</span></b></p>
<p><span>ІМЕНЕМ&nbsp;УКРАЇНИ</span></p>
<p>&nbsp;</p>
<p><span>03&nbsp;грудня&nbsp;2021&nbsp;року</span></p>
<p>Справа № 910/1/18</p>
<p>Верховний Суд у складі суддів: Баранця О. М. (головуючий), Булгакової І. В.</p>
<p>за участю представників: позивача: Коваленко Р. С.</p>
<p>1. Короткий зміст позовних вимог.</p>
<p>Пеня за пунктом 8.2. договору становить 0,1%.</p>
</body></html></textarea></div></body></html>"""

CONTRACT_PAGES = (
    "ДОГОВІР ПОСТАВКИ № 1\nПостачальник ТОВ «Агро», код 12345678\nCOMPANY\n",
    "2\n7.3. Пеня 0,1% за кожен день прострочення.\nCOMPANY\n",
    "3\n12.1. Договір діє до 31.12.2026.\nCOMPANY\n",
)


def source(doc_id: str, doc_type: DocType, raw_path: str) -> Source:
    doc_format = DocFormat.PDF if doc_type is DocType.CONTRACT else DocFormat.HTML
    return Source(
        doc_id=doc_id,
        doc_type=doc_type,
        format=doc_format,
        title=doc_id,
        short_title=doc_id,
        raw_path=raw_path,
    )


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def kodeksy_page(article: str) -> str:
    return KODEKSY_PAGE.replace("{article}", article)


@pytest.fixture
def contract_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every contract PDF reads as CONTRACT_PAGES: Cyrillic PDFs are awkward to build in a test."""
    monkeypatch.setattr(ingest_rag, "pdf_pages", lambda path: CONTRACT_PAGES)


def reviewed_text(tmp_path: Path, doc_id: str, text: str) -> Path:
    """data/texts/ from stage 1: the masked contract a person has already checked."""
    texts_dir = tmp_path / "texts"
    write(texts_dir / f"{doc_id}.txt", text)
    return texts_dir


# --- cleaning ---


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("будь-\nяких оплат", "будь-яких оплат"),  # a line break inside a hyphenated word
        ("контрагентів-\nпостачальників", "контрагентів-постачальників"),
        ("Сторони:\n- Покупець", "Сторони:\n- Покупець"),  # a list dash is not a break
        ("ціна\xa0\xa0100   грн\t за  тонну", "ціна 100 грн за тонну"),
        ("пункт 1\n\n\n\nпункт 2", "пункт 1\n\nпункт 2"),
        ("  відступ  \n", "відступ"),
        ("його", "його"),  # a decomposed "й" from a PDF becomes one character (NFC)
    ],
)
def test_normalize_text(raw: str, clean: str) -> None:
    assert normalize_text(raw) == clean


def test_clean_pages_drops_page_numbers_and_lines_repeated_on_most_pages() -> None:
    assert clean_pages(CONTRACT_PAGES) == (
        "ДОГОВІР ПОСТАВКИ № 1\nПостачальник ТОВ «Агро», код 12345678",
        "7.3. Пеня 0,1% за кожен день прострочення.",
        "12.1. Договір діє до 31.12.2026.",
    )


def test_clean_pages_drops_redaction_labels_left_in_the_pdf() -> None:
    # own_013-018 came with names, stamps and contacts replaced by labels, several per page
    pages = ("пункт 9.1\n\nNAME\nNAME\nCOMPANY\nCOMPANY\nCOMPANY\nEMAIL\nPHONE\nIBAN\n[IBAN]",)

    assert clean_pages(pages) == ("пункт 9.1\n\nIBAN\n[IBAN]",)  # "IBAN" is a field label


def test_clean_pages_keeps_numbers_that_are_not_the_page_number() -> None:
    pages = (
        "Специфікація\nКількість\n1\nтонна",  # "1" in a table, not a page number
        "5\nпункт 9.1",  # page 2 cannot be numbered "5"
    )

    assert clean_pages(pages) == pages


def test_clean_pages_keeps_repeated_lines_of_a_short_document() -> None:
    pages = ("Покупець\nпункт 1", "Покупець\nпункт 2")  # two pages: no header to speak of

    assert clean_pages(pages) == pages


def test_pdf_pages_reads_every_page(tmp_path: Path) -> None:
    path = tmp_path / "contract.pdf"
    with pymupdf.open() as doc:
        for text in ("Page one", "Page two"):
            doc.new_page().insert_text((72, 72), text)
        doc.save(path)

    assert [page.strip() for page in pdf_pages(path)] == ["Page one", "Page two"]


# --- HTML pages ---


def test_law_article_takes_only_the_article_block() -> None:
    part = law_article(kodeksy_page("549"), "549")

    assert part.article == "549"
    assert part.heading == "Стаття 549. Поняття неустойки"
    assert part.text == (  # a source line break is a space in HTML, <br> and blocks are lines
        "1. Неустойкою є грошова сума, яку боржник передає кредиторові у разі порушення "
        "зобов'язання.\n2. Штрафом є неустойка, що обчислюється у відсотках."
    )


def test_law_article_rejects_the_page_of_another_article() -> None:
    with pytest.raises(ValueError, match="550"):
        law_article(kodeksy_page("549"), "550")


def test_law_article_rejects_a_page_without_an_article() -> None:
    with pytest.raises(ValueError, match="article block"):
        law_article("<html><body>502 Bad Gateway</body></html>", "549")


def test_law_article_rejects_a_heading_without_text() -> None:
    page = '<div id="statya"><h1>&nbsp;<img src="x.png"></h1><p>Текст</p></div> <!-- statya -->'

    with pytest.raises(ValueError, match="no text in the heading"):
        law_article(page, "549")


def test_visible_text_keeps_table_cells_apart() -> None:
    table = "<table><tr><th>Сторона</th><th>Назва</th></tr><tr><td>Позивач</td><td>ТОВ</td></tr>"

    assert visible_text(table) == "Сторона\nНазва\nПозивач\nТОВ"


def test_decision_text_takes_only_the_decision() -> None:
    text = decision_text(REYESTR_PAGE)

    assert text.startswith("ПОСТАНОВА\nІМЕНЕМ УКРАЇНИ\n03 грудня 2021 року\nСправа № 910/1/18")
    assert text.endswith("Пеня за пунктом 8.2. договору становить 0,1%.")
    for chrome in ("тестовому режимі", "Версія для друку", "Пошук", "color", "Єдиний державний"):
        assert chrome not in text


def test_decision_text_rejects_a_page_without_a_decision() -> None:
    with pytest.raises(ValueError, match="decision"):
        decision_text("<html><body>Введіть суму чисел</body></html>")


# --- one document ---


def test_ingest_masks_names_in_a_decision_and_lists_them_for_review(tmp_path: Path) -> None:
    write(tmp_path / "raw" / "rag" / "court_1.html", REYESTR_PAGE)
    sources = (source("court_1", DocType.COURT_DECISION, "rag/court_1.html"),)

    result = ingest(sources, tmp_path / "raw", tmp_path / "texts")

    (document,) = result.documents
    assert "[ПІБ]" in document.text
    for name in ("Баранця", "Булгакової", "Коваленко"):
        assert name not in document.text
    assert "Справа № 910/1/18" in document.text  # the case number is what answers cite
    assert "== court_1 (court_1)" in result.review
    assert "ПІБ: 'Коваленко Р. С.'" in result.review
    assert result.problems == ()


def test_ingest_keeps_law_articles_in_their_order(tmp_path: Path) -> None:
    folder = tmp_path / "raw" / "rag" / "law_x"
    for article in ("549", "14-1", "14"):
        write(folder / f"{article}.html", kodeksy_page(article))
    sources = (source("law_x", DocType.LAW, "rag/law_x"),)

    (document,) = ingest(sources, tmp_path / "raw", tmp_path / "texts").documents

    assert [part.article for part in document.parts] == ["14", "14-1", "549"]
    assert document.parts[0].heading == "Стаття 14. Поняття неустойки"


def test_ingest_reports_a_broken_law_page_by_its_path(tmp_path: Path) -> None:
    folder = tmp_path / "raw" / "rag" / "law_x"
    write(folder / "549.html", kodeksy_page("549"))
    broken = write(folder / "550.html", "<html><body>502 Bad Gateway</body></html>")
    sources = (source("law_x", DocType.LAW, "rag/law_x"),)

    result = ingest(sources, tmp_path / "raw", tmp_path / "texts")

    (problem,) = result.problems
    assert problem.startswith(f"law_x: {broken}: no article block")
    assert result.documents == ()


def test_ingest_takes_only_the_law_articles_the_sources_list(tmp_path: Path) -> None:
    folder = tmp_path / "raw" / "rag" / "law_x"
    for article in ("549", "550", "625"):
        write(folder / f"{article}.html", kodeksy_page(article))
    law = source("law_x", DocType.LAW, "rag/law_x").model_copy(update={"articles": "549..550"})

    first = ingest((law,), tmp_path / "raw", tmp_path / "texts")
    write(folder / "625.html", "<html><body>сторінка, якої немає в списку</body></html>")
    second = ingest((law,), tmp_path / "raw", tmp_path / "texts")

    assert [part.article for part in first.documents[0].parts] == ["549", "550"]
    assert second.problems == ()  # a page left outside the list is not even read
    assert second.manifest[0].raw_sha256 == first.manifest[0].raw_sha256


def test_ingest_keeps_contract_pages_and_their_numbers(tmp_path: Path, contract_pdf: None) -> None:
    write(tmp_path / "raw" / "prozorro" / "a.pdf", "")
    texts_dir = reviewed_text(tmp_path, "prozorro_001", "".join(CONTRACT_PAGES))
    sources = (source("prozorro_001", DocType.CONTRACT, "prozorro/a.pdf"),)

    result = ingest(sources, tmp_path / "raw", texts_dir)

    (document,) = result.documents
    assert [part.page for part in document.parts] == [1, 2, 3]
    assert document.parts[1].text == "7.3. Пеня 0,1% за кожен день прострочення."
    assert "код 12345678" in document.text  # a public procurement keeps the company code
    assert result.problems == ()
    assert result.review == ()  # contracts were reviewed at stage 1, data/texts/ proves it


def test_ingest_hides_company_codes_of_own_contracts(tmp_path: Path, contract_pdf: None) -> None:
    write(tmp_path / "raw" / "own" / "a.pdf", "")
    masked = "".join(CONTRACT_PAGES).replace("12345678", "[ЄДРПОУ]")
    texts_dir = reviewed_text(tmp_path, "own_001", masked)
    sources = (source("own_001", DocType.CONTRACT, "own/a.pdf"),)

    result = ingest(sources, tmp_path / "raw", texts_dir)

    (document,) = result.documents
    assert "код [ЄДРПОУ]" in document.text
    assert result.manifest[0].masked == {"ЄДРПОУ": 1}


def test_ingest_stops_when_a_contract_differs_from_its_reviewed_text(
    tmp_path: Path, contract_pdf: None
) -> None:
    write(tmp_path / "raw" / "prozorro" / "a.pdf", "")
    texts_dir = reviewed_text(tmp_path, "prozorro_001", "інший текст")
    sources = (source("prozorro_001", DocType.CONTRACT, "prozorro/a.pdf"),)

    result = ingest(sources, tmp_path / "raw", texts_dir)

    (problem,) = result.problems
    assert "prozorro_001" in problem
    assert str(texts_dir / "prozorro_001.txt") in problem


def test_ingest_stops_when_a_contract_has_no_reviewed_text(
    tmp_path: Path, contract_pdf: None
) -> None:
    write(tmp_path / "raw" / "prozorro" / "a.pdf", "")
    sources = (source("prozorro_001", DocType.CONTRACT, "prozorro/a.pdf"),)

    result = ingest(sources, tmp_path / "raw", tmp_path / "texts")

    (problem,) = result.problems
    assert "prozorro_001" in problem


def test_ingest_skips_a_page_left_empty_by_cleaning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = ("ДОГОВІР № 1\nCOMPANY\n", "2\nCOMPANY\n", "3\nпункт 3\nCOMPANY\n")
    monkeypatch.setattr(ingest_rag, "pdf_pages", lambda path: pages)
    write(tmp_path / "raw" / "prozorro" / "a.pdf", "")
    texts_dir = reviewed_text(tmp_path, "prozorro_001", "".join(pages))
    sources = (source("prozorro_001", DocType.CONTRACT, "prozorro/a.pdf"),)

    (document,) = ingest(sources, tmp_path / "raw", texts_dir).documents

    assert [part.page for part in document.parts] == [1, 3]


# --- the whole corpus ---


def test_ingest_drops_an_exact_duplicate_and_names_the_original(tmp_path: Path) -> None:
    write(tmp_path / "raw" / "rag" / "court_1.html", REYESTR_PAGE)
    write(tmp_path / "raw" / "rag" / "court_2.html", REYESTR_PAGE)
    sources = (
        source("court_1", DocType.COURT_DECISION, "rag/court_1.html"),
        source("court_2", DocType.COURT_DECISION, "rag/court_2.html"),
    )

    result = ingest(sources, tmp_path / "raw", tmp_path / "texts")

    assert [document.doc_id for document in result.documents] == ["court_1"]
    first, second = result.manifest
    assert (first.duplicate_of, second.duplicate_of) == (None, "court_1")
    assert first.text_sha256 == second.text_sha256


def test_near_duplicates_are_flagged_both_ways_and_kept() -> None:
    words = [f"слово{index}" for index in range(40)]
    original = " ".join(words)
    edited = " ".join([*words[:-1], "інше"])  # one word changed, like a second contract
    documents = (
        Document(doc_id="own_005", parts=(DocumentPart(page=1, text=original),)),
        Document(doc_id="own_007", parts=(DocumentPart(page=1, text=edited),)),
        Document(doc_id="law_x", parts=(DocumentPart(article="549", text="зовсім інший текст"),)),
    )

    similar = near_duplicates(documents)

    assert [(n.doc_id, n.similarity) for n in similar["own_005"]] == [("own_007", 0.95)]
    assert [(n.doc_id, n.similarity) for n in similar["own_007"]] == [("own_005", 0.95)]
    assert similar["law_x"] == ()


def test_manifest_describes_each_document(tmp_path: Path) -> None:
    raw_file = write(tmp_path / "raw" / "rag" / "court_1.html", REYESTR_PAGE)
    folder = tmp_path / "raw" / "rag" / "law_x"
    write(folder / "549.html", kodeksy_page("549"))
    sources = (
        source("court_1", DocType.COURT_DECISION, "rag/court_1.html"),
        source("law_x", DocType.LAW, "rag/law_x"),
    )

    result = ingest(sources, tmp_path / "raw", tmp_path / "texts")

    court, law = result.manifest
    document = result.documents[0]
    assert court.raw_sha256 == hashlib.sha256(raw_file.read_bytes()).hexdigest()
    assert court.text_sha256 == hashlib.sha256(document.text.encode()).hexdigest()
    assert (court.doc_type, court.format, court.parts) == (DocType.COURT_DECISION, "html", 1)
    assert court.chars == len(document.text)
    assert court.masked["ПІБ"] >= 3
    assert (law.parts, law.masked) == (1, {})

    write(folder / "549.html", kodeksy_page("549").replace("Штрафом", "Пенею"))
    changed = ingest(sources, tmp_path / "raw", tmp_path / "texts").manifest[1]
    assert changed.raw_sha256 != law.raw_sha256  # a law's hash covers all its article pages


# --- the command ---


def corpus(tmp_path: Path) -> Path:
    """A two-document corpus: one PDF contract and one HTML decision."""
    write(tmp_path / "raw" / "prozorro" / "a.pdf", "")
    write(tmp_path / "raw" / "rag" / "court_1.html", REYESTR_PAGE)
    reviewed_text(tmp_path, "prozorro_001", "".join(CONTRACT_PAGES))
    return write(tmp_path / "sources.csv", HEADER + CONTRACT_ROW + COURT_ROW)


def run_corpus(tmp_path: Path, sources_path: Path) -> int:
    return run(
        sources_path,
        tmp_path / "raw",
        tmp_path / "texts",
        tmp_path / "out",
        tmp_path / "raw" / "rag" / "mask_review.txt",
        min_documents=2,
    )


def test_run_writes_the_same_files_on_every_run(tmp_path: Path, contract_pdf: None) -> None:
    sources_path = corpus(tmp_path)

    assert run_corpus(tmp_path, sources_path) == 0
    first = {name: (tmp_path / "out" / name).read_bytes() for name in ingest_rag.OUTPUT_NAMES}
    assert run_corpus(tmp_path, sources_path) == 0
    second = {name: (tmp_path / "out" / name).read_bytes() for name in ingest_rag.OUTPUT_NAMES}

    assert first == second
    documents = read_documents(tmp_path / "out" / ingest_rag.DOCUMENTS_NAME)
    assert [document.doc_id for document in documents] == ["prozorro_001", "court_1"]
    manifest_lines = (tmp_path / "out" / ingest_rag.MANIFEST_NAME).read_text().splitlines()
    assert [ManifestEntry.model_validate_json(line).doc_id for line in manifest_lines] == [
        "prozorro_001",
        "court_1",
    ]
    assert "Коваленко" in (tmp_path / "raw" / "rag" / "mask_review.txt").read_text()


def test_run_creates_the_folder_of_the_review_file(tmp_path: Path, contract_pdf: None) -> None:
    review_path = tmp_path / "new" / "folder" / "mask_review.txt"

    exit_code = run(
        corpus(tmp_path), tmp_path / "raw", tmp_path / "texts", tmp_path / "out", review_path,
        min_documents=2,
    )

    assert exit_code == 0
    assert "Коваленко" in review_path.read_text(encoding="utf-8")


def test_run_writes_nothing_while_a_raw_file_is_missing(
    tmp_path: Path, contract_pdf: None, capsys: pytest.CaptureFixture[str]
) -> None:
    sources_path = corpus(tmp_path)
    (tmp_path / "raw" / "rag" / "court_1.html").unlink()

    assert run_corpus(tmp_path, sources_path) == 1

    assert not (tmp_path / "out").exists()
    assert "court_1" in capsys.readouterr().out


def test_run_writes_nothing_when_masking_changed(
    tmp_path: Path, contract_pdf: None, capsys: pytest.CaptureFixture[str]
) -> None:
    sources_path = corpus(tmp_path)
    write(tmp_path / "texts" / "prozorro_001.txt", "інший текст")

    assert run_corpus(tmp_path, sources_path) == 1

    assert not (tmp_path / "out").exists()
    assert "prozorro_001" in capsys.readouterr().out
