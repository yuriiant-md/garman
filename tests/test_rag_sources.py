from datetime import date
from pathlib import Path

import pytest

from src.rag.sources import (
    MIN_DOCUMENTS,
    DocFormat,
    DocType,
    corpus_problems,
    dating_problem,
    load_sources,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "data" / "rag_sources.csv"
HEADER = (
    "doc_id,doc_type,format,title,short_title,source_url,raw_path,"
    "version_date,in_force,cites_commercial_code,articles,retrieved_on\n"
)
LAW_ROW = (
    "law_x,law,html,Закон X,ЗX,https://example.org/x.htm,rag/law_x,,true,,"
    '"549..552,625,14-1",2026-11-04\n'
)
COURT_ROW = (
    "court_1,court_decision,html,Постанова у справі № 1/2/3,ВС № 1/2/3,"
    "https://example.org/Review/1,rag/court_1.html,2024-11-15,,true,,\n"
)
CONTRACT_ROW = "own_001,contract,pdf,Договір own_001,Договір own_001,,own/a.pdf,2026-06-30,,,,\n"


def write_sources(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "sources.csv"
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


def test_load_sources_reads_types_and_turns_empty_cells_into_none(tmp_path):
    law, court, contract = load_sources(write_sources(tmp_path, LAW_ROW, COURT_ROW, CONTRACT_ROW))

    assert law.doc_type is DocType.LAW
    assert law.format is DocFormat.HTML
    assert law.in_force is True
    assert law.version_date is None
    assert court.version_date == date(2024, 11, 15)
    assert court.cites_commercial_code is True
    assert contract.format is DocFormat.PDF
    assert contract.source_url is None
    assert contract.articles is None


def test_load_sources_rejects_a_repeated_doc_id(tmp_path):
    path = write_sources(tmp_path, LAW_ROW, LAW_ROW)

    with pytest.raises(ValueError, match="law_x"):
        load_sources(path)


@pytest.mark.parametrize(
    "row",
    [
        LAW_ROW.replace(",law,", ",statute,"),
        LAW_ROW.replace(",html,", ",docx,"),
        COURT_ROW.replace("2024-11-15", "15.11.2024"),
        LAW_ROW.replace("rag/law_x", "../law_x"),
        LAW_ROW.replace("rag/law_x", "/tmp/law_x"),
        LAW_ROW.replace("law_x,law", "Law X,law"),
        LAW_ROW.replace('"549..552,625,14-1"', '"ст. 549"'),
        LAW_ROW.replace("https://example.org/x.htm", "example.org/x.htm"),
        COURT_ROW.replace("Постанова у справі", "Постанова, у справі"),  # an unquoted comma
    ],
)
def test_load_sources_names_the_line_of_a_bad_row(tmp_path, row):
    path = write_sources(tmp_path, CONTRACT_ROW, row)

    with pytest.raises(ValueError, match=r"sources\.csv:3"):
        load_sources(path)


@pytest.mark.parametrize(
    ("article", "wanted"),
    [("549", True), ("552", True), ("625", True), ("14-1", True), ("553", False), ("14", False)],
)
def test_wants_article_follows_ranges_and_single_articles(tmp_path, article, wanted):
    (law,) = load_sources(write_sources(tmp_path, LAW_ROW))

    assert law.wants_article(article) is wanted


def test_a_range_also_takes_the_inserted_articles_inside_it(tmp_path):
    (law,) = load_sources(write_sources(tmp_path, LAW_ROW.replace('"549..552,625,14-1"', "14..15")))

    assert law.wants_article("14-1")
    assert not law.wants_article("16")


def test_named_articles_are_the_single_articles_and_the_ends_of_each_range(tmp_path):
    (law,) = load_sources(write_sources(tmp_path, LAW_ROW))

    assert law.named_articles() == ("549", "552", "625", "14-1")


def test_a_law_without_articles_takes_every_article(tmp_path):
    (law,) = load_sources(write_sources(tmp_path, LAW_ROW.replace('"549..552,625,14-1"', "")))

    assert law.wants_article("1")
    assert law.wants_article("14-1")


def raw_files(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    (raw / "rag" / "law_x").mkdir(parents=True)
    (raw / "rag" / "law_x" / "549.html").write_text("<p>Стаття 549</p>", encoding="utf-8")
    (raw / "rag" / "court_1.html").write_text("<p>Постанова</p>", encoding="utf-8")
    (raw / "own").mkdir()
    (raw / "own" / "a.pdf").write_bytes(b"%PDF")
    return raw


def test_corpus_problems_lists_a_missing_raw_file_and_an_empty_law_folder(tmp_path):
    sources = load_sources(write_sources(tmp_path, LAW_ROW, COURT_ROW, CONTRACT_ROW))
    raw = raw_files(tmp_path)
    (raw / "rag" / "court_1.html").unlink()
    (raw / "rag" / "law_x" / "549.html").unlink()

    problems = corpus_problems(sources, raw, min_documents=3)

    assert any("court_1" in problem and "court_1.html" in problem for problem in problems)
    assert any("law_x" in problem for problem in problems)


def test_corpus_problems_needs_enough_documents_and_two_formats(tmp_path):
    raw = raw_files(tmp_path)
    one_format = load_sources(write_sources(tmp_path, LAW_ROW, COURT_ROW))

    problems = corpus_problems(one_format, raw, min_documents=3)

    assert any(str(3) in problem for problem in problems)
    assert any("format" in problem for problem in problems)


def test_corpus_problems_names_a_document_without_a_date_or_a_version(tmp_path):
    undated_contract = CONTRACT_ROW.replace("2026-06-30", "")
    law_without_snapshot = LAW_ROW.replace("2026-11-04", "")
    sources = load_sources(write_sources(tmp_path, law_without_snapshot, COURT_ROW, undated_contract))

    problems = corpus_problems(sources, raw_files(tmp_path), min_documents=3)

    assert any("own_001" in problem and "version_date" in problem for problem in problems)
    assert any("law_x" in problem and "retrieved_on" in problem for problem in problems)
    assert not any("court_1" in problem for problem in problems)


def test_load_sources_reads_the_date_a_law_was_retrieved(tmp_path):
    law, contract = load_sources(write_sources(tmp_path, LAW_ROW, CONTRACT_ROW))

    assert law.retrieved_on == date(2026, 11, 4)
    assert contract.version_date == date(2026, 6, 30)
    assert contract.retrieved_on is None


def test_every_document_in_the_project_list_has_a_date_or_a_version():
    sources = load_sources(SOURCES_PATH)

    assert [problem for s in sources if (problem := dating_problem(s))] == []


def test_corpus_problems_is_empty_for_a_complete_corpus(tmp_path):
    sources = load_sources(write_sources(tmp_path, LAW_ROW, COURT_ROW, CONTRACT_ROW))

    assert corpus_problems(sources, raw_files(tmp_path), min_documents=3) == []


def test_the_project_source_list_meets_the_course_minimum_without_raw_files():
    sources = load_sources(SOURCES_PATH)

    assert len(sources) >= MIN_DOCUMENTS
    assert {source.format for source in sources} == {DocFormat.PDF, DocFormat.HTML}
    assert {source.doc_type for source in sources} == set(DocType)


def test_every_law_and_decision_in_the_project_list_has_a_url_and_a_short_title():
    online = [s for s in load_sources(SOURCES_PATH) if s.doc_type is not DocType.CONTRACT]

    assert online
    assert all(source.source_url and source.short_title for source in online)


def test_the_repealed_commercial_code_is_marked_as_not_in_force():
    laws = {s.short_title: s for s in load_sources(SOURCES_PATH) if s.doc_type is DocType.LAW}

    assert laws["ГК"].in_force is False
    assert laws["ЦК"].in_force is True
