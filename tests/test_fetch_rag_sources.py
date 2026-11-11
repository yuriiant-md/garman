from pathlib import Path

import httpx

from scripts.fetch_rag_sources import article_links, fetch_source, run
from src.rag.sources import load_sources

HEADER = (
    "doc_id,doc_type,format,title,short_title,source_url,raw_path,"
    "version_date,in_force,cites_commercial_code,articles,retrieved_on\n"
)
LAW_URL = "https://kodeksy.example/law_x.htm"
LAW_INDEX = """
<a href="/law_x/statja-548.htm">548</a>
<a href="/law_x/statja-549.htm">549</a>
<a href="/law_x/14-1.htm">14-1</a>
<a href="/law_x/download.htm">Завантажити</a>
<a href="/other_law/statja-549.htm">чужа 549</a>
"""
LAW_ROW = f'law_x,law,html,Закон X,ЗX,{LAW_URL},rag/law_x,,true,,"549,14-1",2026-11-04\n'
COURT_URL = "https://reyestr.example/Review/1"
COURT_ROW = (
    f"court_1,court_decision,html,Постанова,ВС № 1,{COURT_URL},rag/court_1.html,2024-11-15,,true,,\n"
)
CONTRACT_ROW = "own_001,contract,pdf,Договір own_001,Договір own_001,,own/a.pdf,2026-06-30,,,,\n"


def sources_file(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "sources.csv"
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


CAPTCHA = "<html><body>Підтвердіть, що ви не робот</body></html>"
ARTICLE_PAGE = '<div id="statya"><p>Стаття</p></div> <!-- statya -->'


def site(
    requests: list[str], broken: str | None = None, captcha: str | None = None
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requests.append(url)
        if url == broken:
            return httpx.Response(503)
        if url == captcha:
            return httpx.Response(200, text=CAPTCHA)  # a 200 page without the text we came for
        if url == LAW_URL:
            return httpx.Response(200, text=LAW_INDEX)
        if url == COURT_URL:
            return httpx.Response(200, text=f'<textarea id="txtdepository">{url}</textarea>')
        return httpx.Response(200, text=f'<div id="statya"><p>{url}</p></div> <!-- statya -->')

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_article_links_reads_both_link_styles_of_the_law_and_skips_others():
    links = article_links(LAW_INDEX, LAW_URL)

    assert links == {
        "548": "https://kodeksy.example/law_x/statja-548.htm",
        "549": "https://kodeksy.example/law_x/statja-549.htm",
        "14-1": "https://kodeksy.example/law_x/14-1.htm",
    }


def test_a_law_saves_only_the_wanted_articles(tmp_path):
    (law,) = load_sources(sources_file(tmp_path, LAW_ROW))
    requests: list[str] = []

    failures = fetch_source(site(requests), law, tmp_path / "raw", pause_s=0)

    folder = tmp_path / "raw" / "rag" / "law_x"
    assert failures == []
    assert sorted(path.name for path in folder.iterdir()) == ["14-1.html", "549.html"]
    assert "https://kodeksy.example/law_x/statja-548.htm" not in requests


def test_a_cut_off_table_of_contents_names_the_articles_it_lacks(tmp_path):
    (law,) = load_sources(sources_file(tmp_path, LAW_ROW))
    cut_off = LAW_INDEX.replace('<a href="/law_x/14-1.htm">14-1</a>', "")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=cut_off if str(request.url) == LAW_URL else ARTICLE_PAGE)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    failures = fetch_source(client, law, tmp_path / "raw", pause_s=0)

    assert len(failures) == 1
    assert "14-1" in failures[0]
    assert (tmp_path / "raw" / "rag" / "law_x" / "549.html").exists()


def test_a_cut_off_table_of_contents_is_fine_when_the_lacking_pages_are_on_disk(tmp_path):
    (law,) = load_sources(sources_file(tmp_path, LAW_ROW))
    folder = tmp_path / "raw" / "rag" / "law_x"
    folder.mkdir(parents=True)
    (folder / "14-1.html").write_text("<p>Стаття 14-1</p>", encoding="utf-8")
    cut_off = LAW_INDEX.replace('<a href="/law_x/14-1.htm">14-1</a>', "")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=cut_off if str(request.url) == LAW_URL else ARTICLE_PAGE)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert fetch_source(client, law, tmp_path / "raw", pause_s=0) == []


def test_a_whole_law_with_an_empty_table_of_contents_is_reported(tmp_path):
    (law,) = load_sources(sources_file(tmp_path, LAW_ROW.replace('"549,14-1"', "")))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<p>no links</p>")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert len(fetch_source(client, law, tmp_path / "raw", pause_s=0)) == 1


def test_a_page_already_on_disk_is_not_downloaded_again(tmp_path):
    (court,) = load_sources(sources_file(tmp_path, COURT_ROW))
    saved = tmp_path / "raw" / "rag" / "court_1.html"
    saved.parent.mkdir(parents=True)
    saved.write_text("<p>збережено раніше</p>", encoding="utf-8")
    requests: list[str] = []

    assert fetch_source(site(requests), court, tmp_path / "raw", pause_s=0) == []
    assert requests == []
    assert saved.read_text(encoding="utf-8") == "<p>збережено раніше</p>"


def test_a_failed_download_tells_where_to_save_the_page_by_hand(tmp_path):
    (court,) = load_sources(sources_file(tmp_path, COURT_ROW))

    failures = fetch_source(site([], broken=COURT_URL), court, tmp_path / "raw", pause_s=0)

    assert len(failures) == 1
    assert COURT_URL in failures[0]
    assert str(tmp_path / "raw" / "rag" / "court_1.html") in failures[0]
    assert not (tmp_path / "raw" / "rag" / "court_1.html").exists()


def test_a_page_without_its_text_is_not_saved_and_is_fetched_again_next_time(tmp_path):
    (court,) = load_sources(sources_file(tmp_path, COURT_ROW))
    saved = tmp_path / "raw" / "rag" / "court_1.html"

    failures = fetch_source(site([], captcha=COURT_URL), court, tmp_path / "raw", pause_s=0)

    (failure,) = failures
    assert COURT_URL in failure
    assert "no decision text" in failure
    assert str(saved) in failure
    assert not saved.exists()
    assert fetch_source(site([]), court, tmp_path / "raw", pause_s=0) == []
    assert saved.exists()


def test_a_law_page_without_the_article_is_not_saved(tmp_path):
    (law,) = load_sources(sources_file(tmp_path, LAW_ROW))
    article_url = "https://kodeksy.example/law_x/statja-549.htm"

    failures = fetch_source(site([], captcha=article_url), law, tmp_path / "raw", pause_s=0)

    assert len(failures) == 1
    assert article_url in failures[0]
    assert "no article text" in failures[0]
    assert not (tmp_path / "raw" / "rag" / "law_x" / "549.html").exists()


def test_a_failed_table_of_contents_tells_where_to_save_the_articles(tmp_path):
    (law,) = load_sources(sources_file(tmp_path, LAW_ROW))

    failures = fetch_source(site([], broken=LAW_URL), law, tmp_path / "raw", pause_s=0)

    assert len(failures) == 1
    assert LAW_URL in failures[0]
    assert str(tmp_path / "raw" / "rag" / "law_x") in failures[0]


def test_a_contract_is_already_local_and_makes_no_request(tmp_path):
    (contract,) = load_sources(sources_file(tmp_path, CONTRACT_ROW))
    requests: list[str] = []

    assert fetch_source(site(requests), contract, tmp_path / "raw", pause_s=0) == []
    assert requests == []


def test_run_fails_while_the_corpus_is_incomplete_and_passes_once_it_is(tmp_path, capsys):
    path = sources_file(tmp_path, LAW_ROW, COURT_ROW, CONTRACT_ROW)
    raw = tmp_path / "raw"

    assert run(path, raw, site([]), pause_s=0, min_documents=3) == 1
    assert "own_001" in capsys.readouterr().out

    (raw / "own").mkdir()
    (raw / "own" / "a.pdf").write_bytes(b"%PDF")
    assert run(path, raw, site([]), pause_s=0, min_documents=3) == 0
    assert "3 documents" in capsys.readouterr().out
