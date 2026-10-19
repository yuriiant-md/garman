import pymupdf

from scripts.fetch_prozorro import contract_pdfs, is_candidate, text_flags, text_stats

TENDER = {
    "contracts": [
        {
            "status": "active",
            "documents": [
                {"title": "Договір 45.pdf", "documentType": "contractSigned", "url": "https://x/1"},
                {
                    "title": "Договір 45.pdf.p7s",
                    "documentType": "contractSigned",
                    "url": "https://x/2",
                },
                {
                    "title": "Специфікація.pdf",
                    "documentType": "contractAnnexe",
                    "url": "https://x/3",
                },
                {"title": "Звіт.pdf", "documentType": "evaluationReports", "url": "https://x/4"},
            ],
        },
        {"status": "cancelled", "documents": [{"title": "Старий.pdf", "url": "https://x/5"}]},
    ],
}


def test_contract_pdfs_keeps_signed_contract_pdfs_and_skips_signatures_and_reports():
    urls = [doc["url"] for doc in contract_pdfs(TENDER)]

    assert urls == ["https://x/1", "https://x/3", "https://x/5"]


def test_contract_pdfs_handles_tender_without_contracts():
    assert contract_pdfs({}) == []


def test_is_candidate_needs_grain_in_title_and_large_enough_amount():
    hit = {"title": "Ячмінь фуражний", "value": {"amount": 60_000}}

    assert is_candidate(hit)
    assert not is_candidate({**hit, "title": "Картопля"})
    assert not is_candidate({**hit, "value": {"amount": 12_000}})
    assert not is_candidate({"title": "Пшениця"})


def test_text_flags_find_grain_and_penalty_words():
    flags = text_flags("Постачальник сплачує ПЕНЮ 0,1% за поставку кукурудзи")

    assert flags == {"grain": True, "penalty": True}
    assert text_flags("Договір на поставку картоплі") == {"grain": False, "penalty": False}


def test_text_stats_counts_characters_per_page(tmp_path):
    pdf = tmp_path / "scan.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.new_page().insert_text((72, 72), "x" * 80)
        doc.save(pdf)

    stats = text_stats(pdf)

    assert stats["pages"] == 2
    assert stats["chars_per_page"] == 40
    assert stats["text_layer"] is False  # a scan has almost no text per page
