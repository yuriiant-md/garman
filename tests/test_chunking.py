from datetime import date

import pytest

from src.rag.chunking import Chunk, chunk_document, fixed_size_chunks, pack_units
from src.rag.documents import Document, DocumentPart, ManifestEntry
from src.rag.sources import DocFormat, DocType, Source
from src.rag.structure import Line, Unit


def unit(text: str, *, section: str | None = "1. Предмет", ref: str | None = None) -> Unit:
    return Unit(lines=tuple(Line(1, line) for line in text.splitlines()), section=section, ref=ref)


def test_short_units_of_one_section_merge_and_the_ref_becomes_a_range() -> None:
    units = (
        unit("1.1. Товар.", ref="п. 1.1"),
        unit("1.2. Ціна.", ref="п. 1.2"),
        unit("1.3. Строк.", ref="п. 1.3"),
    )

    (merged,) = pack_units(units, max_chars=60, min_chars=40)

    assert merged.text == "1.1. Товар.\n1.2. Ціна.\n1.3. Строк."
    assert merged.ref == "п. 1.1–1.3"
    assert merged.section == "1. Предмет"


def test_units_of_different_sections_never_merge() -> None:
    units = (unit("1.1. Товар.", ref="п. 1.1"), unit("2.1. Ціна.", section="2. Ціна", ref="п. 2.1"))

    assert pack_units(units, max_chars=60, min_chars=40) == units


def test_units_long_enough_stay_one_per_chunk() -> None:
    units = (unit("1.1. Товар — пшениця.", ref="п. 1.1"), unit("1.2. Ціна — 9000 грн.", ref="п. 1.2"))

    assert pack_units(units, max_chars=60, min_chars=10) == units


def test_merge_stops_at_the_size_limit() -> None:
    units = (
        unit("1.1. " + "а" * 20, ref="п. 1.1"),
        unit("1.2. б", ref="п. 1.2"),
        unit("1.3. " + "в" * 20, ref="п. 1.3"),
    )

    first, second = pack_units(units, max_chars=40, min_chars=10)

    assert (first.ref, second.ref) == ("п. 1.1–1.2", "п. 1.3")


def test_long_unit_is_split_between_lines_and_keeps_its_ref_and_pages() -> None:
    clause = Unit(
        lines=(
            Line(1, "4.1. Ціна товару становить"),
            Line(1, "9000 грн за тонну"),
            Line(2, "без ПДВ."),
        ),
        section="4. Ціна",
        ref="п. 4.1",
    )

    pieces = pack_units((clause,), max_chars=45, min_chars=1)

    assert [(p.text, p.ref, p.pages) for p in pieces] == [
        ("4.1. Ціна товару становить\n9000 грн за тонну", "п. 4.1", (1,)),
        ("без ПДВ.", "п. 4.1", (2,)),
    ]


def test_long_line_is_split_at_sentence_ends_then_between_words() -> None:
    line = "Перше речення тут. Друге речення тут. " + "слово " * 10

    pieces = pack_units((unit(line.strip(), ref="п. 9.1"),), max_chars=25, min_chars=1)

    assert all(len(piece.text) <= 25 for piece in pieces)
    assert pieces[0].text == "Перше речення тут."
    assert " ".join(piece.text for piece in pieces) == line.strip()


def test_refs_with_different_prefixes_are_listed_and_missing_refs_are_skipped() -> None:
    units = (unit("ДОГОВІР № 1", ref="преамбула"), unit("м. Київ"), unit("1.1. Товар.", ref="п. 1.1"))

    (merged,) = pack_units(units, max_chars=60, min_chars=40)

    assert merged.ref == "преамбула, п. 1.1"


def test_merged_unit_without_refs_has_no_ref() -> None:
    (merged,) = pack_units((unit("ПОКУПЕЦЬ"), unit("ТОВ «Лан»")), max_chars=60, min_chars=40)

    assert merged.ref is None


def source(doc_id: str, doc_type: DocType, **fields: object) -> Source:
    return Source.model_validate(
        {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "format": DocFormat.HTML,
            "title": f"Назва {doc_id}",
            "short_title": doc_id.upper(),
            "raw_path": f"rag/{doc_id}",
            **fields,
        }
    )


def entry(doc_id: str, doc_type: DocType) -> ManifestEntry:
    return ManifestEntry(
        doc_id=doc_id,
        doc_type=doc_type,
        format=DocFormat.HTML,
        raw_sha256="r" * 64,
        text_sha256="t" * 64,
        parts=1,
        chars=100,
    )


def contract_chunk(number: int, ref: str, page: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"own_001#{number:03d}",
        doc_id="own_001",
        doc_type=DocType.CONTRACT,
        title="Назва own_001",
        short_title="OWN_001",
        text_sha256="t" * 64,
        section="1. Предмет",
        ref=ref,
        pages=(page,),
        text=text,
    )


def test_contract_chunks_carry_numbered_ids_refs_pages_and_document_metadata() -> None:
    document = Document(
        doc_id="own_001",
        parts=(
            DocumentPart(page=1, text="1. Предмет\n1.1. Продавець передає пшеницю 3 класу."),
            DocumentPart(page=2, text="1.2. Покупець оплачує її протягом 5 днів."),
        ),
    )
    contract_source = source("own_001", DocType.CONTRACT, format=DocFormat.PDF)

    chunks = chunk_document(
        document, contract_source, entry("own_001", DocType.CONTRACT), min_chars=10
    )

    assert chunks == (
        contract_chunk(1, "п. 1.1", 1, "1.1. Продавець передає пшеницю 3 класу."),
        contract_chunk(2, "п. 1.2", 2, "1.2. Покупець оплачує її протягом 5 днів."),
    )


def test_law_chunks_carry_the_edition_and_in_force_flag() -> None:
    heading = "Стаття 231. Розмір штрафних санкцій"
    document = Document(
        doc_id="law_commercial_code",
        parts=(DocumentPart(article="231", heading=heading, text="1. Законом…"),),
    )
    law_source = source(
        "law_commercial_code", DocType.LAW, version_date=date(2025, 8, 28), in_force=False
    )

    (chunk,) = chunk_document(document, law_source, entry("law_commercial_code", DocType.LAW))

    assert (chunk.ref, chunk.heading, chunk.pages) == ("ст. 231", heading, ())
    assert (chunk.version_date, chunk.in_force) == (date(2025, 8, 28), False)


def test_decision_chunks_carry_the_commercial_code_flag() -> None:
    document = Document(
        doc_id="court_1", parts=(DocumentPart(text="ВСТАНОВИВ:\n5. Суд дійшов висновку."),)
    )
    court_source = source("court_1", DocType.COURT_DECISION, cites_commercial_code=True)

    (chunk,) = chunk_document(document, court_source, entry("court_1", DocType.COURT_DECISION))

    assert (chunk.section, chunk.ref, chunk.cites_commercial_code) == ("ВСТАНОВИВ:", "п. 5", True)


def test_chunk_document_refuses_metadata_of_another_document() -> None:
    document = Document(doc_id="court_1", parts=(DocumentPart(text="5. Суд дійшов висновку."),))
    own_source = source("court_1", DocType.COURT_DECISION)
    own_entry = entry("court_1", DocType.COURT_DECISION)

    with pytest.raises(ValueError, match="court_2"):
        chunk_document(document, source("court_2", DocType.COURT_DECISION), own_entry)
    with pytest.raises(ValueError, match="court_2"):
        chunk_document(document, own_source, entry("court_2", DocType.COURT_DECISION))


def test_fixed_size_chunks_are_windows_that_overlap() -> None:
    assert fixed_size_chunks("abcdefghij", size=4, overlap=1) == ("abcd", "defg", "ghij")
    assert fixed_size_chunks("abcdefghijk", size=4, overlap=1) == ("abcd", "defg", "ghij", "jk")
    assert fixed_size_chunks("abc", size=4, overlap=1) == ("abc",)
    assert fixed_size_chunks("", size=4, overlap=1) == ()


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (4, 4), (4, -1)])
def test_fixed_size_chunks_refuse_a_window_that_cannot_move(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap"):
        fixed_size_chunks("abcdef", size=size, overlap=overlap)
