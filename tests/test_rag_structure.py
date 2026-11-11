from src.rag.documents import Document, DocumentPart
from src.rag.structure import (
    REQUISITES_SECTION,
    Line,
    Unit,
    contract_units,
    decision_units,
    law_units,
)


def contract(*pages: str) -> Document:
    return Document(
        doc_id="own_001",
        parts=tuple(DocumentPart(page=n, text=text) for n, text in enumerate(pages, start=1)),
    )


def unit(
    text: str,
    *,
    page: int | None = 1,
    section: str | None = None,
    heading: str | None = None,
    ref: str | None = None,
) -> Unit:
    lines = tuple(Line(page, line) for line in text.splitlines())
    return Unit(lines=lines, section=section, heading=heading, ref=ref)


def test_unit_text_joins_its_lines_and_pages_are_listed_once() -> None:
    clause = Unit(
        lines=(Line(1, "4.1. Ціна товару"), Line(2, "становить 9000 грн"), Line(2, "за тонну.")),
        ref="п. 4.1",
    )

    assert clause.text == "4.1. Ціна товару\nстановить 9000 грн\nза тонну."
    assert clause.pages == (1, 2)


SUBJECT = "1. Предмет Договору"
LIMIT = 1500  # every clause in these tests is shorter


def test_contract_splits_into_preamble_and_clauses_under_their_sections() -> None:
    document = contract(
        "ДОГОВІР ПОСТАВКИ № 1\nм. Київ\n\n1. Предмет Договору\n1.1. Продавець передає\n"
        "товар Покупцю.\n1.2. Покупець оплачує товар.\n2. Ціна\n2.1. Ціна 9000 грн за тонну."
    )

    assert contract_units(document, max_chars=LIMIT) == (
        unit("ДОГОВІР ПОСТАВКИ № 1\nм. Київ", ref="преамбула"),
        unit("1.1. Продавець передає\nтовар Покупцю.", section=SUBJECT, ref="п. 1.1"),
        unit("1.2. Покупець оплачує товар.", section=SUBJECT, ref="п. 1.2"),
        unit("2.1. Ціна 9000 грн за тонну.", section="2. Ціна", ref="п. 2.1"),
    )


def test_contract_clause_spanning_two_pages_keeps_both_pages() -> None:
    document = contract("4.1. Ціна товару", "становить 9000 грн.\n4.2. ПДВ 20%.")

    clause, vat = contract_units(document, max_chars=LIMIT)

    assert (clause.ref, clause.pages) == ("п. 4.1", (1, 2))
    assert (vat.ref, vat.pages) == ("п. 4.2", (2,))


def test_contract_subclauses_and_ocr_spacing_stay_with_their_clause() -> None:
    document = contract(
        "5.1. Покупець зобов’язаний:\n5.1.1. прийняти товар;\n5.1.2. оплатити товар.\n"
        "6.1 .Продавець має право\nвимагати оплати.\n6.2. Сторони"
    )

    assert [(u.ref, u.text) for u in contract_units(document, max_chars=LIMIT)] == [
        ("п. 5.1", "5.1. Покупець зобов’язаний:\n5.1.1. прийняти товар;\n5.1.2. оплатити товар."),
        ("п. 6.1", "6.1 .Продавець має право\nвимагати оплати."),
        ("п. 6.2", "6.2. Сторони"),
    ]


def test_long_clause_is_split_into_its_subclauses_under_the_same_section() -> None:
    document = contract(
        "10. Інші положення\n10.6. Постачальник надає документи:\n10.6.1. витяг з реєстру;\n"
        "10.6.2. декларацію.\n10.7. Строк"
    )

    units = contract_units(document, max_chars=40)

    assert [(u.ref, u.text) for u in units] == [
        ("п. 10.6", "10.6. Постачальник надає документи:"),
        ("п. 10.6.1", "10.6.1. витяг з реєстру;"),
        ("п. 10.6.2", "10.6.2. декларацію."),
        ("п. 10.7", "10.7. Строк"),
    ]
    assert {u.section for u in units} == {"10. Інші положення"}


def test_wrapped_reference_to_an_earlier_clause_does_not_start_a_clause() -> None:
    document = contract("3.1. Поставка товару згідно з п.\n1.1 цього Договору.\n3.2. Строк поставки.")

    assert [u.ref for u in contract_units(document, max_chars=LIMIT)] == ["п. 3.1", "п. 3.2"]


def test_wrapped_reference_to_a_later_clause_does_not_start_a_clause() -> None:
    document = contract(
        "4.1. Ціна остаточна в місці поставки згідно п.\n5.1. цього Договору.\n"
        "4.2. У разі зміни ставки ПДВ.\n5. Поставка Товару\n5.1. Постачальник передає Товар."
    )

    units = contract_units(document, max_chars=LIMIT)

    assert [(u.ref, u.lines[0].text) for u in units] == [
        ("п. 4.1", "4.1. Ціна остаточна в місці поставки згідно п."),
        ("п. 4.2", "4.2. У разі зміни ставки ПДВ."),
        ("п. 5.1", "5.1. Постачальник передає Товар."),
    ]


def test_numbered_list_inside_a_clause_is_not_a_section() -> None:
    document = contract("5.1. Покупець зобов’язаний:\n1. Прийняти Товар\n2. Оплатити Товар\n5.2. Строк")

    first, second = contract_units(document, max_chars=LIMIT)

    assert first.text == "5.1. Покупець зобов’язаний:\n1. Прийняти Товар\n2. Оплатити Товар"
    assert (first.section, second.section) == (None, None)


def test_roman_section_number_names_the_section() -> None:
    document = contract("IV. Якість товару та упаковка\n4.1. Товар відповідає ДСТУ.")

    assert contract_units(document, max_chars=LIMIT) == (
        unit("4.1. Товар відповідає ДСТУ.", section="IV. Якість товару та упаковка", ref="п. 4.1"),
    )


def test_requisites_after_the_last_clause_are_split_by_party() -> None:
    document = contract(
        "10.2. Договір набирає чинності з дати підписання.\nПІДПИСИ СТОРІН\nПРОДАВЕЦЬ\n"
        "ТОВ «Золотий Лан»\nIBAN: [IBAN]",
        "ПОКУПЕЦЬ\nФОП [ПІБ]\nСПЕЦИФІКАЦІЯ\nПшениця 3 клас, 500 т",
    )

    assert contract_units(document, max_chars=LIMIT) == (
        unit("10.2. Договір набирає чинності з дати підписання.", ref="п. 10.2"),
        unit("ПІДПИСИ СТОРІН", section=REQUISITES_SECTION),
        unit("ПРОДАВЕЦЬ\nТОВ «Золотий Лан»\nIBAN: [IBAN]", section=REQUISITES_SECTION),
        unit("ПОКУПЕЦЬ\nФОП [ПІБ]", page=2, section=REQUISITES_SECTION),
        unit("СПЕЦИФІКАЦІЯ\nПшениця 3 клас, 500 т", page=2, section=REQUISITES_SECTION),
    )


def test_numbered_requisites_heading_also_starts_the_requisites() -> None:
    document = contract(
        "10.2. Додатки є частиною Договору.\nXI. Юридичні адреси та реквізити Сторін\nТОВ «Лан»"
    )

    assert contract_units(document, max_chars=LIMIT)[1:] == (
        unit("XI. Юридичні адреси та реквізити Сторін\nТОВ «Лан»", section=REQUISITES_SECTION),
    )


def test_party_named_inside_a_clause_does_not_start_the_requisites() -> None:
    document = contract("5.1. Обов’язки сторін:\nПокупець зобов’язаний:\nприйняти товар.\n5.2. Строк")

    assert [u.ref for u in contract_units(document, max_chars=LIMIT)] == ["п. 5.1", "п. 5.2"]


LAW_HEADING = "Стаття 30. Виключна підсудність"


def law(article: str, heading: str, text: str) -> Document:
    return Document(
        doc_id="law_x", parts=(DocumentPart(article=article, heading=heading, text=text),)
    )


def law_unit(text: str, heading: str, ref: str) -> Unit:
    return unit(text, page=None, heading=heading, ref=ref)


def test_law_article_within_the_limit_is_one_unit() -> None:
    heading = "Стаття 625. Відповідальність"
    document = law("625", heading, "1. Боржник не звільняється.\n2. Боржник сплачує.")

    assert law_units(document, max_chars=100) == (
        law_unit("1. Боржник не звільняється.\n2. Боржник сплачує.", heading, "ст. 625"),
    )


def test_long_law_article_is_split_into_its_parts() -> None:
    document = law(
        "30", LAW_HEADING, "1. Спори з перевезення.\nабзац другий частини.\n2. Справи про арешт судна."
    )

    assert law_units(document, max_chars=40) == (
        law_unit("1. Спори з перевезення.\nабзац другий частини.", LAW_HEADING, "ст. 30, ч. 1"),
        law_unit("2. Справи про арешт судна.", LAW_HEADING, "ст. 30, ч. 2"),
    )


def test_tax_code_points_are_cited_as_points_with_their_editorial_notes() -> None:
    document = law(
        "201",
        "Стаття 201. Податкова накладна",
        "201.1. На дату виникнення зобов’язань.\n{Пункт 201.1 із змінами}\n201.2. Форма накладної.",
    )

    assert [(u.ref, u.text) for u in law_units(document, max_chars=40)] == [
        ("ст. 201, п. 201.1", "201.1. На дату виникнення зобов’язань.\n{Пункт 201.1 із змінами}"),
        ("ст. 201, п. 201.2", "201.2. Форма накладної."),
    ]


def test_long_article_text_before_its_first_part_has_the_article_ref() -> None:
    document = law(
        "14-1", "Стаття 14-1. Терміни", "У цьому Кодексі терміни вживаються так:\n1. договір — угода."
    )

    assert [u.ref for u in law_units(document, max_chars=20)] == ["ст. 14-1", "ст. 14-1, ч. 1"]


def decision(text: str) -> Document:
    return Document(doc_id="court_1", parts=(DocumentPart(text=text),))


def decision_unit(text: str, section: str, ref: str | None = None) -> Unit:
    return unit(text, page=None, section=section, ref=ref)


CLAIMS = "1. Короткий зміст позовних вимог"
RULING = "П О С Т А Н О В И В :"


def test_decision_sections_come_from_capital_and_numbered_headings() -> None:
    document = decision(
        "ПОСТАНОВА\nІМЕНЕМ УКРАЇНИ\nВерховний Суд у складі колегії суддів\nІСТОРІЯ СПРАВИ\n"
        "1. Короткий зміст позовних вимог\n1.1. Товариство звернулося до суду.\n"
        "1.2. Позов обґрунтовано так:\nпродавець не поставив зерно.\n"
        "П О С Т А Н О В И В :\nКасаційну скаргу залишити без задоволення.\nО. [ПІБ]"
    )

    assert decision_units(document) == (
        decision_unit("Верховний Суд у складі колегії суддів", "ІМЕНЕМ УКРАЇНИ"),
        decision_unit("1.1. Товариство звернулося до суду.", CLAIMS, "п. 1.1"),
        decision_unit("1.2. Позов обґрунтовано так:", CLAIMS, "п. 1.2"),
        decision_unit("продавець не поставив зерно.", CLAIMS),
        decision_unit("Касаційну скаргу залишити без задоволення.", RULING),
        decision_unit("О. [ПІБ]", RULING),
    )


def test_decision_paragraph_ending_with_a_colon_or_holding_numbers_is_not_a_heading() -> None:
    document = decision(
        "ВСТАНОВИВ:\n8. Згідно з доводами касаційної скарги:\n"
        "10. Оплата на суму 2 541 548 грн не надійшла\n45. Колегія суддів погоджується з цим."
    )

    assert [(u.section, u.ref) for u in decision_units(document)] == [
        ("ВСТАНОВИВ:", "п. 8"),
        ("ВСТАНОВИВ:", "п. 10"),
        ("ВСТАНОВИВ:", "п. 45"),
    ]
