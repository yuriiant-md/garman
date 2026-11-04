from src.extraction.text import normalize_text


def test_text_normalization_drops_case_quotes_and_legal_form():
    assert normalize_text('ТОВ "Зерно Поле 01"') == "зерно поле 01"
    assert normalize_text("«Зерно поле 01»") == "зерно поле 01"
    assert normalize_text("Товариство з обмеженою відповідальністю «Зерно Поле 01»") == (
        "зерно поле 01"
    )
    assert normalize_text("ФГ  «Колос»") == "колос"


def test_text_normalization_drops_other_business_forms():
    assert normalize_text("ТзОВ «Агро-Світ»") == "агро-світ"
    assert normalize_text("ПрАТ «Агро-Світ»") == "агро-світ"
    assert normalize_text("АТ «Укрзалізниця»") == "укрзалізниця"


def test_text_normalization_keeps_words_that_start_like_a_legal_form():
    assert normalize_text("Товарна біржа") == "товарна біржа"


def test_apostrophes_do_not_split_words():
    assert normalize_text("Пʼятихатки") == normalize_text("П'ятихатки") == "пятихатки"
