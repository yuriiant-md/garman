from datetime import date

import pytest

from src.extraction.scoring import Counts, normalize_text, pair_penalties, score_deal, values_match
from src.schemas.deal import Crop, Deal, Party, PaymentTrigger, Penalty

LATE_DELIVERY = {
    "payer": "seller", "trigger": "late_delivery", "amount": 0.1, "unit": "pct_per_day",
    "cap_pct": 10,
}
LATE_PAYMENT = {
    "payer": "buyer", "trigger": "late_payment", "amount": 2, "unit": "nbu_rate_multiple",
    "cap_pct": None,
}
GOLDEN = {
    "crop": "wheat",
    "quantity_t": 2500,
    "tolerance": None,
    "price": {"amount_per_t": 9250, "currency": "UAH", "vat_included": False},
    "basis": "FCA",
    "delivery": {"place": "Миколаїв", "date_from": "2024-11-09", "date_to": "2024-11-14"},
    "payment": {
        "days": 5, "days_type": "banking", "trigger": "after_delivery",
        "vat_after_registration": True,
    },
    "penalties": [LATE_DELIVERY],
    "quality": None,
    "buyer": 'ТОВ "Зерно Поле 01"',
}
# 21 leaf fields outside penalties + 5 for the one penalty; tolerance and quality are null
ALL_CORRECT = Counts(filled=19, filled_correct=19, empty=7, fabricated=0, total=26, correct=26)


def deal(**changes: object) -> Deal:
    return Deal.model_validate({**GOLDEN, **changes})


def counts(expected: Deal, predicted: Deal | None) -> Counts:
    return Counts.of(score_deal(expected, predicted))


def test_identical_deals_score_every_field_correct():
    assert counts(deal(), deal()) == ALL_CORRECT


def test_empty_nested_object_equals_a_missing_one():
    assert counts(deal(), deal(tolerance={}, quality={})) == ALL_CORRECT


def test_period_raw_is_not_scored():
    predicted = deal(delivery={**GOLDEN["delivery"], "period_raw": "протягом 5 днів"})

    scores = score_deal(deal(), predicted)

    assert "delivery.period_raw" not in {score.field for score in scores}
    assert Counts.of(scores) == ALL_CORRECT


@pytest.mark.parametrize(
    ("expected", "predicted", "match"),
    [
        (9200, 9200.0, True),
        (9200, 9200.004, True),
        (9200, 9250, False),
        (0.1, 0.11, False),
        (0.005, 0.0049, True),  # no rounding bucket boundary between close numbers
        (5, 5, True),
    ],
)
def test_numbers_match_when_equal_to_the_hundredth(expected, predicted, match):
    assert values_match(expected, predicted) is match


def test_false_is_a_value_not_a_null_and_not_a_number():
    assert not values_match(False, None)
    assert not values_match(None, False)
    assert not values_match(True, 1.0)


def test_enums_and_dates_match_only_exactly():
    assert values_match(Crop.WHEAT, Crop.WHEAT)
    assert not values_match(PaymentTrigger.AFTER_DELIVERY, PaymentTrigger.AFTER_DOCUMENTS)
    assert not values_match(date(2024, 11, 9), date(2024, 11, 10))


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
    assert values_match(
        "Державне підприємство «Лугинське лісове господарство»",
        "ДП «Лугинське лісове господарство»",
    )


def test_text_normalization_keeps_words_that_start_like_a_legal_form():
    assert normalize_text("Товарна біржа") == "товарна біржа"


def test_apostrophes_do_not_split_words():
    assert normalize_text("Пʼятихатки") == normalize_text("П'ятихатки") == "пятихатки"


def test_text_tolerates_ocr_noise():
    # buyer of prozorro_007 as OCR read it vs the name a model would write
    assert values_match("ІЛЛШЕЦЬКИИ АГРАРНИЙ ФАХОВИИ КОЛЕДЖ", "Іллінецький аграрний фаховий коледж")
    assert values_match("вул. Шиферна, З, м. Кривий Ріг", "вул. Шиферна, 3, м. Кривий Ріг")


@pytest.mark.parametrize(
    ("expected", "predicted"),
    [
        ('ТОВ "Зерно Поле 01"', "ТОВ "),  # a bare legal form names nobody
        ("Одеса", "Odessa"),  # transliteration is not the name from the text
        ("Миколаїв", "Миколаєва"),
        ("Зерно Поле", "Агро Світ"),
    ],
)
def test_text_rejects_a_different_name(expected, predicted):
    assert not values_match(expected, predicted)


def test_penalties_pair_by_payer_trigger_and_unit_in_any_order():
    expected = deal(penalties=[LATE_DELIVERY, LATE_PAYMENT])
    predicted = deal(penalties=[LATE_PAYMENT, LATE_DELIVERY])

    result = counts(expected, predicted)

    assert result.total == 21 + 2 * 5
    assert result.correct == result.total


def test_penalty_with_several_candidates_pairs_with_the_closest_one():
    golden = Penalty.model_validate(LATE_DELIVERY)
    far = Penalty.model_validate({**LATE_DELIVERY, "amount": 0.5, "cap_pct": 20})
    close = Penalty.model_validate({**LATE_DELIVERY, "cap_pct": 20})

    pairs = pair_penalties([golden], [far, close])

    assert pairs == ((golden, close), (None, far))


def test_penalty_pairing_does_not_depend_on_golden_order():
    other = Penalty.model_validate({**LATE_DELIVERY, "amount": 0.5, "cap_pct": None})
    exact = Penalty.model_validate(LATE_DELIVERY)
    answer = Penalty.model_validate(LATE_DELIVERY)

    # `other` comes first and shares the key, but `answer` agrees with `exact` on every field
    pairs = pair_penalties([other, exact], [answer])

    assert pairs == ((other, None), (exact, answer))


def test_missing_penalty_makes_all_its_fields_wrong():
    result = counts(deal(), deal(penalties=None))

    assert result == Counts(
        filled=19, filled_correct=14, empty=7, fabricated=0, total=26, correct=21
    )


def test_extra_penalty_counts_its_values_as_fabricated():
    extra = {**LATE_DELIVERY, "payer": "buyer", "cap_pct": None}

    result = counts(deal(penalties=None), deal(penalties=[extra]))

    # all 5 fields of the extra penalty are wrong; the 4 with a value are fabricated
    assert result == Counts(
        filled=14, filled_correct=14, empty=12, fabricated=4, total=26, correct=21
    )


def test_penalty_with_another_payer_is_one_missing_and_one_extra():
    predicted = deal(penalties=[{**LATE_DELIVERY, "payer": Party.BUYER}])

    result = counts(deal(), predicted)

    assert result.total == 21 + 2 * 5
    assert result.filled_correct == 14


def test_null_and_empty_penalty_lists_are_equal():
    assert counts(deal(penalties=None), deal(penalties=[])).correct == 21


def test_value_where_golden_has_null_is_fabricated():
    result = counts(deal(), deal(tolerance={"pct": 5}))

    assert result.fabricated == 1
    assert result.correct == 25


def test_failed_answer_makes_every_field_wrong_and_every_null_fabricated():
    assert counts(deal(), None) == Counts(
        filled=19, filled_correct=0, empty=7, fabricated=7, total=26, correct=0
    )


def test_counts_add_up_field_by_field():
    left = Counts(filled=2, filled_correct=1, empty=3, fabricated=1, total=5, correct=3)
    right = Counts(filled=1, filled_correct=1, empty=0, fabricated=0, total=1, correct=1)

    assert left + right == Counts(
        filled=3, filled_correct=2, empty=3, fabricated=1, total=6, correct=4
    )
