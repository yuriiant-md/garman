from datetime import date

import pytest

from src.extraction.scoring import (
    Counts,
    drop_explanation,
    normalize_text,
    pair_penalties,
    score_deal,
    values_match,
)
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
    "price": {"currency": "UAH", "vat_included": False},
    "basis": "FCA",
    "delivery": {"place": "Миколаїв", "date_from": "2024-11-09", "date_to": "2024-11-14"},
    "payment": {
        "days": 5, "days_type": "banking", "trigger": "after_delivery",
        "vat_after_registration": True,
    },
    "penalties": [LATE_DELIVERY],
    "quality": None,
    "grades": [{"grade": None, "amount_per_t": 9250}],
    "buyer": 'ТОВ "Зерно Поле 01"',
}
# 17 leaf fields outside the lists, 2 for the grade and 5 for the penalty; tolerance, both quality
# fields and the grade's class are null
ALL_CORRECT = Counts(filled=19, filled_correct=19, empty=5, fabricated=0, total=24, correct=24)
LEAVES_AND_GRADE = 17 + 2


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

    assert result.total == LEAVES_AND_GRADE + 2 * 5
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
        filled=19, filled_correct=14, empty=5, fabricated=0, total=24, correct=19
    )


def test_extra_penalty_is_one_wrong_extra_item_outside_the_fabricated_share():
    extra = {**LATE_DELIVERY, "payer": "buyer", "cap_pct": None}

    result = counts(deal(penalties=None), deal(penalties=[extra]))

    # golden has no penalty fields, so the base of the fabricated share stays the 5 golden nulls
    assert result == Counts(
        filled=14, filled_correct=14, empty=5, fabricated=0, total=20, correct=19, extra=1
    )


def test_penalty_with_another_payer_is_one_missing_and_one_extra():
    predicted = deal(penalties=[{**LATE_DELIVERY, "payer": Party.BUYER}])

    result = counts(deal(), predicted)

    assert result.total == LEAVES_AND_GRADE + 5 + 1
    assert result.filled_correct == 14
    assert result.extra == 1


def test_null_and_empty_penalty_lists_are_equal():
    assert counts(deal(penalties=None), deal(penalties=[])).correct == LEAVES_AND_GRADE


def test_value_where_golden_has_null_is_fabricated():
    result = counts(deal(), deal(tolerance={"pct": 5}))

    assert result.fabricated == 1
    assert result.correct == 23


def test_failed_answer_makes_every_field_wrong_and_every_null_fabricated():
    assert counts(deal(), None) == Counts(
        filled=19, filled_correct=0, empty=5, fabricated=5, total=24, correct=0
    )


def test_counts_add_up_field_by_field():
    left = Counts(filled=2, filled_correct=1, empty=3, fabricated=1, total=5, correct=3, extra=1)
    right = Counts(filled=1, filled_correct=1, empty=0, fabricated=0, total=1, correct=1)

    assert left + right == Counts(
        filled=3, filled_correct=2, empty=3, fabricated=1, total=6, correct=4, extra=1
    )


CURRENCY = {"currency": "UAH", "vat_included": True}
MOISTURE = {"name": "moisture", "limit": "max", "value": 14, "unit": "pct"}
GLUTEN = {"name": "gluten", "limit": "min", "value": 23, "unit": "pct"}
RYE = {"name": "other", "limit": "max", "value": 4, "unit": "pct", "raw_name": "Зерна жита"}
SECOND = {"grade": "2", "amount_per_t": 7100, "indicators": [MOISTURE, GLUTEN]}
FOURTH = {"grade": "4", "amount_per_t": 6300, "indicators": [MOISTURE]}
# GOLDEN without its grade and penalty: 17 leaf fields, 13 filled and 4 null
PLAIN = {"price": CURRENCY, "penalties": None, "grades": None}


def graded(*grades: dict[str, object]) -> Deal:
    return deal(**{**PLAIN, "grades": list(grades)})


def by_field(expected: Deal, predicted: Deal | None) -> dict[str, list[bool]]:
    result: dict[str, list[bool]] = {}
    for score in score_deal(expected, predicted):
        result.setdefault(score.field, []).append(score.correct)
    return result


def test_grades_pair_by_class_in_any_order():
    result = counts(graded(SECOND, FOURTH), graded(FOURTH, SECOND))

    # each grade: its class and price, then one unit per indicator (2 + 1)
    assert result == Counts(
        filled=13 + 2 * 2 + 3, filled_correct=20, empty=4, fabricated=0, total=24, correct=24
    )


def test_indicator_is_one_unit_named_after_the_indicator():
    fields = by_field(graded(SECOND), graded({**SECOND, "indicators": [MOISTURE, GLUTEN]}))

    assert fields["grades.indicators.moisture"] == [True]
    assert fields["grades.indicators.gluten"] == [True]
    assert fields["grades.grade"] == fields["grades.amount_per_t"] == [True]


@pytest.mark.parametrize(
    "answer",
    [
        {**GLUTEN, "value": 18},
        {**GLUTEN, "unit": "other"},
    ],
)
def test_indicator_with_another_value_or_unit_is_wrong(answer):
    predicted = graded({**SECOND, "indicators": [MOISTURE, answer]})

    assert by_field(graded(SECOND), predicted)["grades.indicators.gluten"] == [False]


def test_indicator_with_another_limit_is_one_missing_and_one_extra():
    predicted = graded({**SECOND, "indicators": [MOISTURE, {**GLUTEN, "limit": "max"}]})

    result = counts(graded(SECOND), predicted)

    assert by_field(graded(SECOND), predicted)["grades.indicators.gluten"] == [False]
    assert result.extra == 1
    assert result.fabricated == 0


def test_other_indicators_pair_by_a_similar_written_name():
    golden = graded({**SECOND, "indicators": [RYE]})
    predicted = graded({**SECOND, "indicators": [{**RYE, "raw_name": "зерна жита,"}]})

    assert by_field(golden, predicted)["grades.indicators.other"] == [True]


CEREALS = {
    "name": "other", "limit": "max", "value": 4, "unit": "pct",
    "raw_name": (
        "Зерна злакових культур, що за характером пошкоджень не віднесені до смітної домішки"
    ),
}


@pytest.mark.parametrize(
    "written",
    [
        f"{CEREALS['raw_name']} (зерна жита, тритикале, ячменю)",  # the explanation the rule drops
        f"{CEREALS['raw_name']} (жито) (тритикале)",
    ],
)
def test_other_indicator_names_are_compared_without_the_explanation_in_brackets(written):
    golden = graded({**SECOND, "indicators": [CEREALS]})
    predicted = graded({**SECOND, "indicators": [{**CEREALS, "raw_name": written}]})

    assert by_field(golden, predicted)["grades.indicators.other"] == [True]


def test_brackets_do_not_make_other_names_equal():
    golden = graded({**SECOND, "indicators": [RYE]})
    predicted = graded({**SECOND, "indicators": [{**RYE, "raw_name": "Зерна ячменю (жита)"}]})

    assert by_field(golden, predicted)["grades.indicators.other"] == [False]


@pytest.mark.parametrize(
    ("written", "kept"),
    [
        ("Зерна злакових культур (жита, ячменю)", "Зерна злакових культур"),
        ("Сажка (всіх видів), ріжки (будь-які)", "Сажка, ріжки"),
        ("(в тому числі ушкоджені під час сушіння", "(в тому числі ушкоджені під час сушіння"),
    ],
)
def test_only_a_closed_bracket_is_dropped_as_an_explanation(written, kept):
    # an unclosed bracket is OCR noise around the name itself, so it stays
    assert drop_explanation(written) == kept


def test_missing_grade_makes_its_class_price_and_indicators_wrong():
    result = counts(graded(SECOND, FOURTH), graded(SECOND))

    # FOURTH: grade, price and moisture are wrong
    assert (result.filled, result.filled_correct, result.extra) == (20, 17, 0)


def test_extra_grade_is_one_extra_item_and_its_indicators_are_not_scored():
    result = counts(graded(SECOND), graded(SECOND, FOURTH))

    assert result == Counts(
        filled=13 + 2 + 2, filled_correct=17, empty=4, fabricated=0, total=22, correct=21, extra=1
    )


def test_grade_without_class_pairs_with_an_answer_without_class():
    unnamed = {"grade": None, "amount_per_t": 9200}

    fields = by_field(graded(unnamed), graded(unnamed))

    assert fields["grades.grade"] == fields["grades.amount_per_t"] == [True]


def test_deals_without_grades_score_no_grade_fields():
    fields = by_field(deal(**PLAIN), deal(**{**PLAIN, "grades": []}))

    assert not any(field.startswith("grades") for field in fields)


def test_failed_answer_makes_every_grade_field_and_indicator_wrong():
    result = counts(graded(SECOND), None)

    # golden nulls (4) count as fabricated, as for any failed answer
    assert result == Counts(
        filled=13 + 2 + 2, filled_correct=0, empty=4, fabricated=4, total=21, correct=0
    )


def test_lab_and_the_party_that_pays_for_analyses_are_scored_apart():
    # own_001, п. 2.3: the delivery point's laboratory «за рахунок Покупця»
    golden = deal(quality={"lab": "other", "analysis_paid_by": "buyer"})
    predicted = deal(quality={"lab": "buyer", "analysis_paid_by": "buyer"})

    fields = by_field(golden, predicted)

    assert fields["quality.lab"] == [False]
    assert fields["quality.analysis_paid_by"] == [True]
