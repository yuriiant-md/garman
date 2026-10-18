from datetime import date

import pytest

from scripts.synthetic_checks import missing_values
from src.schemas.deal import Deal, Delivery, Penalty, Tolerance

FULL_DEAL = Deal(
    crop="wheat",
    quantity_t=500,
    tolerance={"pct": 5, "option": "buyer"},
    price={"amount_per_t": 9200, "currency": "UAH", "vat_included": True},
    basis="CPT",
    delivery={"place": "Одеса", "date_from": "2026-11-05", "date_to": "2026-11-12"},
    payment={
        "days": 5,
        "days_type": "banking",
        "trigger": "after_delivery",
        "vat_after_registration": True,
    },
    penalties=[
        {
            "payer": "buyer",
            "trigger": "late_payment",
            "amount": 0.1,
            "unit": "pct_per_day",
            "cap_pct": 10,
        }
    ],
    quality={
        "grade": "3",
        "protein_min": 11.5,
        "moisture_max": 14,
        "foreign_matter_max": 2,
        "lab": "buyer",
    },
    buyer='ТОВ "Зерно Поле 01"',
)


def test_value_outside_its_field_context_is_missing() -> None:
    deal = Deal(
        tolerance=Tolerance(pct=5, option="seller"),
        payment={"days": 5},
        quality={"grade": "3"},
    )
    text = "Продавець ФГ Колос. Пшениця, толеранс 5% на вибір покупця, вологість 13,5%."

    assert missing_values(deal, text) == ["tolerance.option", "payment.days", "quality.grade"]


def test_messenger_recap_with_every_field_passes() -> None:
    text = (
        "Пш 3 кл, 500т ±5% на вибір покупця. Ціна 9 200 грн/т з ПДВ, CPT порт Одеса, "
        "поставка 5–12 листопада. Оплата 5 б.д. після поставки, ПДВ після реєстрації ПН. "
        "За прострочення оплати покупець платить пеню 0,1% за кожен день, але не більше 10%. "
        "Протеїн від 11,5%, вологість до 14%, смітна домішка до 2%, лабораторія покупця. "
        "Покупець ТОВ «Зерно Поле 01»."
    )

    assert missing_values(FULL_DEAL, text) == []


def test_mixed_russian_and_english_wording_passes() -> None:
    text = (
        "Пшеница 3 класс, 500 тонн, tolerance +/-5% at buyer's option. "
        "Price 9200 UAH/t incl. VAT, CPT Одесса, поставка 5-12 ноября. "
        "Оплата 5 банковских дней после поставки, НДС после регистрации налоговой накладной. "
        "За просрочку оплаты покупатель платит 0,1% за каждый день, но не более 10%. "
        "Protein 11,5%, влажность 14%, сорная примесь 2%, лаборатория покупателя. "
        'Покупатель ТОВ "Зерно Поле 01".'
    )

    assert missing_values(FULL_DEAL, text) == []


def test_number_formats_and_crop_aliases_are_recognized() -> None:
    deal = Deal(crop="wheat", quantity_t=9200, buyer="ТОВ Зерно")

    assert missing_values(deal, "Пшениця, 9 200 тонн. ТОВ «Зерно».") == []


def test_unmentioned_field_is_reported() -> None:
    deal = Deal(crop="corn", buyer="ТОВ Зерно")

    assert missing_values(deal, "ТОВ «Зерно».") == ["crop"]


@pytest.mark.parametrize(
    "text",
    [
        "поставка 5–12 листопада",
        "з 5 по 12 листопада",
        "з 05.11 по 12.11",
        "05.11.2026–12.11.2026",
        "2026-11-05 — 2026-11-12",
        "5–12.11",
    ],
)
def test_delivery_dates_in_natural_formats_are_found(text: str) -> None:
    deal = Deal(delivery=Delivery(date_from=date(2026, 11, 5), date_to=date(2026, 11, 12)))

    assert missing_values(deal, text) == []


def test_inflected_place_name_is_found() -> None:
    deal = Deal(delivery=Delivery(place="Одеса"))

    assert missing_values(deal, "CPT порт в Одесі") == []


def test_vat_excluded_is_not_mistaken_for_vat_included() -> None:
    deal = Deal(price={"amount_per_t": 9200, "currency": "UAH", "vat_included": True})

    assert missing_values(deal, "9200 грн/т без ПДВ") == ["price.vat_included"]


def test_penalty_details_must_sit_next_to_their_trigger() -> None:
    deal = Deal(
        penalties=[
            Penalty(payer="seller", trigger="late_delivery", amount=0.5, unit="pct_per_day")
        ]
    )
    text = (
        "За прострочення поставки продавець відповідає згідно з договором. "
        "Поставка автотранспортом продавця на елеватор покупця, вивантаження протягом доби, "
        "документи — ТТН, сертифікат якості й посвідчення про походження зерна. "
        "За прострочення оплати покупець платить 0,5% за кожен день."
    )

    assert missing_values(deal, text) == ["penalties[0].amount", "penalties[0].unit"]


def test_double_nbu_rate_is_not_confused_with_single_rate() -> None:
    deal = Deal(
        penalties=[
            Penalty(payer="buyer", trigger="late_payment", amount=1, unit="nbu_rate_multiple")
        ]
    )
    text = "За прострочення оплати покупець сплачує пеню в розмірі подвійної облікової ставки НБУ."

    assert missing_values(deal, text) == ["penalties[0].amount"]


def test_other_enum_value_cannot_be_confirmed_by_text() -> None:
    deal = Deal(basis="other")

    assert missing_values(deal, "базис: інший базис") == ["basis"]


PRICE_UAH = {"amount_per_t": 9200, "currency": "UAH"}


@pytest.mark.parametrize(
    ("deal", "text", "expected"),
    [
        pytest.param(
            Deal(price={**PRICE_UAH, "vat_included": True}),
            "9200 грн/т, не включаючи ПДВ",
            ["price.vat_included"],
            id="vat-not-included",
        ),
        pytest.param(
            Deal(penalties=[Penalty(payer="seller", trigger="late_delivery", amount=5, unit="one_time_pct")]),
            "За прострочення поставки продавець сплачує штраф. Толеранс 5%.",
            ["penalties[0].amount"],
            id="penalty-amount-from-another-sentence",
        ),
        pytest.param(
            Deal(penalties=[Penalty(payer="seller", trigger="late_delivery", amount=0.5, unit="one_time_pct")]),
            "За прострочення поставки продавець сплачує штраф 0,5% за кожен день.",
            ["penalties[0].unit"],
            id="per-day-penalty-is-not-one-time",
        ),
        pytest.param(
            Deal(penalties=[Penalty(payer="seller", trigger="late_delivery", amount=0.5, unit="pct_per_day")]),
            "За прострочення поставки продавець відповідає за договором. "
            "За прострочення оплати покупець платить 0,5% за кожен день.",
            ["penalties[0].amount", "penalties[0].unit"],
            id="details-of-the-next-penalty-sentence",
        ),
        pytest.param(
            Deal(penalties=[Penalty(trigger="late_delivery", amount=0.1, unit="pct_per_day")]),
            "Пеня за прострочення поставки — за договором, а за прострочення оплати — 0,1% за кожен день.",
            ["penalties[0].amount", "penalties[0].unit"],
            id="details-of-another-penalty-in-one-sentence",
        ),
        pytest.param(
            Deal(quality={"moisture_max": 14}),
            "протеїн не менше 14%, вологість не більше 13,5%",
            ["quality.moisture_max"],
            id="protein-number-is-not-moisture",
        ),
        pytest.param(
            Deal(payment={"days": 5, "days_type": "calendar"}),
            "Оплата протягом 5 банківських днів, пеня за кожен календарний день.",
            ["payment.days_type"],
            id="days-type-belongs-to-the-days-number",
        ),
        pytest.param(
            Deal(payment={"days": 5, "trigger": "after_delivery"}),
            "Оплата після поставки. Поставка протягом 5 днів.",
            ["payment.days"],
            id="delivery-days-are-not-payment-days",
        ),
        pytest.param(
            Deal(delivery=Delivery(date_from=date(2026, 5, 14))),
            "вологість 14.5%",
            ["delivery.date_from"],
            id="percent-is-not-a-date",
        ),
        pytest.param(
            Deal(buyer="ТОВ Зерно Поле 01"),
            "ТОВ Зерно Поле 012",
            ["buyer"],
            id="buyer-name-must-match-whole-words",
        ),
    ],
)
def test_value_stated_for_another_field_is_missing(
    deal: Deal, text: str, expected: list[str]
) -> None:
    assert missing_values(deal, text) == expected


@pytest.mark.parametrize(
    ("deal", "text"),
    [
        pytest.param(Deal(tolerance=Tolerance(pct=5)), "толеранс 5 відсотків", id="percent-word"),
        pytest.param(Deal(price=PRICE_UAH), "Price 9,200 UAH/t", id="english-thousands"),
        pytest.param(
            Deal(price={**PRICE_UAH, "vat_included": False}), "9200 грн/т плюс ПДВ", id="plus-vat"
        ),
        pytest.param(
            Deal(payment={"trigger": "after_delivery"}), "оплата після розвантаження", id="unloading"
        ),
        pytest.param(
            Deal(payment={"trigger": "after_documents"}),
            "оплата після отримання всіх документів",
            id="all-documents",
        ),
        pytest.param(
            Deal(delivery=Delivery(date_to=date(2026, 11, 12))),
            "поставка до 12-го листопада",
            id="ordinal-day",
        ),
        pytest.param(
            Deal(delivery=Delivery(date_from=date(2026, 11, 5))), "відвантаження з 5/11", id="slash-date"
        ),
        pytest.param(
            Deal(penalties=[Penalty(payer="buyer", trigger="late_payment", amount=2, unit="nbu_rate_multiple")]),
            "За прострочення оплати покупець сплачує пеню в розмірі двох облікових ставок НБУ.",
            id="two-nbu-rates",
        ),
        pytest.param(
            Deal(penalties=[Penalty(payer="seller", trigger="late_delivery", amount=0.1, unit="pct_per_day")]),
            "За прострочення термінів поставки продавець платить 0,1% за кожен день.",
            id="delivery-terms",
        ),
        pytest.param(
            Deal(penalties=[Penalty(trigger="late_payment", amount=0.1, unit="pct_per_day", cap_pct=10)]),
            "За прострочення оплати пеня 0,1% за кожен день, але до 10% від суми.",
            id="cap-with-do",
        ),
        pytest.param(
            Deal(penalties=[Penalty(trigger="late_payment", amount=0.1, unit="pct_per_day", cap_pct=10)]),
            "За прострочення оплати пеня 0,1% за кожен день, 10% максимум.",
            id="cap-after-number",
        ),
        pytest.param(Deal(quality={"grade": "3"}), "пшениця третього класу", id="ordinal-grade"),
        pytest.param(
            Deal(tolerance=Tolerance(option="buyer")), "толеранс на розсуд покупця", id="at-discretion"
        ),
        pytest.param(
            Deal(price=PRICE_UAH), "беремо по 9200 за тонну, у гривнях", id="per-tonne-price"
        ),
    ],
)
def test_natural_phrasing_is_accepted(deal: Deal, text: str) -> None:
    assert missing_values(deal, text) == []
