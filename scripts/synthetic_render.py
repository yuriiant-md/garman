"""Spell a Deal as plain Ukrainian conditions for the LLM that writes synthetic texts.

The wording here must pass scripts/synthetic_checks.py: a test renders hundreds of
deals and runs the checker on the result, so the two modules cannot drift apart.
"""

import re
from datetime import date

from scripts.synthetic_checks import MONTH_NAMES
from src.schemas.deal import (
    FEED_GRADE,
    Crop,
    Currency,
    DaysType,
    Deal,
    Delivery,
    GradeTerms,
    IndicatorName,
    IndicatorUnit,
    Lab,
    Limit,
    Party,
    Payment,
    PaymentTrigger,
    Penalty,
    PenaltyTrigger,
    PenaltyUnit,
    Price,
    Quality,
    Tolerance,
)

MONTHS_GENITIVE = tuple(names[0] for names in MONTH_NAMES)
CROP_NAMES = {
    Crop.WHEAT: "пшениця",
    Crop.CORN: "кукурудза",
    Crop.BARLEY: "ячмінь",
    Crop.SUNFLOWER: "соняшник",
    Crop.SOY: "соя",
    Crop.RAPESEED: "ріпак",
}
PARTY_GENITIVE = {Party.BUYER: "покупця", Party.SELLER: "продавця"}
PARTY_NOMINATIVE = {Party.BUYER: "покупець", Party.SELLER: "продавець"}
CURRENCY_WORDS = {Currency.UAH: "грн", Currency.USD: "USD", Currency.EUR: "EUR"}
DAYS_TYPE_WORDS = {DaysType.CALENDAR: "календарних", DaysType.BANKING: "банківських"}
PAYMENT_TRIGGER_PHRASES = {
    PaymentTrigger.AFTER_DELIVERY: "після поставки",
    PaymentTrigger.AFTER_DOCUMENTS: "після отримання документів",
}
VAT_AFTER_REGISTRATION_LINES = {
    True: "ПДВ-частина ціни оплачується після реєстрації податкової накладної",
    False: "ПДВ оплачується незалежно від реєстрації податкової накладної",
}
PENALTY_EVENTS = {
    PenaltyTrigger.LATE_DELIVERY: "прострочення поставки",
    PenaltyTrigger.LATE_PAYMENT: "прострочення оплати",
    PenaltyTrigger.UNDER_DELIVERY: "недопоставку",
}
NBU_RATES = {1.0: "облікової ставки НБУ", 2.0: "подвійної облікової ставки НБУ"}
LAB_PHRASES = {
    Lab.BUYER: "лабораторії покупця",
    Lab.SELLER: "лабораторії продавця",
    Lab.INDEPENDENT: "незалежної лабораторії",
}
FIELD_LABELS = {
    "crop": "культура",
    "quantity_t": "обсяг у тоннах",
    "tolerance.pct": "толеранс у відсотках",
    "tolerance.option": "хто обирає толеранс",
    "price.currency": "валюта ціни",
    "price.vat_included": "ціна з ПДВ чи без",
    "basis": "базис поставки",
    "delivery.place": "місце поставки",
    "delivery.date_from": "дата початку поставки",
    "delivery.date_to": "дата кінця поставки",
    "delivery.period_raw": "строк поставки дослівно",
    "payment.days": "строк оплати в днях",
    "payment.days_type": "банківські чи календарні дні оплати",
    "payment.trigger": "від чого рахується строк оплати",
    "payment.vat_after_registration": "ПДВ після реєстрації податкової накладної",
    "quality.lab": "чия лабораторія",
    "grades.grade": "клас",
    "grades.amount_per_t": "ціна за тонну",
    "grades.indicators.protein": "протеїн",
    "grades.indicators.moisture": "вологість",
    "grades.indicators.foreign_matter": "сміттєва домішка",
    "buyer": "назва покупця",
}
# "grades[0].grade" has the label of "grades.grade"
GRADE_INDEX = re.compile(r"(?<=^grades)\[\d+\]")
INDICATOR_WORDS = {  # the indicators synthetic deals carry, as the checker looks for them
    IndicatorName.PROTEIN: "протеїн",
    IndicatorName.MOISTURE: "вологість",
    IndicatorName.FOREIGN_MATTER: "сміттєва домішка",
}
LIMIT_WORDS = {Limit.MIN: "не менше", Limit.MAX: "не більше"}
PENALTY_FIELD_LABELS = {
    "payer": "хто платить",
    "trigger": "за яке порушення",
    "amount": "розмір",
    "unit": "одиниця: % за день, ставка НБУ чи разовий штраф",
    "cap_pct": "обмеження «не більше»",
}
PENALTY_PATH = re.compile(r"penalties\[(\d+)\]\.(\w+)")


def render_conditions(deal: Deal) -> list[str]:
    """Return one Ukrainian line per filled group of Deal values."""
    lines = (
        f"культура: {CROP_NAMES[deal.crop]}" if deal.crop in CROP_NAMES else None,
        f"обсяг: {_num(deal.quantity_t)} т" if deal.quantity_t is not None else None,
        _tolerance_line(deal.tolerance),
        *_price_lines(deal.price, deal.grades or []),
        f"базис: {deal.basis.value}" if deal.basis else None,
        *_delivery_lines(deal.delivery),
        *_payment_lines(deal.payment),
        *(_penalty_line(penalty) for penalty in deal.penalties or []),
        _quality_line(deal.quality, deal.grades or []),
        f"покупець: {deal.buyer}" if deal.buyer else None,
    )
    return [line for line in lines if line]


def describe_missing(path: str) -> str:
    """Turn a checker path like "payment.days" into words the LLM understands."""
    penalty = PENALTY_PATH.fullmatch(path)
    if penalty:
        field = PENALTY_FIELD_LABELS.get(penalty[2], penalty[2])
        return f"неустойка №{int(penalty[1]) + 1}: {field} ({path})"
    label = FIELD_LABELS.get(GRADE_INDEX.sub("", path))
    return f"{label} ({path})" if label else path


def _num(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def _tolerance_line(tolerance: Tolerance | None) -> str | None:
    if tolerance is None:
        return None
    pct = f" ±{_num(tolerance.pct)}%" if tolerance.pct is not None else ""
    option = (
        f" на вибір {PARTY_GENITIVE[tolerance.option]}"
        if tolerance.option in PARTY_GENITIVE
        else ""
    )
    return f"толеранс{pct}{option}" if pct or option else None


def _price_lines(price: Price | None, grades: list[GradeTerms]) -> tuple[str, ...]:
    """One line per priced grade; the class is named only when there are several grades."""
    currency = CURRENCY_WORDS.get(price.currency, "") if price else ""
    vat = {True: " з ПДВ", False: " без ПДВ"}.get(price.vat_included, "") if price else ""
    return tuple(
        f"ціна{_grade_suffix(grade, len(grades))}: {_num(grade.amount_per_t)} {currency}/т{vat}"
        for grade in grades
        if grade.amount_per_t is not None
    )


def _grade_suffix(grade: GradeTerms, grades: int) -> str:
    phrase = _grade_phrase(grade.grade)
    return f" ({phrase})" if grades > 1 and phrase else ""


def _delivery_lines(delivery: Delivery | None) -> tuple[str | None, ...]:
    if delivery is None:
        return ()
    place = f"місце поставки: {delivery.place}" if delivery.place else None
    if delivery.period_raw:
        return place, f"строк поставки: {delivery.period_raw}"
    return place, _dates_line(delivery.date_from, delivery.date_to)


def _dates_line(start: date | None, end: date | None) -> str | None:
    if start and end and (start.month, start.year) == (end.month, end.year):
        return f"строк поставки: {start.day}–{end.day} {MONTHS_GENITIVE[start.month - 1]} {end.year}"
    if start and end:
        return f"строк поставки: {_day_month(start)} – {_day_month(end)} {end.year}"
    single = start or end
    return f"строк поставки: {_day_month(single)} {single.year}" if single else None


def _day_month(day: date) -> str:
    return f"{day.day} {MONTHS_GENITIVE[day.month - 1]}"


def _payment_lines(payment: Payment | None) -> tuple[str | None, ...]:
    if payment is None:
        return ()
    return _payment_terms_line(payment), VAT_AFTER_REGISTRATION_LINES.get(
        payment.vat_after_registration
    )


def _payment_terms_line(payment: Payment) -> str | None:
    period = _days_phrase(payment)
    if payment.trigger == PaymentTrigger.PREPAYMENT:
        terms = f"передоплата {period} з дати рахунку" if period else "передоплата"
    else:
        terms = " ".join(
            part for part in (period, PAYMENT_TRIGGER_PHRASES.get(payment.trigger)) if part
        )
    return f"оплата: {terms}" if terms else None


def _days_phrase(payment: Payment) -> str:
    if payment.days is None:
        return ""
    kind = DAYS_TYPE_WORDS.get(payment.days_type)
    return f"протягом {payment.days} {kind} днів" if kind else f"протягом {payment.days} днів"


def _penalty_line(penalty: Penalty) -> str:
    kind = "штраф" if penalty.unit == PenaltyUnit.ONE_TIME_PCT else "пеня"
    event = PENALTY_EVENTS.get(penalty.trigger, "порушення умов")
    payer = PARTY_NOMINATIVE.get(penalty.payer, "винна сторона")
    cap = f", але не більше {_num(penalty.cap_pct)}%" if penalty.cap_pct is not None else ""
    return f"{kind}: за {event} {payer} сплачує {_penalty_size(penalty)}{cap}"


def _penalty_size(penalty: Penalty) -> str:
    amount = penalty.amount or 0
    if penalty.unit == PenaltyUnit.NBU_RATE_MULTIPLE:
        rate = NBU_RATES.get(amount, f"{_num(amount)}-кратної облікової ставки НБУ")
        return f"пеню в розмірі {rate} за кожен день прострочення"
    if penalty.unit == PenaltyUnit.ONE_TIME_PCT:
        return f"штраф {_num(amount)}% від вартості партії"
    if penalty.unit == PenaltyUnit.PCT_PER_DAY:
        return f"{_num(amount)}% від суми за кожен день прострочення"
    return f"{_num(amount)}%"


def _quality_line(quality: Quality | None, grades: list[GradeTerms]) -> str | None:
    parts = (
        *(part for grade in grades for part in _grade_parts(grade)),
        f"за даними {LAB_PHRASES[quality.lab]}" if quality and quality.lab in LAB_PHRASES else None,
    )
    filled = [part for part in parts if part]
    return f"якість: {', '.join(filled)}" if filled else None


def _grade_parts(grade: GradeTerms) -> tuple[str | None, ...]:
    """The class, then each percent limit the checker knows: "протеїн не менше 11%"."""
    return _grade_phrase(grade.grade), *(
        _limit(f"{INDICATOR_WORDS[item.name]} {LIMIT_WORDS[item.limit]}", item.value)
        for item in grade.indicators or []
        if item.name in INDICATOR_WORDS and item.unit == IndicatorUnit.PCT
    )


def _grade_phrase(grade: str | None) -> str | None:
    if grade is None:
        return None
    return "фуражна" if grade == FEED_GRADE else f"{grade} клас"


def _limit(label: str, value: float | None) -> str | None:
    return f"{label} {_num(value)}%" if value is not None else None
