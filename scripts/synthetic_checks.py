"""Field-aware checks that a generated text states every value of its Deal.

A value counts as present only next to the words of its own field and in the same
sentence: the "5" in "толеранс 5%" confirms tolerance.pct, never payment.days, and a
penalty's details are searched only between its trigger and the next penalty's trigger.
Patterns cover Ukrainian, Russian and English wording because synthetic texts mix them.
"""

import bisect
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

from src.schemas.deal import (
    FEED_GRADE,
    Basis,
    Crop,
    Currency,
    DaysType,
    Deal,
    Lab,
    Party,
    PaymentTrigger,
    Penalty,
    PenaltyTrigger,
    PenaltyUnit,
)

FIELD_WINDOW = 40  # max characters between a field keyword and its number
QUALITY_WINDOW = 25  # quality indicators sit close together, so the window is tighter
PAYMENT_WINDOW = 60  # "оплата протягом 5 банківських днів" puts the number further away
PRICE_WINDOW = 25
CAP_WINDOW = 25  # max characters between "не більше" and the cap percentage

Span = tuple[int, int]

NUMBER = re.compile(r"(?<!\w)\d+(?: \d{3}(?!\d))*(?:[.,]\d+)?")
ENGLISH_THOUSANDS = re.compile(r"\d{1,3},\d{3}")  # "9,200" is 9.2 in Ukrainian, 9200 in English
SENTENCE_END = re.compile(r"\n|(?<!\b\w)[.!?;](?=\s)")  # "б.д." and "т." do not end a sentence
PERCENT = r"\s*(?:%|відсотк\w*|процент\w*|percent)"
TONNE = r"\s*(?:тонн\w*|тн\b|т\b|т\.|mt\b|t\b|tons?\b)"
DAY_COUNT = r"\s*(?:-?\s*(?:х|ти|ми)\b)?\s*"
DAY = (
    DAY_COUNT
    + r"(?:банківськ\w*|календарн\w*|робоч\w*|банковск\w*|рабоч\w*|bank\w*|calendar|business|working)?"
    + r"\s*(?:дн\w*|день|доб\w*|сут\w*|days?\b|б\.\s?д\.?|к\.\s?д\.?|б/д|к/д)"
)
NOT_AFTER_LETTER = r"(?<![^\W\d_])"
BUYER_WORD = r"покуп"
SELLER_WORD = r"(?:продав|постачальн|поставщик)"
CHOICE_WORD = r"(?:вибір|вибором|розсуд\w*|опціон\w*|выбор\w*|опцион\w*|усмотрени\w*)"
CLASS_WORD = r"(?:кл\b|кл\.|клас\w*|класс\w*|class\b|grade\b)"
TERM_WORD = r"(?:(?:термін\w*|строк\w*|срок\w*)\s+)?"
ROMAN_BY_DIGIT = {"1": "i", "2": "ii", "3": "iii", "4": "iv", "5": "v", "6": "vi"}
ORDINAL_GRADES = {
    "1": ("перш", "перв", "first"),
    "2": ("друг", "втор", "second"),
    "3": ("трет", "third"),
    "4": ("четверт", "четвёрт", "fourth"),
    "5": ("п['’ʼ]?ят", "fifth"),
    "6": ("шост", "шест", "sixth"),
}

CROP_PATTERNS = {
    Crop.WHEAT: (r"пшени", r"\bпш\b", r"\bwheat"),
    Crop.CORN: (r"кукуру", r"\bкук\b", r"\bcorn\b", r"maize"),
    Crop.BARLEY: (r"ячм[іе]н", r"\bbarley"),
    Crop.SUNFLOWER: (r"соняшник", r"подсолнеч", r"sunflower"),
    Crop.SOY: (r"\bсо[яїюи]\b", r"\bsoy"),
    Crop.RAPESEED: (r"ріпак", r"рапс", r"rapeseed", r"canola"),
}
BASIS_PATTERNS = {
    Basis.EXW: (r"\bexw\b",),
    Basis.FCA: (r"\bfca\b",),
    Basis.CPT: (r"\bcpt\b", r"\bсрт\b", r"\bспт\b", r"\bсpt\b"),
    Basis.DAP: (r"\bdap\b",),
    Basis.FOB: (r"\bfob\b", r"\bфоб\b"),
}
CURRENCY_PATTERNS = {
    Currency.UAH: (r"грн", r"гривн", r"₴", r"\buah\b"),
    Currency.USD: (r"\$", r"\busd\b", r"долар", r"доллар"),
    Currency.EUR: (r"€", r"\beur\b", r"євро", r"евро"),
}
VAT_INCLUDED_PATTERNS = {
    True: (
        rf"{NOT_AFTER_LETTER}з\s+пдв",
        rf"{NOT_AFTER_LETTER}с\s+ндс",
        r"(?<!не )\bвкл\w*\.?\s+(?:з\s+)?(?:пдв|ндс)",
        r"(?:пдв|ндс)\s+включ",
        rf"{NOT_AFTER_LETTER}з\s+урахуванням\s+пдв",
        r"incl\w*\.?\s+vat",
        r"vat\s+incl",
        r"with\s+vat",
    ),
    False: (
        r"без\s+(?:пдв|ндс)",
        r"без\s+(?:урахування|врахування)\s+пдв",
        r"не\s+включаючи\s+(?:пдв|ндс)",
        r"(?:пдв|ндс)\s+не\s+включ",
        r"(?:плюс|\+)\s*(?:пдв|ндс)",
        r"excl\w*\.?\s+vat",
        r"vat\s+excl",
        r"without\s+vat",
        r"net\s+of\s+vat",
        r"(?:plus|\+)\s*vat",
    ),
}
PLACE_PATTERNS = {
    "Одеса": (r"одес", r"odes"),
    "Миколаїв": (r"микола[їєи]в", r"николаев", r"mykolai"),
    "Полтава": (r"полтав", r"poltav"),
    "Черкаси": (r"черкас", r"cherkas"),
}
TOLERANCE_WORDS = (r"толер", r"tolerance", r"±", r"\+\s*/?\s*-", r"опці", r"опци", r"option")
OPTION_PATTERNS = {
    Party.BUYER: (
        rf"{CHOICE_WORD}\s+{BUYER_WORD}",
        r"buyer['’]?s\s+option",
        r"option\s+of\s+(?:the\s+)?buyer",
    ),
    Party.SELLER: (
        rf"{CHOICE_WORD}\s+{SELLER_WORD}",
        r"seller['’]?s\s+option",
        r"option\s+of\s+(?:the\s+)?seller",
    ),
}
PRICE_WORDS = (
    r"грн",
    r"₴",
    r"\buah\b",
    r"гривн",
    r"\bцін",
    r"\bцен",
    r"\bprice",
    r"/\s*т\b",
    r"/\s*тонн",
    r"за\s+(?:тонн|т\b)",
    r"per\s+(?:ton|t)\b",
    r"\b(?:cpt|fca|exw|dap|fob)\b",
)
PAYMENT_WORDS = (
    r"оплат",
    r"розрахун",
    r"платіж",
    r"платеж",
    r"сплат",
    r"передоплат",
    r"предоплат",
    r"аванс",
    r"\bpay",
    r"расч[её]т",
)
DAYS_TYPE_PATTERNS = {
    DaysType.CALENDAR: (
        r"календарн\w*\s+(?:дн|день)",
        r"\bк\.\s?д\.?",
        r"\bк/д\b",
        r"calendar\s+days?",
    ),
    DaysType.BANKING: (
        r"банківськ\w*\s+(?:дн|день)",
        r"банковск\w*\s+(?:дн|день)",
        r"\bб\.\s?д\.?",
        r"\bб/д\b",
        r"bank(?:ing)?\s+days?",
    ),
}
# The day type must be attached to the number of days: "5 календарних днів".
DAYS_TYPE_SUFFIXES = {
    days_type: DAY_COUNT + "(?:" + "|".join(patterns) + ")"
    for days_type, patterns in DAYS_TYPE_PATTERNS.items()
}
PAYMENT_TRIGGER_PATTERNS = {
    PaymentTrigger.PREPAYMENT: (
        r"передоплат",
        r"предоплат",
        r"\bаванс",
        r"наперед",
        r"prepay",
        r"in\s+advance",
        r"advance\s+payment",
    ),
    PaymentTrigger.AFTER_DELIVERY: (
        (
            r"після\s+(?:поставки|доставки|відвантаження|вивантаження|розвантаження|приймання"
            r"|отримання\s+(?:товару|зерна))"
        ),
        r"по\s+факту\s+(?:поставки|відвантаження|вивантаження|розвантаження|отгрузки)",
        r"после\s+(?:поставки|отгрузки|доставки|выгрузки|разгрузки)",
        r"after\s+(?:delivery|unloading|shipment)",
        r"upon\s+delivery",
    ),
    PaymentTrigger.AFTER_DOCUMENTS: (
        r"після\s+(?:отримання\s+|надання\s+)?(?:всіх\s+|усіх\s+)?(?:оригіналів\s+)?(?:документ|ттн)",
        r"после\s+(?:получения\s+)?(?:всех\s+)?(?:оригиналов\s+)?документ",
        r"after\s+(?:receipt\s+of\s+)?(?:all\s+)?(?:original\s+)?documents",
        r"against\s+documents",
    ),
}
VAT_AFTER_REGISTRATION_PATTERNS = {
    True: (
        r"після\s+реєстрац\w*\s+(?:податков|пн\b|рпн)",
        r"после\s+регистрац\w*\s+(?:налогов|нн\b)",
        r"after\s+(?:the\s+)?(?:tax|vat)\s+invoice\s+(?:is\s+)?registered",
        r"after\s+registration\s+of\s+(?:the\s+)?(?:tax|vat)\s+invoice",
    ),
    False: (
        r"незалежно\s+від\s+реєстрац",
        r"независимо\s+от\s+регистрац",
        r"regardless\s+of\s+(?:the\s+)?(?:tax|vat)\s+invoice",
    ),
}
PROTEIN_WORDS = (r"проте[їи]н", r"біл(?:ок|ка)", r"бел(?:ок|ка)", r"protein")
MOISTURE_WORDS = (r"волог", r"влажн", r"moisture", r"humidity")
FOREIGN_MATTER_WORDS = (
    r"смітт",
    r"смітн",
    r"домішк",
    r"засміч",
    r"сорн",
    r"примес",
    r"foreign",
    r"impurit",
)
QUALITY_WORDS = {
    "quality.protein_min": PROTEIN_WORDS,
    "quality.moisture_max": MOISTURE_WORDS,
    "quality.foreign_matter_max": FOREIGN_MATTER_WORDS,
}
LAB_PATTERNS = {
    Lab.BUYER: (rf"лаб\w*\.?\s+{BUYER_WORD}", r"buyer['’]?s\s+lab"),
    Lab.SELLER: (rf"лаб\w*\.?\s+{SELLER_WORD}", r"seller['’]?s\s+lab"),
    Lab.INDEPENDENT: (
        r"незалежн\w*\s+лаб",
        r"независим\w*\s+лаб",
        r"independent\s+lab",
        r"сюрве",
        r"surveyor",
    ),
}
PAYER_PATTERNS = {
    Party.BUYER: (BUYER_WORD, r"\bbuyer"),
    Party.SELLER: (SELLER_WORD, r"\bseller", r"\bsupplier"),
}
PENALTY_TRIGGER_PATTERNS = {
    PenaltyTrigger.LATE_DELIVERY: (
        rf"простроч\w*\s+(?:у\s+)?{TERM_WORD}(?:постав|постача|відвантаж|доставк)",
        rf"затримк\w*\s+{TERM_WORD}(?:постав|відвантаж|доставк)",
        r"несвоєчасн\w*\s+(?:постав|відвантаж|постача)",
        rf"просроч\w*\s+{TERM_WORD}(?:постав|отгрузк)",
        r"late\s+delivery",
        r"delay\w*\s+(?:in\s+)?delivery",
    ),
    PenaltyTrigger.LATE_PAYMENT: (
        rf"простроч\w*\s+(?:у\s+)?{TERM_WORD}(?:оплат|платеж|платіж|розрахун|сплат)",
        rf"затримк\w*\s+{TERM_WORD}(?:оплат|платеж|платіж|розрахун)",
        r"несвоєчасн\w*\s+(?:оплат|розрахун|сплат)",
        rf"просроч\w*\s+{TERM_WORD}(?:оплат|платеж)",
        r"late\s+payment",
        r"delay\w*\s+(?:in\s+)?payment",
    ),
    PenaltyTrigger.UNDER_DELIVERY: (
        r"недопостав",
        r"непостав",
        r"недовантаж",
        r"недовоз",
        r"under-?\s*deliver",
        r"non-?\s*deliver",
        r"short\s+delivery",
    ),
}
PENALTY_WORDS = (r"пен[яюіи]", r"штраф", r"неустойк", r"penalt", r"\bfine\b")
PER_DAY_PATTERNS = (
    r"за\s+(?:кожен\s+|кожний\s+|кожного\s+|каждый\s+)?(?:календарний\s+)?(?:день|добу|дня)",
    r"(?:в|на)\s+день",
    r"/\s*(?:день|добу|day)",
    r"щоденно",
    r"в\s+сутки",
    r"per\s+day",
    r"a\s+day",
    r"daily",
)
NBU_PATTERNS = (r"облік\w*\s+ставк", r"учетн\w*\s+ставк", r"ставк\w*\s+нбу", r"\bнбу\b", r"\bnbu\b")
DOUBLE_PATTERNS = (
    r"подвійн",
    r"двократн",
    r"\b2\s*-?\s*(?:х|кратн)",
    r"двох\s+(?:облік|учетн)",
    r"двойн",
    r"double",
    r"twice",
)
ONE_TIME_PATTERNS = (r"штраф", r"одноразов", r"разов", r"\bfine\b", r"one-?\s*time")
UNIT_PATTERNS = {
    PenaltyUnit.PCT_PER_DAY: PER_DAY_PATTERNS,
    PenaltyUnit.NBU_RATE_MULTIPLE: NBU_PATTERNS,
    PenaltyUnit.ONE_TIME_PCT: ONE_TIME_PATTERNS,
}
CAP_PATTERNS = (
    r"не\s+більш",
    r"не\s+вище",
    r"не\s+бол",
    r"максимум",
    r"\bмакс\b",
    r"\bдо\b",
    r"\bmax",
    r"\bcap\b",
    r"not\s+(?:more|exceed)",
    r"up\s+to",
)
POST_CAP_PATTERNS = (r"максимум", r"\bмакс\b", r"\bmax(?:imum)?\b")  # "10% максимум"

MONTH_NAMES = (
    ("січня", "января", "january", "jan"),
    ("лютого", "февраля", "february", "feb"),
    ("березня", "марта", "march", "mar"),
    ("квітня", "апреля", "april", "apr"),
    ("травня", "мая", "may"),
    ("червня", "июня", "june", "jun"),
    ("липня", "июля", "july", "jul"),
    ("серпня", "августа", "august", "aug"),
    ("вересня", "сентября", "september", "sept", "sep"),
    ("жовтня", "октября", "october", "oct"),
    ("листопада", "ноября", "november", "nov"),
    ("грудня", "декабря", "december", "dec"),
)
MONTHS = {name: number for number, names in enumerate(MONTH_NAMES, start=1) for name in names}
MONTH = "(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b"
ORDINAL_SUFFIX = r"(?:-?го)?"
DAY_RANGE = rf"(?:\s*(?:-|–|—|до|по)\s*(\d{{1,2}}){ORDINAL_SUFFIX})?"
WORD_DATE = re.compile(rf"(?<!\d)(\d{{1,2}}){ORDINAL_SUFFIX}{DAY_RANGE}\s+{MONTH}")
MONTH_FIRST_DATE = re.compile(rf"\b{MONTH}\s+(\d{{1,2}}){DAY_RANGE}(?!\d)")
NUMERIC_DATE = re.compile(
    rf"(?<![\d.,/])(\d{{1,2}}){DAY_RANGE}[./](\d{{1,2}})(?:[./](\d{{2,4}}))?(?![\d.,])(?!\s*%)"
)
ISO_DATE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")


@dataclass(frozen=True)
class Number:
    value: Decimal
    span: Span


@dataclass(frozen=True)
class ParsedText:
    """Casefolded text with spaces collapsed, newlines kept, numbers and sentence ends located."""

    text: str
    numbers: tuple[Number, ...]
    sentence_ends: tuple[int, ...]

    @classmethod
    def parse(cls, raw: str) -> "ParsedText":
        lines = re.sub(r"\s*\n\s*", "\n", raw.casefold().strip())
        text = re.sub(r"[^\S\n]+", " ", lines)
        ends = tuple(match.start() for match in SENTENCE_END.finditer(text))
        return cls(text, tuple(_parse_numbers(text)), ends)

    def has(self, patterns: Iterable[str]) -> bool:
        return any(re.search(pattern, self.text) for pattern in patterns)

    def spans(self, patterns: Iterable[str]) -> list[Span]:
        return [match.span() for pattern in patterns for match in re.finditer(pattern, self.text)]

    def numbers_equal(self, value: object, suffix: str = "") -> list[Span]:
        target = Decimal(str(value))
        return [
            number.span
            for number in self.numbers
            if number.value == target
            and (not suffix or re.compile(suffix).match(self.text, number.span[1]))
        ]

    def sentence(self, position: int) -> Span:
        index = bisect.bisect_left(self.sentence_ends, position)
        start = self.sentence_ends[index - 1] + 1 if index else 0
        end = self.sentence_ends[index] if index < len(self.sentence_ends) else len(self.text)
        return start, end

    def close(self, spans: list[Span], anchors: list[Span], window: int) -> bool:
        """True when some span lies within window characters of an anchor in the same sentence."""
        return any(
            _gap(span, anchor) <= window and self.sentence(span[0]) == self.sentence(anchor[0])
            for span in spans
            for anchor in anchors
        )

    def region(self, anchor: Span, rivals: list[Span]) -> Span:
        """The anchor's sentence, cut at the nearest rival anchors on either side."""
        start, end = self.sentence(anchor[0])
        before = [rival[1] for rival in rivals if rival[1] <= anchor[0]]
        after = [rival[0] for rival in rivals if rival[0] >= anchor[1]]
        return max([start, *before]), min([end, *after])


Check = Callable[[ParsedText, object, dict[str, object]], bool]


def missing_values(deal: Deal, text: str) -> list[str]:
    """Return paths of filled Deal values that the text does not state in their field's context."""
    parsed = ParsedText.parse(text)
    return [path for path, present in _check_fields(deal, parsed) if not present]


def _check_fields(deal: Deal, text: ParsedText) -> Iterator[tuple[str, bool]]:
    for name, value in deal.model_dump(exclude_none=True).items():
        if name == "penalties":
            for index, penalty in enumerate(deal.penalties or []):
                for field, present in _check_penalty(penalty, text):
                    yield f"penalties[{index}].{field}", present
        elif isinstance(value, dict):
            for field, child in value.items():
                yield f"{name}.{field}", _check_value(f"{name}.{field}", child, text, value)
        else:
            yield name, _check_value(name, value, text, {})


def _check_value(path: str, value: object, text: ParsedText, group: dict[str, object]) -> bool:
    check = FIELD_CHECKS.get(path)
    return check is not None and check(text, value, group)


def _check_penalty(penalty: Penalty, text: ParsedText) -> Iterator[tuple[str, bool]]:
    own = PENALTY_TRIGGER_PATTERNS.get(penalty.trigger, ()) if penalty.trigger else PENALTY_WORDS
    rivals = text.spans(
        pattern
        for trigger, patterns in PENALTY_TRIGGER_PATTERNS.items()
        if trigger != penalty.trigger
        for pattern in patterns
    )
    regions = [text.region(anchor, rivals) for anchor in text.spans(own)]
    checks: dict[str, Callable[[], bool]] = {
        "payer": lambda: _inside(text.spans(PAYER_PATTERNS.get(penalty.payer, ())), regions),
        "trigger": lambda: bool(regions),
        "amount": lambda: _penalty_amount_present(penalty, text, regions),
        "unit": lambda: _penalty_unit_present(penalty.unit, text, regions),
        "cap_pct": lambda: _cap_present(penalty.cap_pct, text, regions),
    }
    for field in penalty.model_dump(exclude_none=True):
        yield field, checks[field]()


def _penalty_amount_present(penalty: Penalty, text: ParsedText, regions: list[Span]) -> bool:
    if penalty.unit != PenaltyUnit.NBU_RATE_MULTIPLE:
        return _inside(text.numbers_equal(penalty.amount, PERCENT), regions)
    doubled = _inside(text.spans(DOUBLE_PATTERNS), regions)
    if penalty.amount == 2:
        return doubled
    if penalty.amount == 1:
        return not doubled and _inside(text.spans(NBU_PATTERNS), regions)
    return _inside(text.numbers_equal(penalty.amount), regions)


def _penalty_unit_present(unit: PenaltyUnit | None, text: ParsedText, regions: list[Span]) -> bool:
    if unit != PenaltyUnit.ONE_TIME_PCT:
        return _inside(text.spans(UNIT_PATTERNS.get(unit, ())), regions)
    one_time, per_day = text.spans(ONE_TIME_PATTERNS), text.spans(PER_DAY_PATTERNS)
    return any(
        _inside(one_time, [region]) and not _inside(per_day, [region]) for region in regions
    )


def _cap_present(cap: object, text: ParsedText, regions: list[Span]) -> bool:
    percents = [span for span in text.numbers_equal(cap, PERCENT) if _inside([span], regions)]
    word_first = any(
        0 <= number[0] - word[1] <= CAP_WINDOW
        for word in text.spans(CAP_PATTERNS)
        for number in percents
    )
    number_first = any(
        0 <= word[0] - number[1] <= CAP_WINDOW
        for word in text.spans(POST_CAP_PATTERNS)
        for number in percents
    )
    return word_first or number_first


def _inside(spans: list[Span], regions: list[Span]) -> bool:
    return any(start <= span[0] and span[1] <= end for span in spans for start, end in regions)


def _gap(first: Span, second: Span) -> int:
    return max(0, second[0] - first[1], first[0] - second[1])


def _parse_numbers(text: str) -> Iterator[Number]:
    for match in NUMBER.finditer(text):
        raw = match.group().replace(" ", "")
        readings = [raw.replace(",", ".")]
        if ENGLISH_THOUSANDS.fullmatch(raw):
            readings.append(raw.replace(",", ""))
        for reading in readings:
            try:
                yield Number(Decimal(reading), match.span())
            except InvalidOperation:
                continue


def _phrase(table: dict[object, tuple[str, ...]]) -> Check:
    def check(text: ParsedText, value: object, _group: dict[str, object]) -> bool:
        return text.has(table.get(value, ()))

    return check


def _number_with(suffix: str) -> Check:
    def check(text: ParsedText, value: object, _group: dict[str, object]) -> bool:
        return bool(text.numbers_equal(value, suffix))

    return check


def _number_near(keywords: tuple[str, ...], window: int = FIELD_WINDOW, suffix: str = "") -> Check:
    def check(text: ParsedText, value: object, _group: dict[str, object]) -> bool:
        return text.close(text.numbers_equal(value, suffix), text.spans(keywords), window)

    return check


def _quality_number(path: str) -> Check:
    """The number must follow its own keyword with no other quality keyword in between."""
    rivals = tuple(word for other, words in QUALITY_WORDS.items() if other != path for word in words)

    def check(text: ParsedText, value: object, _group: dict[str, object]) -> bool:
        rival_spans = text.spans(rivals)
        return any(
            0 <= number[0] - anchor[1] <= QUALITY_WINDOW
            and text.sentence(number[0]) == text.sentence(anchor[0])
            and not any(anchor[1] <= rival[0] < number[0] for rival in rival_spans)
            for anchor in text.spans(QUALITY_WORDS[path])
            for number in text.numbers_equal(value)
        )

    return check


def _days_type(text: ParsedText, value: object, group: dict[str, object]) -> bool:
    payment_words = text.spans(PAYMENT_WORDS)
    days = group.get("days")
    if days is None:
        found = text.spans(DAYS_TYPE_PATTERNS.get(value, ()))
    else:
        found = text.numbers_equal(days, DAYS_TYPE_SUFFIXES.get(value, r"(?!)"))
    return text.close(found, payment_words, PAYMENT_WINDOW)


def _place(text: ParsedText, value: object, _group: dict[str, object]) -> bool:
    place = str(value)
    return text.has(PLACE_PATTERNS.get(place, (re.escape(place.casefold()),)))


def _verbatim(text: ParsedText, value: object, _group: dict[str, object]) -> bool:
    return " ".join(str(value).casefold().split()) in " ".join(text.text.split())


def _name(text: ParsedText, value: object, _group: dict[str, object]) -> bool:
    """Match a company name word by word, ignoring quotes and punctuation between words."""
    words = re.findall(r"\w+", str(value).casefold())
    pattern = r"(?<!\w)" + r"[\W_]*".join(map(re.escape, words)) + r"(?!\w)"
    return bool(words) and re.search(pattern, text.text) is not None


def _grade(text: ParsedText, value: object, _group: dict[str, object]) -> bool:
    return text.has(_grade_patterns(str(value)))


def _grade_patterns(grade: str) -> tuple[str, ...]:
    if grade == FEED_GRADE:
        return (r"фураж", r"\bfeed\b")
    if not grade.isdigit():
        return (re.escape(grade),)
    patterns = (
        rf"(?<!\d)(?<!\d[.,]){grade}\s*-?\s*(?:го|ий|ій|й)?\s*{CLASS_WORD}",
        rf"{CLASS_WORD}\s*{grade}(?!\d|[.,]\d)",
    )
    roman = ROMAN_BY_DIGIT.get(grade)
    ordinals = ORDINAL_GRADES.get(grade)
    return (
        patterns
        + ((rf"\b{roman}\s*{CLASS_WORD}",) if roman else ())
        + ((rf"\b(?:{'|'.join(ordinals)})\w*\s+{CLASS_WORD}",) if ordinals else ())
    )


def _date(text: ParsedText, value: object, _group: dict[str, object]) -> bool:
    return isinstance(value, date) and value in _dates(text.text, value.year)


def _dates(text: str, default_year: int) -> set[date]:
    candidates = _date_candidates(text, default_year)
    return {found for found in (_safe_date(*parts) for parts in candidates) if found}


def _date_candidates(text: str, year: int) -> Iterator[tuple[int, int, int]]:
    for match in ISO_DATE.finditer(text):
        yield int(match[1]), int(match[2]), int(match[3])
    for match in WORD_DATE.finditer(text):
        yield from ((year, MONTHS[match[3]], int(day)) for day in (match[1], match[2]) if day)
    for match in MONTH_FIRST_DATE.finditer(text):
        yield from ((year, MONTHS[match[1]], int(day)) for day in (match[2], match[3]) if day)
    for match in NUMERIC_DATE.finditer(text):
        found_year = _full_year(match[4]) if match[4] else year
        yield from ((found_year, int(match[3]), int(day)) for day in (match[1], match[2]) if day)


def _full_year(digits: str) -> int:
    return 2000 + int(digits) if len(digits) == 2 else int(digits)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


FIELD_CHECKS: dict[str, Check] = {
    "crop": _phrase(CROP_PATTERNS),
    "quantity_t": _number_with(TONNE),
    "tolerance.pct": _number_near(TOLERANCE_WORDS, suffix=PERCENT),
    "tolerance.option": _phrase(OPTION_PATTERNS),
    "price.amount_per_t": _number_near(PRICE_WORDS, window=PRICE_WINDOW),
    "price.currency": _phrase(CURRENCY_PATTERNS),
    "price.vat_included": _phrase(VAT_INCLUDED_PATTERNS),
    "basis": _phrase(BASIS_PATTERNS),
    "delivery.place": _place,
    "delivery.date_from": _date,
    "delivery.date_to": _date,
    "delivery.period_raw": _verbatim,
    "payment.days": _number_near(PAYMENT_WORDS, window=PAYMENT_WINDOW, suffix=DAY),
    "payment.days_type": _days_type,
    "payment.trigger": _phrase(PAYMENT_TRIGGER_PATTERNS),
    "payment.vat_after_registration": _phrase(VAT_AFTER_REGISTRATION_PATTERNS),
    "quality.grade": _grade,
    "quality.protein_min": _quality_number("quality.protein_min"),
    "quality.moisture_max": _quality_number("quality.moisture_max"),
    "quality.foreign_matter_max": _quality_number("quality.foreign_matter_max"),
    "quality.lab": _phrase(LAB_PATTERNS),
    "buyer": _name,
}
