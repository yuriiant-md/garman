"""Hide personal data in contract, law and court decision texts before they reach git or a model.

Phones, IBANs, bank accounts, personal tax numbers (РНОКПП), emails and people's names become
tags like [ТЕЛЕФОН] or [ПІБ]. Company names and their 8-digit ЄДРПОУ codes are public and stay,
unless `company_codes=True` (the author's private contracts). A person reviews what was masked:
names the rules missed go to extra_names.txt next to the raw file, and the text is masked again."""

import re
from collections.abc import Sequence
from pathlib import Path

from src.documents.pdf import clean_text

EXTRA_NAMES_NAME = "extra_names.txt"
EXTRA_ADDRESSES_NAME = "extra_addresses.txt"
EXTRA_HIDDEN_NAME = "extra_hidden.txt"
PAGE_BREAK = "⟦page⟧"  # a line no masking rule touches: parts are masked as one text

UPPER = "А-ЯІЇЄҐ"
LOWER = "а-яіїєґ'’"
NAME_WORD = rf"[{UPPER}](?:[{LOWER}]+|[{UPPER}'’]+)"  # "Петренко" or "ПЕТРЕНКО"
INITIALS = rf"[{UPPER}A-Z]\.\s?[{UPPER}A-Z]\."  # OCR sometimes reads "О.І." as Latin "O.I."
MIN_STEM = 4
MIN_PATRONYMIC = 7  # "Іванович", "Петрівна"; shorter look-alikes are words like "Рівні"

# Whole words only: "автомобіля 10 000 кг" and "факсимільного 01.01.2024" are not phones.
PHONE_CUE = r"\b(?:тел|телефон\w*|факс|моб|мобільн\w*|phone)\b"

# Order matters: an IBAN contains a phone-like run of digits, a phone with 38 is 12 digits
# like a VAT number, and a phone without it is 10 digits like a personal tax number.
CODE_PATTERNS = (
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    (  # OCR loses or garbles "UA" ("ІІА", "СІА"), so the country letters are optional
        "IBAN",
        re.compile(rf"(?<!\d)(?:[A-Za-z{UPPER}{LOWER}]{{2}} ?)?\d{{2}}(?: ?\d){{25}}(?!\d)"),
    ),
    ("РАХУНОК", re.compile(r"(?<!\d)\d{14,20}(?!\d)")),
    ("РАХУНОК", re.compile(r"(?<=[рР]/[рР] )\d[\d ]{4,}\d")),  # pre-IBAN accounts are shorter
    (  # after "тел."/"факс" any digit run is a phone: "(04137) 5-4.3-08", "2-31-98, 2-16-98"
        "ТЕЛЕФОН",
        re.compile(rf"{PHONE_CUE}[^\d\n]{{0,12}}?(?P<value>[+(]?\d[\d ().\-/,]{{4,}}\d)", re.I),
    ),
    (
        "ТЕЛЕФОН",
        re.compile(
            r"(?<![\d+])(?:\+?38[\s-]?)?(?:\(0\d{2,4}\)|0\d{2})"
            r"[\s.\-/]?\d{3}[\s.\-/]?\d{2}[\s.\-/]?\d{2}(?!\d)"
        ),
    ),
    ("ІПН", re.compile(r"(?<!\d)\d{12}(?!\d)")),
    ("РНОКПП", re.compile(r"(?<!\d)(?<!№)(?<!№ )[1-9]\d{9}(?!\d)")),  # "№ 2024100012" is a contract
)
# ЄДРПОУ of a party or of the elevator; "12500000,00 грн" is a sum and "№ 20260484" a contract
COMPANY_CODE = re.compile(r"(?<!\d)(?<!№)(?<!№ )\d{8}(?!\d)(?![.,]\d)")
# Nominative for signatures, genitive for "в особі директора Василя ...". Names that are also
# common words (Надія, Віра, Любов) are left out.
FIRST_NAMES = tuple(
    """
    Олександр Олександра Олексій Олексія Іван Івана Петро Петра Микола Миколи Василь Василя
    Віктор Віктора Володимир Володимира Сергій Сергія Андрій Андрія Юрій Юрія Михайло Михайла
    Дмитро Дмитра Віталій Віталія Анатолій Анатолія Григорій Григорія Леонід Леоніда Олег Олега
    Ігор Ігоря Роман Романа Тарас Тараса Богдан Богдана Ярослав Ярослава Валерій Валерія
    Валентин Валентина Степан Степана Павло Павла Максим Максима Євген Євгена Вадим Вадима
    Руслан Руслана Артем Артема Денис Дениса Костянтин Костянтина Станіслав Станіслава Федір
    Федора Борис Бориса Геннадій Геннадія
    Олена Олени Ольга Ольги Наталія Наталії Тетяна Тетяни Світлана Світлани Людмила Людмили
    Галина Галини Ірина Ірини Оксана Оксани Марія Марії Юлія Юлії Катерина Катерини Анна Анни
    Ганна Ганни Вікторія Вікторії Лариса Лариси Інна Інни Валентини Лідія Лідії Ніна Ніни
    Тамара Тамари Яна Яни Аліна Аліни
""".split()
)
FIRST_NAME = re.compile(rf"\b(?:{'|'.join(FIRST_NAMES)})\b", re.IGNORECASE)
PATRONYMIC = re.compile(rf"[{UPPER}{LOWER}]+(?:ович|евич|івн|ївн)(?:а|у|ем|і|и|ою)?", re.IGNORECASE)
WORD = re.compile(rf"[{UPPER}{LOWER}]+")
TWO_NAMES_BEFORE = re.compile(rf"(?:({NAME_WORD})\s+)?({NAME_WORD})\s+$")
NAME_BEFORE = re.compile(rf"({NAME_WORD})\s+$")
NAME_AFTER = re.compile(rf"\s+({NAME_WORD})\b")
INITIALS_THEN_SURNAME = re.compile(rf"{INITIALS}\s?({NAME_WORD})")
SURNAME_THEN_INITIALS = re.compile(rf"\b({NAME_WORD})\s+{INITIALS}")
# A name right after these words belongs to an institution or a street, not to a signatory:
# "імені Л.П. Симиренка" (OCR also gives "iMCHi", "iivieni"), "вул. Петра Сагайдачного".
# A letter after the "і": initials "І.В." of the previous name in a list are not "ім.".
NAMED_AFTER = re.compile(
    r"(?:\b[іiІI1][^\W\d_]\S{0,5}|вул\.|вулиця|просп\.|проспект|пров\.|провулок|бульв\.|бульвар|пл\.|площа)\s*$",
    re.IGNORECASE,
)
CASE_ENDINGS = ("ого", "ому", "ою", "ої", "ей", "ем", "а", "я", "у", "ю", "і", "и", "о", "е")
MAX_ENDING = 4  # a surname form is its stem plus a case ending, not a longer word
MAX_ABBREVIATION = 3  # "ТОВ", "ДП", "ФГ" after a name are legal forms, not surnames
# A vowel drops in some surnames: "Кравець" → "Кравця", "Квасок" → "Кваска".
FLEETING_VOWELS = (("ець", "ц"), ("ок", "к"))
NOT_NAMES = frozenset(
    """
    директор начальник голова керівник заступник бухгалтер представник постачальник покупець
    продавець замовник учасник фізична особа підприємець фоп генеральний головний
    уповноважений уповноважена
""".split()
)

Found = list[tuple[str, str]]


def mask_pii(
    text: str,
    extra_names: tuple[str, ...] = (),
    extra_addresses: tuple[str, ...] = (),
    extra_hidden: tuple[str, ...] = (),
    *,
    company_codes: bool = False,
) -> tuple[str, Found]:
    found: Found = []
    text = _mask_listed(text, extra_hidden, "ПРИХОВАНО", found)
    text = _mask_listed(text, extra_addresses, "АДРЕСА", found)  # a sole trader's home address
    if company_codes:
        text = _replace(COMPANY_CODE, text, "ЄДРПОУ", found)
    for kind, pattern in CODE_PATTERNS:
        text = _replace(pattern, text, kind, found)
    surnames = list(extra_names)
    text = _mask_full_names(text, surnames, found)
    text = _mask_initials(text, surnames, found)
    text = _mask_first_names(text, surnames, found)
    stems = {stem for name in surnames for stem in _stems(name)}
    for stem in sorted(stems, key=lambda stem: (-len(stem), stem)):  # longest first, then a-z
        surname_forms = re.compile(
            rf"\b{re.escape(stem)}[{UPPER}{LOWER}]{{0,{MAX_ENDING}}}\b", re.IGNORECASE
        )
        text = _replace(surname_forms, text, "ПІБ", found)
    return text, found


def _mask_listed(text: str, values: tuple[str, ...], kind: str, found: Found) -> str:
    """Reviewer-listed strings, matched even when the PDF broke them across lines."""
    for value in values:
        words = (re.escape(word) for word in value.split())
        text = _replace(re.compile(r"\s+".join(words)), text, kind, found)
    return text


def _replace(pattern: re.Pattern[str], text: str, kind: str, found: Found) -> str:
    """Replace the match, or only its "value" group when the pattern keeps a cue like "тел."."""

    def tag(match: re.Match[str]) -> str:
        group = "value" if "value" in pattern.groupindex else 0
        found.append((kind, match.group(group)))
        start, end = match.start(group) - match.start(), match.end(group) - match.start()
        return f"{match.group(0)[:start]}[{kind}]{match.group(0)[end:]}"

    return pattern.sub(tag, text)


def _is_name(word: str | None, text: str) -> bool:
    """A name word is not a job title, not a legal form like "ТОВ", and not a common word:
    "Договір" after a name is a common word because "договірна" appears in lower case."""
    if word is None or word.casefold() in NOT_NAMES:
        return False
    if word.isupper() and len(word) <= MAX_ABBREVIATION:
        return False
    return re.search(rf"\b{re.escape(_stem(word))}", text) is None


def _is_eponym(text: str, start: int) -> bool:
    return NAMED_AFTER.search(text, 0, start) is not None


def _mask_full_names(text: str, surnames: list[str], found: Found) -> str:
    """Find a patronymic and take the name words around it: "Петренко Іван Степанович",
    "Олена Петрівна Мирошник", "Іван Степанович" in a signature, or the patronymic alone
    when OCR garbled the name next to it."""
    spans: list[tuple[int, int]] = []
    for word in WORD.finditer(text):
        value = word.group(0)
        if not value[0].isupper() or len(value) < MIN_PATRONYMIC or not PATRONYMIC.fullmatch(value):
            continue
        before = TWO_NAMES_BEFORE.search(text, 0, word.start())
        if before is None or not _is_name(before.group(2), text):
            spans.append((word.start(), word.end()))
            continue
        if _is_eponym(text, before.start()):
            continue
        start, end = before.start(2), word.end()
        after = NAME_AFTER.match(text, word.end())
        if _is_name(before.group(1), text):
            start = before.start(1)
            surnames.append(before.group(1))
        elif after and _is_name(after.group(1), text):
            end = after.end(1)
            surnames.append(after.group(1))
        spans.append((start, end))
    for start, end in reversed(spans):
        found.append(("ПІБ", text[start:end]))
        text = f"{text[:start]}[ПІБ]{text[end:]}"
    return text


def _mask_initials(text: str, surnames: list[str], found: Found) -> str:
    def tag(match: re.Match[str]) -> str:
        if _is_eponym(text, match.start()) or not _is_name(match.group(1), text):
            return match.group(0)
        surnames.append(match.group(1))
        found.append(("ПІБ", match.group(0)))
        return "[ПІБ]"

    text = INITIALS_THEN_SURNAME.sub(tag, text)
    return SURNAME_THEN_INITIALS.sub(tag, text)


def _mask_first_names(text: str, surnames: list[str], found: Found) -> str:
    """Catch names written without a patronymic: "Директор Василь Пиндус" or a lone "СЕРГІЙ"."""
    spans: list[tuple[int, int]] = []
    for name in FIRST_NAME.finditer(text):
        if not name.group(0)[0].isupper() or (spans and name.start() < spans[-1][1]):
            continue
        if _is_eponym(text, name.start()):
            continue
        start, end = name.start(), name.end()
        after = NAME_AFTER.match(text, end)
        before = NAME_BEFORE.search(text, 0, start)
        if after and _is_name(after.group(1), text):
            end = after.end(1)
            surnames.append(after.group(1))
        elif before and _is_name(before.group(1), text):
            start = before.start(1)
            surnames.append(before.group(1))
        spans.append((start, end))
    for start, end in reversed(spans):
        found.append(("ПІБ", text[start:end]))
        text = f"{text[:start]}[ПІБ]{text[end:]}"
    return text


def _stem(surname: str) -> str:
    """Strip one case ending so "Коваленка" also finds "Коваленко" and "КОВАЛЕНКО"."""
    word = surname.casefold()
    for ending in CASE_ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= MIN_STEM:
            return word[: -len(ending)]
    return word


def _stems(surname: str) -> set[str]:
    stems = {_stem(surname)}
    word = surname.casefold()
    for ending, short in FLEETING_VOWELS:
        if word.endswith(ending) and len(word) - len(ending) + len(short) >= MIN_STEM:
            stems.add(word[: -len(ending)] + short)
    return stems


def suspicious_words(text: str) -> list[str]:
    """Capitalised or Latin-lookalike words right next to a masked name: a name OCR garbled
    ("IOpia" for "Юрія") or one the rules missed. Listed in the review file for a person."""
    words = []
    for tag in re.finditer(r"\[ПІБ\]", text):
        before = re.search(r"(\S+)\s+$", text[max(0, tag.start() - 40) : tag.start()])
        after = re.match(r"\s+(\S+)", text[tag.end() :])
        for match in (before, after):
            word = match.group(1).strip(".,;:()«»\"'") if match else ""
            if word[:1].isupper() and word.casefold() not in NOT_NAMES and "[" not in word:
                words.append(word)
    return list(dict.fromkeys(words))


def read_reviewer_list(path: Path) -> tuple[str, ...]:
    """One entry per line, "#" starts a comment; a missing file means an empty list."""
    if not path.exists():
        return ()
    lines = (line.strip() for line in path.read_text().splitlines())
    return tuple(line for line in lines if line and not line.startswith("#"))



def mask_parts(
    texts: Sequence[str], raw_dir: Path, *, company_codes: bool = False
) -> tuple[tuple[str, ...], Found]:
    """Mask all parts as one text, as stage 1 did: a surname found on page 1 is hidden on page 5.
    Lists a reviewer added (extra_names.txt and others) are read from the folder of the raw file."""
    joined = f"\n{PAGE_BREAK}\n".join(texts)
    masked, found = mask_pii(
        clean_text(joined),
        read_reviewer_list(raw_dir / EXTRA_NAMES_NAME),
        read_reviewer_list(raw_dir / EXTRA_ADDRESSES_NAME),
        read_reviewer_list(raw_dir / EXTRA_HIDDEN_NAME),
        company_codes=company_codes,
    )
    parts = tuple(part.strip("\n") for part in masked.split(PAGE_BREAK))
    if len(parts) != len(texts):
        raise ValueError(f"masking merged parts: {len(texts)} in, {len(parts)} out")
    return parts, found
