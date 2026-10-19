"""Turn the Prozorro contract PDFs from data/raw/prozorro/ into anonymized texts in data/texts/.

PyMuPDF reads the text layer as it is: stamp and signature noise stays, the texts must be messy.
Phones, IBANs, bank accounts, personal tax numbers (РНОКПП), emails and people's names become
tags. Company names and their 8-digit ЄДРПОУ codes are public and stay. Every masked value goes
to a review file inside data/raw/ (outside git), so a person checks the result before labeling;
a missed surname goes to data/raw/prozorro/extra_names.txt and the script is run again."""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "prozorro"
TEXTS_DIR = ROOT / "data" / "texts"
SOURCES_PATH = ROOT / "data" / "prozorro_sources.csv"
REVIEW_NAME = "mask_review.txt"
EXTRA_NAMES_NAME = "extra_names.txt"
FILE_PREFIX = "prozorro_"
TENDER_URL = "https://prozorro.gov.ua/tender/{tender_id}"

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
NAMED_AFTER = re.compile(
    r"(?:\b[іiІI1]\S{1,6}|вул\.|вулиця|просп\.|проспект|пров\.|провулок|бульв\.|бульвар|пл\.|площа)\s*$",
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


def extract_text(path: Path) -> tuple[str, int]:
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc), doc.page_count


def clean_text(text: str) -> str:
    text = re.sub(r"\xad\s*\n\s*", "", text).replace("\xad", "")  # soft hyphen splits a word
    lines = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", lines).strip("\n") + "\n"


def mask_pii(text: str, extra_names: tuple[str, ...] = ()) -> tuple[str, Found]:
    found: Found = []
    for kind, pattern in CODE_PATTERNS:
        text = _replace(pattern, text, kind, found)
    surnames = list(extra_names)
    text = _mask_full_names(text, surnames, found)
    text = _mask_initials(text, surnames, found)
    text = _mask_first_names(text, surnames, found)
    stems = {stem for name in surnames for stem in _stems(name)}
    for stem in sorted(stems, key=len, reverse=True):
        surname_forms = re.compile(
            rf"\b{re.escape(stem)}[{UPPER}{LOWER}]{{0,{MAX_ENDING}}}\b", re.IGNORECASE
        )
        text = _replace(surname_forms, text, "ПІБ", found)
    return text, found


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


def read_extra_names(raw_dir: Path) -> tuple[str, ...]:
    path = raw_dir / EXTRA_NAMES_NAME
    if not path.exists():
        return ()
    lines = (line.strip() for line in path.read_text().splitlines())
    return tuple(line for line in lines if line and not line.startswith("#"))


def convert_all(
    raw_dir: Path,
    texts_dir: Path,
    sources_path: Path,
    review_path: Path,
    overwrite: bool = False,
) -> list[dict[str, str]]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"no folder with contract PDFs: {raw_dir}")
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDFs in {raw_dir}")
    ids = [f"{FILE_PREFIX}{index:03d}" for index in range(1, len(pdfs) + 1)]
    existing = [
        texts_dir / f"{doc_id}.txt" for doc_id in ids if (texts_dir / f"{doc_id}.txt").exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"{existing[0]} already exists; pass --overwrite to replace the texts"
        )

    extra_names = read_extra_names(raw_dir)
    texts_dir.mkdir(parents=True, exist_ok=True)
    rows, review = [], []
    for doc_id, pdf in zip(ids, pdfs, strict=True):
        raw_text, pages = extract_text(pdf)
        text, found = mask_pii(clean_text(raw_text), extra_names)
        (texts_dir / f"{doc_id}.txt").write_text(text)
        tender_id = pdf.stem.removeprefix(FILE_PREFIX)
        counts = Counter(kind for kind, _ in found)
        rows.append(
            {
                "id": doc_id,
                "tender_id": tender_id,
                "tender_url": TENDER_URL.format(tender_id=tender_id),
                "pages": str(pages),
                "chars": str(len(text)),
                "masked": " ".join(f"{kind}:{count}" for kind, count in sorted(counts.items())),
            }
        )
        review.append(f"== {doc_id} ({tender_id})")
        review.extend(f"{kind}: {value!r}" for kind, value in found)
        review.extend(f"CHECK: {word!r}" for word in suspicious_words(text))
    _write_sources(sources_path, rows)
    review_path.write_text("\n".join(review) + "\n")
    return rows


def _write_sources(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and anonymize Prozorro contract texts.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--texts-dir", type=Path, default=TEXTS_DIR)
    parser.add_argument("--sources", type=Path, default=SOURCES_PATH)
    parser.add_argument("--overwrite", action="store_true", help="replace existing prozorro texts")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    review_path = args.raw_dir / REVIEW_NAME
    try:
        rows = convert_all(args.raw_dir, args.texts_dir, args.sources, review_path, args.overwrite)
    except (OSError, pymupdf.FileDataError) as error:
        print(f"Conversion failed: {error}", file=sys.stderr)
        return 1
    for row in rows:
        print(f"{row['id']}  {row['tender_id']}  {row['pages']} p.  masked: {row['masked'] or '-'}")
    print(f"Check every masked value and the texts themselves: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
