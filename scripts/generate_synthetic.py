import argparse
import json
import random
import re
import sys
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path

import ollama

from src.schemas.deal import (
    Basis,
    Crop,
    Currency,
    Deal,
    Delivery,
    DaysType,
    Lab,
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

ROOT = Path(__file__).resolve().parents[1]
MAX_ATTEMPTS = 3

ALIASES: dict[type[StrEnum], dict[str, tuple[str, ...]]] = {
    Crop: {
        "wheat": ("пшениця", "пшениці", "пшеницю", "пш"),
        "corn": ("кукурудза", "кукурудзи", "кукурудзу", "кук"),
        "barley": ("ячмінь", "ячменю", "яч"),
        "sunflower": ("соняшник", "соняшнику", "соняш"),
        "soy": ("соя", "сої", "сою"),
        "rapeseed": ("ріпак", "ріпаку", "ріп"),
        "other": ("інша культура", "інша зернова"),
    },
    Party: {
        "buyer": ("покупець", "покупця", "покупцем"),
        "seller": ("продавець", "продавця", "продавцем", "постачальник"),
    },
    Currency: {
        "UAH": ("грн", "гривень", "гривні", "₴", "uah"),
        "USD": ("долар", "доларів", "дол.", "$", "usd"),
        "EUR": ("євро", "€", "eur"),
    },
    Basis: {
        "EXW": ("exw",),
        "FCA": ("fca",),
        "CPT": ("cpt", "сpt", "спт", "срт"),
        "DAP": ("dap",),
        "FOB": ("fob",),
        "other": ("інший базис",),
    },
    DaysType: {
        "calendar": ("календарних днів", "календарний день", "календарні дні"),
        "banking": ("банківських днів", "банківський день", "робочих днів"),
    },
    PaymentTrigger: {
        "prepayment": ("передоплата", "передоплати", "аванс"),
        "after_delivery": ("після поставки", "після доставки", "від поставки"),
        "after_documents": ("після документів", "після отримання документів"),
        "other": ("інша умова оплати",),
    },
    PenaltyTrigger: {
        "late_delivery": ("прострочення поставки", "затримка поставки"),
        "late_payment": ("прострочення оплати", "затримка оплати"),
        "under_delivery": ("недопоставка", "непоставка"),
        "other": ("інше порушення",),
    },
    PenaltyUnit: {
        "pct_per_day": ("% за день", "відсотка за день", "відсотків за день"),
        "nbu_rate_multiple": ("подвійної облікової ставки нбу", "ставки нбу"),
        "one_time_pct": ("одноразово", "разово", "% одноразово"),
    },
    Lab: {
        "buyer": ("лабораторія покупця", "лабораторії покупця"),
        "seller": ("лабораторія продавця", "лабораторії продавця"),
        "independent": ("незалежна лабораторія", "незалежної лабораторії"),
    },
}

BOOLEAN_ALIASES: dict[str, dict[bool, tuple[str, ...]]] = {
    "price.vat_included": {
        True: ("з пдв", "включно з пдв", "пдв включено"),
        False: ("без пдв", "пдв не включено"),
    },
    "payment.vat_after_registration": {
        True: ("після реєстрації податкової накладної", "після реєстрації пн"),
        False: ("незалежно від реєстрації податкової накладної", "до реєстрації пн"),
    },
}


def make_deal(rng: random.Random, index: int) -> Deal:
    crop = rng.choice(list(Crop))
    quantity = rng.choice((100.0, 250.0, 500.0, 1000.0, 2500.0))
    price_amount = rng.choice((8400.0, 8750.0, 9200.0, 9450.0))
    start_date = date(2026, 11, 1) + timedelta(days=rng.randrange(45))

    return Deal(
        crop=crop if rng.random() < 0.9 else None,
        quantity_t=quantity if rng.random() < 0.85 else None,
        tolerance=(
            Tolerance(
                pct=rng.choice((3.0, 5.0, 10.0)),
                option=rng.choice(list(Party)),
            )
            if rng.random() < 0.5
            else None
        ),
        price=(
            Price(
                amount_per_t=price_amount,
                currency=Currency.UAH,
                vat_included=rng.choice((True, False)),
            )
            if rng.random() < 0.9
            else None
        ),
        basis=rng.choice(list(Basis)) if rng.random() < 0.8 else None,
        delivery=(
            Delivery(
                place=rng.choice(("Одеса", "Миколаїв", "Полтава", "Черкаси")),
                date_from=start_date,
                date_to=start_date + timedelta(days=rng.randrange(1, 8)),
            )
            if rng.random() < 0.65
            else Delivery(
                period_raw="протягом 10 днів після передоплати",
            )
            if rng.random() < 0.25
            else None
        ),
        payment=(
            Payment(
                days=rng.choice((3, 5, 7, 10, 14)),
                days_type=rng.choice(list(DaysType)),
                trigger=rng.choice(list(PaymentTrigger)),
                vat_after_registration=rng.choice((True, False)),
            )
            if rng.random() < 0.7
            else None
        ),
        penalties=(
            [
                Penalty(
                    payer=rng.choice(list(Party)),
                    trigger=rng.choice(list(PenaltyTrigger)),
                    amount=rng.choice((0.1, 0.2, 0.5)),
                    unit=rng.choice(list(PenaltyUnit)),
                    cap_pct=rng.choice((5.0, 10.0, 20.0)),
                )
            ]
            if rng.random() < 0.45
            else None
        ),
        quality=(
            Quality(
                grade=rng.choice(("3", "фуражна")),
                protein_min=rng.choice((11.0, 11.5, 12.0)),
                moisture_max=rng.choice((13.5, 14.0)),
                foreign_matter_max=rng.choice((1.5, 2.0)),
                lab=rng.choice(list(Lab)),
            )
            if rng.random() < 0.55
            else None
        ),
        buyer=f'ТОВ "Зерно Поле {index:02d}"',
    )


def normalize_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold())


def normalize_number(value: int | float | Decimal) -> Decimal:
    return Decimal(str(value)).normalize()


def text_numbers(text: str) -> set[Decimal]:
    numbers: set[Decimal] = set()
    for match in re.finditer(r"(?<!\w)\d+(?:[\s\u00a0]\d{3})*(?:[.,]\d+)?", text):
        value = re.sub(r"[\s\u00a0]", "", match.group())
        try:
            numbers.add(Decimal(value.replace(",", ".")).normalize())
        except InvalidOperation:
            continue
    return numbers


def value_is_present(path: str, value: object, text: str, numbers: set[Decimal]) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        aliases = BOOLEAN_ALIASES.get(path, {}).get(value, ())
        return any(normalize_text(alias) in normalize_text(text) for alias in aliases)
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return normalize_number(value) in numbers
    if isinstance(value, date):
        date_formats = (value.isoformat(), value.strftime("%d.%m.%Y"))
        normalized_text = normalize_text(text)
        return any(normalize_text(item) in normalized_text for item in date_formats)
    if isinstance(value, StrEnum):
        aliases = ALIASES.get(type(value), {}).get(value.value, (value.value,))
        normalized_text = normalize_text(text)
        return any(normalize_text(alias) in normalized_text for alias in aliases)
    if isinstance(value, str):
        return normalize_text(value) in normalize_text(text)
    return False


def missing_values(deal: Deal, text: str) -> list[str]:
    missing: list[str] = []
    numbers = text_numbers(text)

    def visit(path: str, value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{path}.{key}" if path else key, child)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(f"{path}[{index}]", child)
        elif not value_is_present(path, value, text, numbers):
            missing.append(path)

    visit("", deal.model_dump(mode="python", exclude_none=True))
    return missing


def build_prompt(deal: Deal, style_index: int, missing: list[str] | None = None) -> str:
    styles = (
        "Напиши короткий Viber-рекап в 1–3 реченнях, з розмовними скороченнями.",
        "Напиши лист трейдера з темою та коротким діловим текстом.",
        "Напиши неформальне повідомлення змішаною українською та російською, з 1–2 друкарськими помилками.",
        "Напиши докладний рекап на кілька речень, але без зайвих умов.",
    )
    deal_json = json.dumps(deal.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
    correction = (
        f"\nУ попередній відповіді не вдалося знайти в тексті ці значення: "
        f"{', '.join(missing)}. Перефразуй, щоб кожне значення було явно згадане."
        if missing
        else ""
    )
    return (
        "Ти пишеш синтетичне повідомлення українського трейдера про продаж зерна. "
        "Створи лише текст повідомлення, без пояснень і без JSON. "
        "Включи кожну надану умову та значення без змін; не додавай інших умов. "
        f"Стиль: {styles[style_index % len(styles)]}\n"
        f"Еталонні умови (використай усі ненульові значення): {deal_json}"
        f"{correction}"
    )


def ollama_generate(
    client: ollama.Client,
    model: str,
    prompt: str,
) -> str:
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7, "top_p": 0.9},
    )
    return response.message.content.strip()


def generate_valid_text(
    deal: Deal,
    style_index: int,
    generate: Callable[[str], str],
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[str, int]:
    missing: list[str] | None = None
    for attempt in range(1, max_attempts + 1):
        text = generate(build_prompt(deal, style_index, missing)).strip()
        if not text:
            missing = ["порожній текст"]
            continue
        missing = missing_values(deal, text)
        if not missing:
            return text, attempt
    raise ValueError(
        f"Generated text did not contain all deal values after {max_attempts} attempts: "
        f"{', '.join(missing or ['unknown'])}"
    )


def output_paths(count: int, examples: int) -> list[Path]:
    paths: list[Path] = []
    for index in range(1, count + 1):
        name = f"synthetic_{index:03d}"
        paths.extend((ROOT / "data/texts" / f"{name}.txt", ROOT / "data/synthetic" / f"{name}.json"))
    for index in range(1, examples + 1):
        name = f"prompt_example_{index:03d}"
        paths.extend(
            (
                ROOT / "data/synthetic/prompt_examples" / f"{name}.txt",
                ROOT / "data/synthetic" / f"{name}.json",
            )
        )
    return paths


def generate_dataset(
    model: str,
    seed: int,
    count: int,
    examples: int,
    overwrite: bool = False,
    host: str | None = None,
) -> None:
    paths = output_paths(count, examples)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing files; use --overwrite to replace them: "
            + ", ".join(str(path.relative_to(ROOT)) for path in existing)
        )

    client = ollama.Client(host=host, timeout=120)
    rng = random.Random(seed)
    records: list[tuple[Path, Path, dict[str, object]]] = []
    jobs = [(i, "dev") for i in range(1, count + 1)] + [
        (i, "prompt_example") for i in range(1, examples + 1)
    ]

    for index, split in jobs:
        offset = index if split == "dev" else count + index
        deal = make_deal(random.Random(seed + offset), offset)
        text, attempts = generate_valid_text(
            deal,
            style_index=offset - 1,
            generate=lambda prompt: ollama_generate(client, model, prompt),
        )
        stem = f"synthetic_{index:03d}" if split == "dev" else f"prompt_example_{index:03d}"
        text_path = (
            ROOT / "data/texts" / f"{stem}.txt"
            if split == "dev"
            else ROOT / "data/synthetic/prompt_examples" / f"{stem}.txt"
        )
        json_path = ROOT / "data/synthetic" / f"{stem}.json"
        record: dict[str, object] = {
            "id": stem,
            "split": split,
            "model": model,
            "seed": seed + offset,
            "generation_attempts": attempts,
            "text_path": str(text_path.relative_to(ROOT)),
            "deal": deal.model_dump(mode="json"),
        }
        records.append((text_path, json_path, {"record": record, "text": text}))
        print(f"Generated {stem} (attempt {attempts}/{MAX_ATTEMPTS})")

    for text_path, json_path, result in records:
        text_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(f"{result['text']}\n", encoding="utf-8")
        json_path.write_text(
            json.dumps(result["record"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Saved {len(records)} synthetic records using {model}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic deal texts with Ollama.")
    parser.add_argument("--model", required=True, help="Installed Ollama model name.")
    parser.add_argument("--host", default=None, help="Ollama server URL; defaults to localhost.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=20, help="Number of dev texts.")
    parser.add_argument("--examples", type=int, default=3, help="Prompt examples outside dev.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 0 or args.examples < 0:
        print("count and examples must be non-negative", file=sys.stderr)
        return 2
    try:
        generate_dataset(
            model=args.model,
            seed=args.seed,
            count=args.count,
            examples=args.examples,
            overwrite=args.overwrite,
            host=args.host,
        )
    except (FileExistsError, ValueError, ollama.OllamaError) as error:
        print(f"Generation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
