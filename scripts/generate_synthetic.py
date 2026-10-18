"""Generate the synthetic dev set: Python builds a Deal, an Ollama model writes a messy text,
and a field-aware check confirms the text states every value of that Deal."""

import argparse
import json
import random
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import httpx
import ollama

from scripts.synthetic_checks import missing_values
from scripts.synthetic_render import describe_missing, render_conditions
from src.schemas.deal import (
    FEED_GRADE,
    Basis,
    Crop,
    Currency,
    DaysType,
    Deal,
    Delivery,
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
MAX_CANDIDATES_PER_TEXT = 3  # a skipped deal is replaced by the next seed, up to this many per text
OLLAMA_TIMEOUT_S = 120
EMPTY_TEXT = "порожній текст"
# Lapa is built on Gemma 3 and Qwen 2.5 is its fallback: a generator from these families
# would give one candidate a head start on texts written in its own style.
COMPARED_MODEL_FAMILIES = ("lapa", "gemma", "qwen")

WHEAT_PRICES_UAH = tuple(float(price) for price in range(8300, 9701, 50))
QUANTITIES_T = (100.0, 250.0, 500.0, 1000.0, 2500.0)
TOLERANCES_PCT = (3.0, 5.0, 10.0)
GENERATED_BASES = (Basis.EXW, Basis.FCA, Basis.CPT, Basis.DAP)  # FOB is a port basis priced in USD
DELIVERY_PLACES = ("Одеса", "Миколаїв", "Полтава", "Черкаси")
RELATIVE_PERIODS = (
    "протягом 20 днів після підписання договору",
    "протягом 30 днів з дати договору",
)
PAYMENT_DAYS = (3, 5, 7, 10, 14)
WHEAT_GRADES = ("2", "3", "4", FEED_GRADE)
PROTEIN_MIN_PCT = {"2": (12.0, 12.5), "3": (11.0, 11.5), "4": (10.0, 10.5)}  # feed has no minimum
MOISTURE_MAX_PCT = (13.5, 14.0)
FOREIGN_MATTER_MAX_PCT = (1.0, 2.0)
PENALTY_CAPS_PCT = (5.0, 10.0, 20.0)
# Who pays and which units make sense for each breach; the NBU-rate limit covers money debts only.
PENALTY_PAYER = {
    PenaltyTrigger.LATE_DELIVERY: Party.SELLER,
    PenaltyTrigger.UNDER_DELIVERY: Party.SELLER,
    PenaltyTrigger.LATE_PAYMENT: Party.BUYER,
}
PENALTY_UNITS = {
    PenaltyTrigger.LATE_DELIVERY: (PenaltyUnit.PCT_PER_DAY, PenaltyUnit.ONE_TIME_PCT),
    PenaltyTrigger.UNDER_DELIVERY: (PenaltyUnit.ONE_TIME_PCT,),
    PenaltyTrigger.LATE_PAYMENT: (PenaltyUnit.PCT_PER_DAY, PenaltyUnit.NBU_RATE_MULTIPLE),
}
PENALTY_AMOUNTS = {
    PenaltyUnit.PCT_PER_DAY: (0.1, 0.2, 0.5),
    PenaltyUnit.NBU_RATE_MULTIPLE: (2.0, 2.0, 1.0),  # contracts usually cite the legal maximum
    PenaltyUnit.ONE_TIME_PCT: (5.0, 10.0, 20.0),
}

LENGTHS = {
    "line": "один рядок, як у Viber",
    "short": "2–4 короткі речення",
    "letter": "діловий лист на 1–2 абзаци з темою і підписом",
    "long": "докладний лист на 1–2 сторінки: привітання, логістика, умови, підпис",
}
LANGUAGES = {
    "uk": "українською",
    "uk_ru": "українською з російськими словами й фразами, як у живому листуванні",
    "uk_en": "українською з англійськими термінами (wheat, tolerance, payment, buyer's option)",
}
TYPOS = {
    "none": "без друкарських помилок",
    "few": "з 1–2 друкарськими помилками",
    "many": "з 3–5 друкарськими помилками",
}
ABBREVIATIONS = "скорочуй, як у месенджері: пш, кл, т, б.д., грн, ПН"


@dataclass(frozen=True)
class Dirtiness:
    """How messy one synthetic text should be; every axis is chosen independently."""

    length: str
    language: str
    typos: str
    abbreviations: bool


def make_dirtiness(rng: random.Random) -> Dirtiness:
    return Dirtiness(
        length=rng.choice(tuple(LENGTHS)),
        language=rng.choice(tuple(LANGUAGES)),
        typos=rng.choice(tuple(TYPOS)),
        abbreviations=rng.random() < 0.5,
    )


def make_deal(rng: random.Random, index: int) -> Deal:
    """Build a realistic wheat deal; some fields stay null on purpose."""
    return Deal(
        crop=Crop.WHEAT if rng.random() < 0.9 else None,
        quantity_t=rng.choice(QUANTITIES_T) if rng.random() < 0.85 else None,
        tolerance=_tolerance(rng) if rng.random() < 0.5 else None,
        price=_price(rng) if rng.random() < 0.9 else None,
        basis=rng.choice(GENERATED_BASES) if rng.random() < 0.8 else None,
        delivery=_delivery(rng),
        payment=_payment(rng) if rng.random() < 0.7 else None,
        penalties=_penalties(rng) if rng.random() < 0.45 else None,
        quality=_quality(rng) if rng.random() < 0.55 else None,
        buyer=f'ТОВ "Зерно Поле {index:02d}"',
    )


def _tolerance(rng: random.Random) -> Tolerance:
    return Tolerance(
        pct=rng.choice(TOLERANCES_PCT),
        option=rng.choice((Party.BUYER, Party.SELLER)),
    )


def _price(rng: random.Random) -> Price:
    return Price(
        amount_per_t=rng.choice(WHEAT_PRICES_UAH),
        currency=Currency.UAH,
        vat_included=rng.choice((True, False)),
    )


def _delivery(rng: random.Random) -> Delivery | None:
    roll = rng.random()
    if roll < 0.65:
        start = date(2026, 11, 1) + timedelta(days=rng.randrange(45))
        return Delivery(
            place=rng.choice(DELIVERY_PLACES),
            date_from=start,
            date_to=start + timedelta(days=rng.randrange(1, 8)),
        )
    if roll < 0.85:
        return Delivery(place=rng.choice(DELIVERY_PLACES), period_raw=rng.choice(RELATIVE_PERIODS))
    return None


def _payment(rng: random.Random) -> Payment:
    return Payment(
        days=rng.choice(PAYMENT_DAYS),
        days_type=rng.choice((DaysType.CALENDAR, DaysType.BANKING)),
        trigger=rng.choice(
            (PaymentTrigger.PREPAYMENT, PaymentTrigger.AFTER_DELIVERY, PaymentTrigger.AFTER_DOCUMENTS)
        ),
        vat_after_registration=True if rng.random() < 0.3 else None,
    )


def _penalties(rng: random.Random) -> list[Penalty]:
    triggers = rng.sample(tuple(PENALTY_PAYER), k=rng.choice((1, 1, 2)))
    return [_penalty(rng, trigger) for trigger in triggers]


def _penalty(rng: random.Random, trigger: PenaltyTrigger) -> Penalty:
    unit = rng.choice(PENALTY_UNITS[trigger])
    has_cap = unit == PenaltyUnit.PCT_PER_DAY and rng.random() < 0.6
    return Penalty(
        payer=PENALTY_PAYER[trigger],
        trigger=trigger,
        amount=rng.choice(PENALTY_AMOUNTS[unit]),
        unit=unit,
        cap_pct=rng.choice(PENALTY_CAPS_PCT) if has_cap else None,
    )


def _quality(rng: random.Random) -> Quality:
    grade = rng.choice(WHEAT_GRADES)
    return Quality(
        grade=grade,
        protein_min=rng.choice(PROTEIN_MIN_PCT[grade]) if grade in PROTEIN_MIN_PCT else None,
        moisture_max=rng.choice(MOISTURE_MAX_PCT),
        foreign_matter_max=rng.choice(FOREIGN_MATTER_MAX_PCT),
        lab=rng.choice((Lab.BUYER, Lab.SELLER, Lab.INDEPENDENT)),
    )


def build_prompt(deal: Deal, dirtiness: Dirtiness, missing: list[str] | None = None) -> str:
    conditions = "\n".join(f"- {line}" for line in render_conditions(deal))
    style = [
        f"Довжина: {LENGTHS[dirtiness.length]}.",
        f"Мова: {LANGUAGES[dirtiness.language]}.",
        f"Помилки: {TYPOS[dirtiness.typos]}, лише у звичайних словах, не в числах і не в назвах умов.",
    ]
    if dirtiness.abbreviations:
        style.append(f"Скорочення: {ABBREVIATIONS}.")
    correction = (
        "\n\nУ попередній відповіді не вдалося знайти ці умови: "
        f"{'; '.join(describe_missing(path) for path in missing)}. "
        "Перепиши текст так, щоб кожна з них була явно згадана поруч зі своїм значенням."
        if missing
        else ""
    )
    return (
        "Ти пишеш синтетичне повідомлення українського зернотрейдера фермеру. "
        "Поверни лише текст повідомлення, без пояснень, JSON чи переліку умов. "
        "Передай кожну умову зі списку з тим самим значенням, але своїми словами; "
        "не додавай нових умов і не пропускай жодної. "
        "Дати пиши як у живому листуванні, наприклад «5–12 листопада» або «05.11»; "
        "не згадуй інших дат, сум і відсотків, крім тих, що в умовах.\n"
        + "\n".join(style)
        + f"\nУмови:\n{conditions}{correction}"
    )


def ollama_generate(client: ollama.Client, model: str, prompt: str) -> str:
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7, "top_p": 0.9},
    )
    return (response.message.content or "").strip()


def generate_valid_text(
    deal: Deal,
    dirtiness: Dirtiness,
    generate: Callable[[str], str],
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[str, int]:
    missing: list[str] | None = None
    for attempt in range(1, max_attempts + 1):
        text = generate(build_prompt(deal, dirtiness, missing)).strip()
        missing = missing_values(deal, text) if text else [EMPTY_TEXT]
        if not missing:
            return text, attempt
    raise ValueError(
        f"Generated text did not contain all deal values after {max_attempts} attempts: "
        f"{', '.join(missing or ['unknown'])}"
    )


def _slots(count: int, examples: int) -> list[tuple[str, int]]:
    return [("dev", index) for index in range(1, count + 1)] + [
        ("prompt_example", index) for index in range(1, examples + 1)
    ]


def _slot_paths(split: str, index: int, root: Path) -> tuple[Path, Path]:
    if split == "dev":
        stem = f"synthetic_{index:03d}"
        return root / "data/texts" / f"{stem}.txt", root / "data/synthetic" / f"{stem}.json"
    stem = f"prompt_example_{index:03d}"
    return (
        root / "data/synthetic/prompt_examples" / f"{stem}.txt",
        root / "data/synthetic" / f"{stem}.json",
    )


def output_paths(count: int, examples: int, root: Path = ROOT) -> list[Path]:
    return [
        path
        for split, index in _slots(count, examples)
        for path in _slot_paths(split, index, root)
    ]


def generate_dataset(
    generate: Callable[[str], str],
    model: str,
    seed: int,
    count: int,
    examples: int,
    overwrite: bool = False,
    root: Path = ROOT,
) -> int:
    """Save count dev texts and examples prompt examples; return how many deals were skipped.

    A deal whose text still fails the checks after MAX_ATTEMPTS is skipped and replaced by the
    next seed. Every text is written as soon as it passes, so a crash keeps the finished ones.
    """
    _prepare_outputs(output_paths(count, examples, root), overwrite, root)
    slots = _slots(count, examples)
    max_candidates = len(slots) * MAX_CANDIDATES_PER_TEXT
    saved = skipped = 0
    for candidate in range(1, max_candidates + 1):
        if saved == len(slots):
            break
        rng = random.Random(f"{seed}:{candidate}")  # seed + candidate would repeat across runs
        deal, dirtiness = make_deal(rng, candidate), make_dirtiness(rng)
        try:
            text, attempts = generate_valid_text(deal, dirtiness, generate)
        except ValueError as error:
            skipped += 1
            print(f"Skipping candidate {candidate}: {error}", file=sys.stderr)
            continue
        details = {
            "model": model,
            "seed": seed,
            "candidate": candidate,
            "generation_attempts": attempts,
            "dirtiness": asdict(dirtiness),
            "deal": deal.model_dump(mode="json"),
        }
        _save_text(root, slots[saved], text, details)
        saved += 1
    if saved < len(slots):
        raise ValueError(
            f"Only {saved} of {len(slots)} texts passed the checks after {max_candidates} "
            "candidates; the saved texts are kept."
        )
    return skipped


def _prepare_outputs(paths: list[Path], overwrite: bool, root: Path) -> None:
    """Refuse to touch existing outputs, or with overwrite remove them so no stale file survives."""
    existing = [path for path in paths if path.exists()]
    if overwrite:
        for path in existing:
            path.unlink()
        return
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing files; use --overwrite to replace them: "
            + ", ".join(str(path.relative_to(root)) for path in existing)
        )


def _save_text(root: Path, slot: tuple[str, int], text: str, details: dict[str, object]) -> None:
    split, index = slot
    text_path, json_path = _slot_paths(split, index, root)
    record = {
        "id": text_path.stem,
        "split": split,
        **details,
        "text_path": text_path.relative_to(root).as_posix(),
    }
    text_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(f"{text}\n", encoding="utf-8")
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {text_path.stem} (attempt {details['generation_attempts']}/{MAX_ATTEMPTS})")


def is_compared_model(model: str) -> bool:
    return any(family in model.casefold() for family in COMPARED_MODEL_FAMILIES)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic deal texts with Ollama.")
    parser.add_argument("--model", required=True, help="Installed Ollama model name.")
    parser.add_argument("--host", default=None, help="Ollama server URL; defaults to localhost.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=20, help="Number of dev texts.")
    parser.add_argument("--examples", type=int, default=3, help="Prompt examples outside dev.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.count < 0 or args.examples < 0:
        print("count and examples must be non-negative", file=sys.stderr)
        return 2
    if is_compared_model(args.model):
        print(
            f"{args.model} belongs to a model family under comparison "
            f"({', '.join(COMPARED_MODEL_FAMILIES)}); pick another family, e.g. llama3.1 or mistral",
            file=sys.stderr,
        )
        return 2
    client = ollama.Client(host=args.host, timeout=OLLAMA_TIMEOUT_S)
    try:
        skipped = generate_dataset(
            generate=lambda prompt: ollama_generate(client, args.model, prompt),
            model=args.model,
            seed=args.seed,
            count=args.count,
            examples=args.examples,
            overwrite=args.overwrite,
        )
    except (
        FileExistsError,
        ValueError,
        ConnectionError,
        httpx.HTTPError,  # a timeout reaches us as httpx.ReadTimeout, not wrapped by ollama
        ollama.RequestError,
        ollama.ResponseError,
    ) as error:
        print(f"Generation failed: {error}", file=sys.stderr)
        return 1
    print(f"Done with {args.model}; skipped {skipped} candidate deal(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
