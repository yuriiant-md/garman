"""Stage-1 metrics for every configuration (model x prompt x sampling), per data split.

    uv run python -m src.extraction.metrics

The counting rules are in README.md; one document is scored in src/extraction/scoring.py.
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.extraction.result import ExtractionStatus
from src.extraction.scoring import Counts, score_deal
from src.schemas.deal import Deal

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "stage1_2"  # Lapa's stage-1 runs: --results-dir results/stage1/…
REFERENCE_DIRS = (ROOT / "data" / "synthetic", ROOT / "data" / "golden")
MODELS_PATH = ROOT / "configs" / "models.yaml"
TOKENS_PER_PRICE_UNIT = 1_000_000  # prices in configs/models.yaml are per 1M tokens
COST_BATCH_DOCS = 1000
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 0  # fixed, so the same results always give the same interval
CI_TAIL_PCT = 2.5  # the 95% interval runs from the 2.5th to the 97.5th percentile
LATENCY_TAIL_PCT = 95
SPLIT_ORDER = ("dev", "test")
SOURCE_ORDER = ("synthetic", "prozorro", "own")
DEFAULT_SOURCE = "synthetic"  # data/synthetic/*.json carry no "source" field
ALL_SOURCES = "all"  # every golden source of a split together: the model is chosen on this row
NO_VALUE = "—"
RUN_LABEL_LENGTH = len("2026-10-23T14:05")  # date and minutes of a run's ISO start time
ZERO = Counts(filled=0, filled_correct=0, empty=0, fabricated=0, total=0, correct=0)
HEADER = (
    "Модель", "Промпт", "Семплінг", "Прогін", "Набір", "Док.", "Точність заповнених, % (95% ДІ)",
    "Вигадані, %", "Загальна, %", "Валідні, %", "Failed, %", "Виправлено, %", "p50, с", "p95, с",
    "$ / 1000 док.",
)

FIELD_HEADER = ("Поле", "Заповнено у golden", "null у golden")
FIELD_TOTAL = "**Разом**"
FIELD_LEGEND = (
    "Клітинка: точність по заповнених значеннях golden (правильні/заповнені) · вигад. — "
    "скільки разів модель дала значення там, де в golden null."
)

Record = Mapping[str, object]
Prices = Mapping[str, tuple[float, float]]  # model id -> USD per 1M input and output tokens
ConfigKey = tuple[str, str, str, str]  # model, prompt version, sampling, start of the run


@dataclass(frozen=True)
class FieldGroup:
    """Documents of one kind scored field by field in one table: a split and its sources."""

    title: str
    split: str
    sources: tuple[str, ...]


# Offers (with their own golden set) join here as a second group.
FIELD_GROUPS = (FieldGroup(title="Угоди", split="test", sources=("prozorro", "own")),)


@dataclass(frozen=True)
class Reference:
    deal: Deal
    split: str
    source: str = DEFAULT_SOURCE


@dataclass(frozen=True)
class ConfigMetrics:
    model: str
    prompt_version: str
    sampling: str
    run_at: str  # when the run behind this row started: the latest run of the configuration
    split: str
    source: str
    docs: int
    filled_accuracy: float | None  # the main number: share of golden values the model got right
    filled_ci: tuple[float, float] | None
    fabricated_pct: float | None
    overall_pct: float | None
    valid_pct: float
    failed_pct: float
    fixed_pct: float  # valid only after the one fix attempt; part of valid_pct, not of failed_pct
    latency_p50_s: float
    latency_p95_s: float
    cost_per_1000_docs_usd: float | None


def cost_usd(
    input_tokens: int, output_tokens: int, price_input: float, price_output: float
) -> float:
    return (input_tokens * price_input + output_tokens * price_output) / TOKENS_PER_PRICE_UNIT


def cost_per_1000_docs(
    records: Iterable[Record], price_input: float, price_output: float
) -> float | None:
    """Tokens of all results over answered documents only: an `error` result has no token
    usage, so counting it as a document would make the per-document cost look lower."""
    records = tuple(records)
    answered = sum(record["status"] != ExtractionStatus.ERROR for record in records)
    if not answered:
        return None
    input_tokens = sum(record["input_tokens"] for record in records)
    output_tokens = sum(record["output_tokens"] for record in records)
    cost = cost_usd(input_tokens, output_tokens, price_input, price_output)
    return cost / answered * COST_BATCH_DOCS


def percentile(values: Iterable[float], pct: float) -> float:
    """Linear interpolation between the closest ranks, as numpy does by default."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def share(part: int, whole: int) -> float | None:
    return 100 * part / whole if whole else None


def bootstrap_ci(
    doc_counts: Sequence[Counts], samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float] | None:
    """95% interval of the filled accuracy over documents resampled with replacement: with 5-10
    test documents one error moves the accuracy by several points."""
    rng = random.Random(seed)
    resampled = (sum(rng.choices(doc_counts, k=len(doc_counts)), ZERO) for _ in range(samples))
    estimates = [
        estimate
        for estimate in (share(counts.filled_correct, counts.filled) for counts in resampled)
        if estimate is not None
    ]
    if not estimates:
        return None
    return percentile(estimates, CI_TAIL_PCT), percentile(estimates, 100 - CI_TAIL_PCT)


def _predicted(record: Record) -> Deal | None:
    """Only a valid answer has a deal to score; `failed` and `error` results count as failed."""
    if record["status"] != ExtractionStatus.OK:
        return None
    try:
        return Deal.model_validate(record["deal"])
    except ValidationError as error:
        raise ValueError(
            f"{record['doc_id']} ({record['model']}): the stored deal no longer fits the Deal "
            f"schema ({error.error_count()} errors); rerun this configuration."
        ) from error


def _config_metrics(
    records: Sequence[Record], references: Mapping[str, Reference], prices: Prices, source: str
) -> ConfigMetrics:
    first = records[0]
    doc_counts = [
        Counts.of(score_deal(references[record["doc_id"]].deal, _predicted(record)))
        for record in records
    ]
    total = sum(doc_counts, ZERO)
    valid = sum(record["status"] == ExtractionStatus.OK for record in records)
    fixed = sum(
        record["status"] == ExtractionStatus.OK and record.get("attempts", 1) > 1
        for record in records
    )
    latencies = [record["latency_s"] for record in records]
    price = prices.get(first["model"])
    return ConfigMetrics(
        model=first["model"],
        prompt_version=first["prompt_version"],
        sampling=first["sampling"],
        run_at=first["run_at"],
        split=references[first["doc_id"]].split,
        source=source,
        docs=len(records),
        filled_accuracy=share(total.filled_correct, total.filled),
        filled_ci=bootstrap_ci(doc_counts),
        fabricated_pct=share(total.fabricated, total.empty),
        overall_pct=share(total.correct, total.total),
        valid_pct=100 * valid / len(records),
        failed_pct=100 * (len(records) - valid) / len(records),
        fixed_pct=100 * fixed / len(records),
        latency_p50_s=percentile(latencies, 50),
        latency_p95_s=percentile(latencies, LATENCY_TAIL_PCT),
        cost_per_1000_docs_usd=None if price is None else cost_per_1000_docs(records, *price),
    )


def field_counts(
    records: Iterable[Record], references: Mapping[str, Reference], group: FieldGroup
) -> dict[ConfigKey, dict[str, Counts]]:
    """Counts per configuration and leaf field over the documents of `group`. The scores are the
    ones behind the main accuracy, only not yet added up across fields."""
    scores: dict[ConfigKey, dict[str, list]] = {}
    for record in latest_runs(records, references):
        reference = references[record["doc_id"]]
        if reference.split != group.split or reference.source not in group.sources:
            continue
        by_field = scores.setdefault((*_config(record), record["run_at"]), {})
        for score in score_deal(reference.deal, _predicted(record)):
            by_field.setdefault(score.field, []).append(score)
    return {
        config: {field: Counts.of(items) for field, items in by_field.items()}
        for config, by_field in sorted(scores.items())
    }


def _rank(value: str, order: tuple[str, ...]) -> int:
    return order.index(value) if value in order else len(order)


def _config(record: Record) -> tuple[str, str, str]:
    return record["model"], record["prompt_version"], record["sampling"]


def run_label(run_at: str) -> str:
    """`2026-10-23T14:05:33+03:00` -> `2026-10-23 14:05`: the date and time of a run for a table."""
    return run_at[:RUN_LABEL_LENGTH].replace("T", " ")


def latest_runs(records: Iterable[Record], references: Mapping[str, Reference]) -> list[Record]:
    """Records of the latest run of each configuration and data split. Older runs stay in their
    files as a record; a later run on one split does not hide the earlier run on the other.
    Documents without a golden deal are left out (see `missing_references`)."""
    known = [record for record in records if record["doc_id"] in references]

    def run_key(record: Record) -> tuple:
        return _config(record), references[record["doc_id"]].split

    newest = {
        key: max(record["run_at"] for record in known if run_key(record) == key)
        for key in {run_key(record) for record in known}
    }
    return [record for record in known if record["run_at"] == newest[run_key(record)]]


def _row_keys(
    record: Record, reference: Reference, split_sources: Mapping[tuple, set[str]]
) -> Iterator[tuple]:
    """The record's own row and, when its split mixes golden sources, the `all` row."""
    split = _rank(reference.split, SPLIT_ORDER), reference.split
    yield (*split, _rank(reference.source, SOURCE_ORDER), reference.source, *_config(record))
    if len(split_sources[(reference.split, _config(record))]) > 1:
        yield (*split, len(SOURCE_ORDER) + 1, ALL_SOURCES, *_config(record))


def compute_metrics(
    records: Iterable[Record], references: Mapping[str, Reference], prices: Prices
) -> tuple[ConfigMetrics, ...]:
    """One row per configuration, split and source, dev before test, plus an `all` row for a
    split with several golden sources. Only the latest run of a configuration is scored (see
    `latest_runs`)."""
    known = latest_runs(records, references)
    split_sources: dict[tuple, set[str]] = {}
    for record in known:
        reference = references[record["doc_id"]]
        split_sources.setdefault((reference.split, _config(record)), set()).add(reference.source)
    rows = sorted(
        (
            (key, record)
            for record in known
            for key in _row_keys(record, references[record["doc_id"]], split_sources)
        ),
        key=lambda row: row[0],
    )
    return tuple(
        _config_metrics(tuple(record for _, record in group), references, prices, source=key[3])
        for key, group in groupby(rows, key=lambda row: row[0])
    )


def missing_references(
    records: Iterable[Record], references: Mapping[str, Reference]
) -> tuple[str, ...]:
    return tuple(sorted({record["doc_id"] for record in records} - references.keys()))


def latest_run_dir(base: Path) -> Path | None:
    """The newest launch folder (they are named with the start time, so by name) that holds at
    least one saved run; None when there is none."""
    folders = sorted(path for path in base.glob("*") if path.is_dir()) if base.is_dir() else []
    return next((path for path in reversed(folders) if any(path.glob("*.jsonl"))), None)


def load_records(results_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for path in sorted(results_dir.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_references(directories: Iterable[Path]) -> dict[str, Reference]:
    """Golden deals by document id: field `deal` of data/synthetic/*.json and data/golden/*.json.
    An id in two files stops the load: otherwise the later file silently replaces the earlier."""
    paths = [path for directory in directories for path in sorted(directory.glob("*.json"))]
    entries = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    ids = Counter(entry["id"] for _, entry in entries)
    duplicates = [
        f"{doc_id} in {', '.join(str(path) for path, entry in entries if entry['id'] == doc_id)}"
        for doc_id, count in ids.items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"the same id in several reference files: {'; '.join(duplicates)}")
    return {
        entry["id"]: Reference(
            deal=Deal.model_validate(entry["deal"]),
            split=entry["split"],
            source=entry.get("source", DEFAULT_SOURCE),
        )
        for _, entry in entries
    }


def load_price_table(path: Path = MODELS_PATH) -> dict[str, tuple[float, float]]:
    """Prices by model id from every section of configs/models.yaml."""
    sections = yaml.safe_load(path.read_text(encoding="utf-8")).values()
    return {
        section["model"]: (
            float(section["price_input_per_mtok"]), float(section["price_output_per_mtok"])
        )
        for section in sections
    }


def _pct(value: float | None) -> str:
    return NO_VALUE if value is None else f"{value:.1f}"


def _row(metrics: ConfigMetrics) -> tuple[str, ...]:
    interval = "" if metrics.filled_ci is None else (
        f" ({metrics.filled_ci[0]:.1f}–{metrics.filled_ci[1]:.1f})"
    )
    cost = metrics.cost_per_1000_docs_usd
    return (
        metrics.model, metrics.prompt_version, metrics.sampling, run_label(metrics.run_at),
        f"{metrics.split} · {metrics.source}", str(metrics.docs),
        _pct(metrics.filled_accuracy) + interval, _pct(metrics.fabricated_pct),
        _pct(metrics.overall_pct), _pct(metrics.valid_pct), _pct(metrics.failed_pct),
        _pct(metrics.fixed_pct), f"{metrics.latency_p50_s:.1f}", f"{metrics.latency_p95_s:.1f}",
        NO_VALUE if cost is None else f"{cost:.2f}",
    )


def format_table(metrics: Iterable[ConfigMetrics]) -> str:
    rows = (HEADER, tuple("---" for _ in HEADER), *(_row(item) for item in metrics))
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _field_cell(counts: Counts | None) -> str:
    if counts is None:
        return NO_VALUE
    accuracy = share(counts.filled_correct, counts.filled)
    shown = NO_VALUE if accuracy is None else (
        f"{accuracy:.1f} ({counts.filled_correct}/{counts.filled})"
    )
    return f"{shown} · вигад. {counts.fabricated}"


def _group_table(group: FieldGroup, counts: Mapping[ConfigKey, Mapping[str, Counts]]) -> str:
    fields = tuple(dict.fromkeys(field for by_field in counts.values() for field in by_field))
    totals = {config: sum(by_field.values(), ZERO) for config, by_field in counts.items()}
    widest_config = max(totals, key=lambda config: totals[config].total)  # base of golden columns
    widest = totals[widest_config]
    header = (
        *FIELD_HEADER,
        *(" · ".join((*config[:-1], run_label(config[-1]))) for config in counts),
    )
    rows = [
        (
            field,
            str(counts[widest_config].get(field, ZERO).filled),
            str(counts[widest_config].get(field, ZERO).empty),
            *(_field_cell(by_field.get(field)) for by_field in counts.values()),
        )
        for field in fields
    ]
    rows.append(
        (FIELD_TOTAL, str(widest.filled), str(widest.empty),
         *(_field_cell(totals[config]) for config in counts))
    )
    lines = (header, tuple("---" for _ in header), *rows)
    title = f"## {group.title} ({group.split} · {' + '.join(group.sources)})"
    table = "\n".join("| " + " | ".join(row) + " |" for row in lines)
    return f"{title}\n\n{table}\n\n{FIELD_LEGEND}"


def format_field_tables(
    records: Iterable[Record],
    references: Mapping[str, Reference],
    groups: Iterable[FieldGroup] = FIELD_GROUPS,
) -> str:
    """One table per group of documents: a row per field, a column per configuration. A group
    nobody answered is left out."""
    records = tuple(records)
    sections = (
        _group_table(group, counts)
        for group in groups
        if (counts := field_counts(records, references, group))
    )
    return "\n\n".join(sections)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score stage-1 results against golden deals.")
    parser.add_argument(
        "--results-dir", type=Path,
        help="Folder of one launch (results/stage1_2/<time>/); default: the latest one.",
    )
    parser.add_argument(
        "--references", type=Path, nargs="+", default=list(REFERENCE_DIRS),
        help="Folders with golden *.json files (fields id, split, deal).",
    )
    parser.add_argument("--models", type=Path, default=MODELS_PATH, help="Prices per model.")
    parser.add_argument("--output", type=Path, help="Also write the table to this file.")
    parser.add_argument(
        "--by-field-output", type=Path, help="Also write the per-field accuracy tables here."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir or latest_run_dir(RESULTS_DIR) or RESULTS_DIR
    records = load_records(results_dir)
    if not records:
        print(f"No results in {results_dir}; run `python -m src.extraction.run` first.",
              file=sys.stderr)
        return 2
    try:
        references = load_references(args.references)
        missing = missing_references(records, references)
        if missing:
            print(f"No golden deal for {', '.join(missing)}: left out of the metrics.",
                  file=sys.stderr)
        metrics = compute_metrics(records, references, load_price_table(args.models))
        field_tables = format_field_tables(records, references)
    except ValueError as error:
        print(f"Stopped: {error}", file=sys.stderr)
        return 1
    table = format_table(metrics)
    print(table)
    if args.output:
        args.output.write_text(table + "\n", encoding="utf-8")
    if args.by_field_output:
        args.by_field_output.write_text(field_tables + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
