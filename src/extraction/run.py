"""Run an extractor over the stage-1 texts and save one JSON line per document.

Every run asks the model and is saved in a folder named with the time of the launch, to a file
named with the model, the prompt version and the sampling, so earlier launches stay as a record.
Each line carries the same `run_at`.

    uv run --env-file .env python -m src.extraction.run --split dev
    uv run --env-file .env python -m src.extraction.run --model gpt-4o   # GPT-4o through OpenRouter
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

import anthropic
import openai
import yaml

from src.extraction.anthropic_extractor import AnthropicConfig, AnthropicExtractor
from src.extraction.metrics import RESULTS_DIR, cost_per_1000_docs, cost_usd
from src.extraction.openrouter_extractor import OpenRouterConfig, OpenRouterExtractor
from src.extraction.prompts import Prompt, load_prompt
from src.extraction.result import ExtractionResult, ExtractionStatus, Record
from src.extraction.sampling import DEFAULT_SAMPLING, Sampling, load_sampling

ROOT = Path(__file__).resolve().parents[2]
TEXTS_DIR = ROOT / "data" / "texts"
MODELS_PATH = ROOT / "configs" / "models.yaml"
DEFAULT_PROMPT = ROOT / "prompts" / "extract_v2.md"
DEV_PREFIX = "synthetic_"  # dev = synthetic texts; everything else (golden set) is test
SPLITS = ("dev", "test", "all")
MAIN_MODEL = "haiku-5.5"  # the production choice, REPORT.md, stage 1, section 6
# --model keys: the sections of models.yaml in file order, each named after its model
MODELS: tuple[str, ...] = tuple(yaml.safe_load(MODELS_PATH.read_text(encoding="utf-8")))
ANTHROPIC, OPENROUTER = "anthropic", "openrouter"  # the `provider` of a models.yaml section
API_KEY_VARIABLES = {ANTHROPIC: "ANTHROPIC_API_KEY", OPENROUTER: "OPENROUTER_API_KEY"}
PRECISE_SAMPLING = "precise"  # without --sampling, for a model that takes temperature/top_p
PARTIAL_SUFFIX = ".partial"
RUN_STAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"  # colons are not allowed in file names on some systems


class Extractor(Protocol):
    def extract(self, text: str, prompt: Prompt, sampling: Sampling) -> ExtractionResult: ...


def first_per_source(paths: list[Path], count: int) -> list[Path]:
    """The first `count` documents of each source; a source is the file name before its number."""
    taken: Counter[str] = Counter()
    kept = []
    for path in paths:
        source = path.stem.rsplit("_", 1)[0]
        if taken[source] < count:
            taken[source] += 1
            kept.append(path)
    return kept


def text_paths(texts_dir: Path, split: str, per_source: int | None = None) -> list[Path]:
    paths = sorted(texts_dir.glob("*.txt"))
    if split == "dev":
        paths = [path for path in paths if path.stem.startswith(DEV_PREFIX)]
    elif split == "test":
        paths = [path for path in paths if not path.stem.startswith(DEV_PREFIX)]
    return paths if per_source is None else first_per_source(paths, per_source)


def current_time() -> datetime:
    """The start of a run in local time with its UTC offset."""
    return datetime.now().astimezone()


def run_dir_path(results_dir: Path, started: datetime) -> Path:
    """The folder of one launch: its runs, and the tables built from them, live together."""
    return results_dir / started.strftime(RUN_STAMP_FORMAT)


def result_path(run_dir: Path, model: str, prompt_version: str, sampling_name: str) -> Path:
    file_model = model.replace("/", "_")  # openai/gpt-4o would otherwise open a subfolder
    return run_dir / f"{file_model}__{prompt_version}__{sampling_name}.jsonl"


def run_extraction(
    extractor: Extractor,
    paths: list[Path],
    prompt: Prompt,
    sampling: Sampling,
    output: Path,
    started: datetime,
) -> list[Record]:
    """Lines go to a .partial file as each document finishes; it becomes `output` only when the
    whole run succeeds, so the metrics never see half a run. Every line carries `run_at`."""
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + PARTIAL_SUFFIX)
    run_at = started.isoformat(timespec="seconds")
    records: list[Record] = []
    with partial.open("w", encoding="utf-8") as file:
        for path in paths:
            result = extractor.extract(path.read_text(encoding="utf-8"), prompt, sampling)
            # model_dump is untyped; it holds exactly the ExtractionResult fields
            record = cast(
                Record, {"doc_id": path.stem, "run_at": run_at, **result.model_dump(mode="json")}
            )
            print(f"{path.stem}: {result.status} in {result.latency_s:.1f} s")
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            records.append(record)
    partial.replace(output)
    return records


def summarize(
    records: list[Record], price_input: float, price_output: float
) -> dict[str, object]:
    counts = Counter(record["status"] for record in records)
    input_tokens = sum(record["input_tokens"] for record in records)
    output_tokens = sum(record["output_tokens"] for record in records)
    return {
        "statuses": {status.value: counts.get(status.value, 0) for status in ExtractionStatus},
        "median_latency_s": statistics.median(record["latency_s"] for record in records),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        # OpenRouter may leave usage out: such answers are paid for, but the cost below misses them
        "answers_without_usage": sum(
            record["status"] != ExtractionStatus.ERROR and record["input_tokens"] == 0
            for record in records
        ),
        "cost_usd": cost_usd(input_tokens, output_tokens, price_input, price_output),
        "cost_per_1000_docs_usd": cost_per_1000_docs(records, price_input, price_output),
    }


def load_anthropic_config(path: Path = MODELS_PATH, section: str = MAIN_MODEL) -> AnthropicConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))[section]
    return AnthropicConfig(
        model=values["model"],
        timeout_s=float(values["timeout_s"]),
        max_retries=int(values["max_retries"]),
        max_tokens=int(values["max_tokens"]),
        effort=values.get("effort"),
        accepts_sampling=bool(values.get("sampling", True)),
    )


def takes_sampling(model: str, path: Path = MODELS_PATH) -> bool:
    """False for a model that rejects temperature/top_p: it runs with the default sampling only."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))[model]
    return bool(values.get("sampling", True))


def default_sampling(model: str) -> str:
    """Without --sampling: precise (as good as varied in REPORT.md, and reproducible) for a model
    that takes temperature/top_p, the default sampling for one that rejects them."""
    return PRECISE_SAMPLING if takes_sampling(model) else DEFAULT_SAMPLING


def load_openrouter_config(path: Path = MODELS_PATH, *, section: str) -> OpenRouterConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))[section]
    return OpenRouterConfig(
        model=values["model"],
        timeout_s=float(values["timeout_s"]),
        max_retries=int(values["max_retries"]),
        max_tokens=int(values["max_tokens"]),
    )


def load_prices(path: Path = MODELS_PATH, section: str = MAIN_MODEL) -> tuple[float, float]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))[section]
    return float(values["price_input_per_mtok"]), float(values["price_output_per_mtok"])


def model_provider(model: str, path: Path = MODELS_PATH) -> str:
    """The `provider` of a models.yaml section: it picks the extractor and the API key."""
    provider = yaml.safe_load(path.read_text(encoding="utf-8"))[model].get("provider")
    if provider not in API_KEY_VARIABLES:
        raise ValueError(
            f"{path}: section {model} needs provider: {' or '.join(API_KEY_VARIABLES)}, "
            f"not {provider!r}"
        )
    return str(provider)


def configured_model(model: str) -> str:
    """The model id from configs/models.yaml for a --model choice; it names the results file."""
    if model_provider(model) == OPENROUTER:
        return load_openrouter_config(section=model).model
    return load_anthropic_config(section=model).model


def build_extractor(model: str) -> Extractor:
    """Built only when some document needs the model: API clients require their keys."""
    if model_provider(model) == OPENROUTER:
        return OpenRouterExtractor(load_openrouter_config(section=model))
    return AnthropicExtractor(load_anthropic_config(section=model))


def positive_int(value: str) -> int:
    """The argparse type of --per-source: 0 or a negative count would leave no texts to run."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive whole number, not {value}")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Deal JSON from stage-1 texts.")
    parser.add_argument("--model", choices=MODELS, default=MAIN_MODEL)
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument(
        "--per-source",
        type=positive_int,
        help="Only the first N texts of each source (a quick check).",
    )
    parser.add_argument("--texts-dir", type=Path, default=TEXTS_DIR)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--sampling",
        help=f"Name from configs/sampling.yaml; default: {PRECISE_SAMPLING}, or "
        f"{DEFAULT_SAMPLING} for a model that rejects temperature and top_p.",
    )
    parser.add_argument(
        "--run-dir", type=Path, help="Folder for the run; default: a new one named with the time."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = text_paths(args.texts_dir, args.split, args.per_source)
    if not paths:
        print(f"No {args.split} texts in {args.texts_dir}; see data/README.md.", file=sys.stderr)
        return 2
    try:
        sampling = load_sampling(args.sampling or default_sampling(args.model))
    except KeyError as error:
        print(error.args[0], file=sys.stderr)
        return 2
    if sampling.name != DEFAULT_SAMPLING and not takes_sampling(args.model):
        print(f"{configured_model(args.model)} rejects temperature and top_p: run it with "
              f"--sampling {DEFAULT_SAMPLING}.", file=sys.stderr)
        return 2
    try:
        prompt = load_prompt(args.prompt)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    started = current_time()
    run_dir = (args.run_dir or run_dir_path(RESULTS_DIR, started)).resolve()
    output = result_path(run_dir, configured_model(args.model), prompt.version, sampling.name)
    shown = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output  # --run-dir /tmp/x
    if output.exists():
        print(f"{shown} already exists: give the prompt a new version in its header "
              "or choose another --run-dir.", file=sys.stderr)
        return 2
    key = API_KEY_VARIABLES[model_provider(args.model)]
    if not os.environ.get(key):
        print(f"{key} is not set: put it in .env and run with "
              "`uv run --env-file .env python -m src.extraction.run`.", file=sys.stderr)
        return 2
    try:
        records = run_extraction(
            build_extractor(args.model), paths, prompt, sampling, output, started
        )
    except (anthropic.APIStatusError, openai.APIStatusError) as error:
        print(f"Stopped: API error {error.status_code}: {error.message}", file=sys.stderr)
        return 1
    except (anthropic.AnthropicError, openai.OpenAIError) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        return 1
    summary = summarize(records, *load_prices(section=args.model))
    print(f"Saved {len(records)} results to {shown}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
