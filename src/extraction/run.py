"""Run an extractor over the stage-1 texts and save one JSON line per document.

Answers already saved for the same document, prompt file and sampling are reused; `--fresh`
asks the model again for the chosen split and keeps the saved answers of the other split.

    uv run --env-file .env python -m src.extraction.run --split dev
    uv run python -m src.extraction.run --model lapa   # local model through Ollama, no API key
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Protocol

import anthropic
import ollama
import yaml

from src.extraction.anthropic_extractor import AnthropicConfig, AnthropicExtractor
from src.extraction.metrics import cost_per_1000_docs, cost_usd
from src.extraction.ollama_extractor import OllamaConfig, OllamaExtractor
from src.extraction.prompts import Prompt, load_prompt
from src.extraction.result import ExtractionResult, ExtractionStatus
from src.extraction.sampling import Sampling, load_sampling

ROOT = Path(__file__).resolve().parents[2]
TEXTS_DIR = ROOT / "data" / "texts"
RESULTS_DIR = ROOT / "results" / "stage1"
MODELS_PATH = ROOT / "configs" / "models.yaml"
DEFAULT_PROMPT = ROOT / "prompts" / "extract_v2.md"
DEV_PREFIX = "synthetic_"  # dev = synthetic texts; everything else (golden set) is test
SPLITS = ("dev", "test", "all")
API_KEY_VARIABLE = "ANTHROPIC_API_KEY"
PARTIAL_SUFFIX = ".partial"
MODEL_SECTIONS = {"claude": "anthropic", "lapa": "ollama"}  # --model -> section of models.yaml


class Extractor(Protocol):
    def extract(self, text: str, prompt: Prompt, sampling: Sampling) -> ExtractionResult: ...


def text_paths(texts_dir: Path, split: str) -> list[Path]:
    paths = sorted(texts_dir.glob("*.txt"))
    if split == "dev":
        return [path for path in paths if path.stem.startswith(DEV_PREFIX)]
    if split == "test":
        return [path for path in paths if not path.stem.startswith(DEV_PREFIX)]
    return paths


def result_path(results_dir: Path, model: str, prompt_version: str, sampling_name: str) -> Path:
    return results_dir / f"{model}__{prompt_version}__{sampling_name}.jsonl"


def load_cached(output: Path, prompt: Prompt, sampling: Sampling) -> dict[str, dict[str, object]]:
    """Saved answers a rerun may reuse, by document id: same prompt file (its hash) and sampling
    name. An `error` result holds no answer, so that document is asked again."""
    if not output.exists():
        return {}
    lines = output.read_text(encoding="utf-8").splitlines()
    records = (json.loads(line) for line in lines if line.strip())
    return {
        record["doc_id"]: record
        for record in records
        if record["prompt_sha256"] == prompt.sha256
        and record["sampling"] == sampling.name
        and record["status"] != ExtractionStatus.ERROR
    }


def run_extraction(
    extractor: Extractor,
    paths: list[Path],
    prompt: Prompt,
    sampling: Sampling,
    output: Path,
    cached: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Lines go to a .partial file as each document finishes; it replaces `output` only when the
    whole run succeeds, so a failed rerun never wipes the previous results. A document with a
    cached answer is not sent to the model; cached answers of documents outside `paths` stay in
    the file, so a run on one split keeps the other split's answers."""
    cached = cached or {}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + PARTIAL_SUFFIX)
    records = []
    with partial.open("w", encoding="utf-8") as file:
        for path in paths:
            record = cached.get(path.stem)
            if record is None:
                result = extractor.extract(path.read_text(encoding="utf-8"), prompt, sampling)
                record = {"doc_id": path.stem, **result.model_dump(mode="json")}
                print(f"{path.stem}: {result.status} in {result.latency_s:.1f} s")
            else:
                print(f"{path.stem}: cached")
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            records.append(record)
        outside = sorted(set(cached) - {path.stem for path in paths})
        file.writelines(json.dumps(cached[doc], ensure_ascii=False) + "\n" for doc in outside)
    partial.replace(output)
    return records


def summarize(
    records: list[dict[str, object]], price_input: float, price_output: float
) -> dict[str, object]:
    counts = Counter(record["status"] for record in records)
    input_tokens = sum(record["input_tokens"] for record in records)
    output_tokens = sum(record["output_tokens"] for record in records)
    return {
        "statuses": {status.value: counts.get(status.value, 0) for status in ExtractionStatus},
        "median_latency_s": statistics.median(record["latency_s"] for record in records),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd(input_tokens, output_tokens, price_input, price_output),
        "cost_per_1000_docs_usd": cost_per_1000_docs(records, price_input, price_output),
    }


def load_anthropic_config(path: Path = MODELS_PATH) -> AnthropicConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))["anthropic"]
    return AnthropicConfig(
        model=values["model"],
        timeout_s=float(values["timeout_s"]),
        max_retries=int(values["max_retries"]),
        max_tokens=int(values["max_tokens"]),
    )


def load_ollama_config(path: Path = MODELS_PATH) -> OllamaConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))["ollama"]
    return OllamaConfig(
        model=values["model"], timeout_s=float(values["timeout_s"]), num_ctx=int(values["num_ctx"])
    )


def load_prices(path: Path = MODELS_PATH, section: str = "anthropic") -> tuple[float, float]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))[section]
    return float(values["price_input_per_mtok"]), float(values["price_output_per_mtok"])


def configured_model(model: str) -> str:
    """The model id from configs/models.yaml for a --model choice; it names the results file."""
    return load_ollama_config().model if model == "lapa" else load_anthropic_config().model


def build_extractor(model: str) -> Extractor:
    """Built only when some document needs the model: the Claude client requires the API key."""
    if model == "lapa":
        return OllamaExtractor(load_ollama_config())
    return AnthropicExtractor(load_anthropic_config())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Deal JSON from stage-1 texts.")
    parser.add_argument("--model", choices=tuple(MODEL_SECTIONS), default="claude")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--texts-dir", type=Path, default=TEXTS_DIR)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--sampling", default="precise", help="Name from configs/sampling.yaml.")
    parser.add_argument(
        "--fresh", action="store_true", help="Ask the model again for every text of the split."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = text_paths(args.texts_dir, args.split)
    if not paths:
        print(f"No {args.split} texts in {args.texts_dir}; see data/README.md.", file=sys.stderr)
        return 2
    try:
        sampling = load_sampling(args.sampling)
    except KeyError as error:
        print(error.args[0], file=sys.stderr)
        return 2
    prompt = load_prompt(args.prompt)
    output = result_path(RESULTS_DIR, configured_model(args.model), prompt.version, sampling.name)
    cached = load_cached(output, prompt, sampling)
    if args.fresh:  # ask this split again; answers of the other split stay in the file
        asked = {path.stem for path in paths}
        cached = {doc: record for doc, record in cached.items() if doc not in asked}
    if all(path.stem in cached for path in paths):
        print(f"All {len(paths)} answers are cached in {output.relative_to(ROOT)}")
        return 0
    if args.model == "claude" and not os.environ.get(API_KEY_VARIABLE):
        print(f"{API_KEY_VARIABLE} is not set: put it in .env and run with "
              "`uv run --env-file .env python -m src.extraction.run`.", file=sys.stderr)
        return 2
    try:
        records = run_extraction(
            build_extractor(args.model), paths, prompt, sampling, output, cached
        )
    except anthropic.APIStatusError as error:
        print(f"Stopped: API error {error.status_code}: {error.message}", file=sys.stderr)
        return 1
    except anthropic.AnthropicError as error:
        print(f"Stopped: {error}", file=sys.stderr)
        return 1
    except ConnectionError as error:
        print(f"Stopped: Ollama is not reachable ({error}); start it with `ollama serve`.",
              file=sys.stderr)
        return 1
    except ollama.ResponseError as error:
        print(f"Stopped: Ollama error {error.status_code}: {error.error}", file=sys.stderr)
        return 1
    summary = summarize(records, *load_prices(section=MODEL_SECTIONS[args.model]))
    print(f"Saved {len(records)} results to {output.relative_to(ROOT)}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
