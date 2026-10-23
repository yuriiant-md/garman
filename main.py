"""Stage 1 in one command: every model x prompt x sampling on all texts, then the metrics table.

    make run                                        # the same as the next line
    uv run --env-file .env python main.py

Every run asks the models and saves its answers to a new file in results/stage1 named with the
model, prompt version, sampling and the time of the run. The tables take the latest run of each
configuration; `make metrics` rebuilds them from the saved runs without an API key or Ollama.
A configuration whose model is unavailable is skipped with the reason.
"""

import argparse
import sys
from itertools import product
from pathlib import Path

from src.extraction import metrics, run

ROOT = Path(__file__).resolve().parent
MODELS = ("claude", "lapa")
PROMPTS = (ROOT / "prompts" / "extract_v1.md", ROOT / "prompts" / "extract_v2.md")
SAMPLINGS = ("precise",)  # varied, with 3 repeats on the golden set, comes in Е1.4
METRICS_PATH = ROOT / "results" / "stage1" / "metrics.md"
BY_FIELD_PATH = ROOT / "results" / "stage1" / "metrics_by_field.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the whole stage-1 matrix and its metrics.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    not_run = []
    for model, prompt, sampling in product(MODELS, PROMPTS, SAMPLINGS):
        name = f"{model} · {prompt.stem} · {sampling}"
        print(f"\n== {name}")
        config = ["--model", model, "--prompt", str(prompt), "--sampling", sampling]
        if run.main([*config, "--split", "all"]) != 0:
            not_run.append(name)
    print()
    code = metrics.main(
        ["--output", str(METRICS_PATH), "--by-field-output", str(BY_FIELD_PATH)]
    )
    if not_run:
        print("\nNot finished, see the messages above: " + ", ".join(not_run), file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
