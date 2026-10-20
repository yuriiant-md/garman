"""Stage 1 in one command: every model x prompt x sampling on all texts, then the metrics table.

    make run                                        # the same as the next line
    uv run --env-file .env python main.py
    uv run --env-file .env python main.py --fresh   # ask the models again, ignore saved answers

Answers already in results/stage1 for the same document, prompt file and sampling are reused,
so a fresh clone rebuilds results/stage1/metrics.md from the committed answers without an API key
or Ollama. A configuration whose model is needed but unavailable is skipped with the reason.
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the whole stage-1 matrix and its metrics.")
    parser.add_argument(
        "--fresh", action="store_true", help="Ask the models again even for saved answers."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    not_run = []
    for model, prompt, sampling in product(MODELS, PROMPTS, SAMPLINGS):
        name = f"{model} · {prompt.stem} · {sampling}"
        print(f"\n== {name}")
        config = ["--model", model, "--prompt", str(prompt), "--sampling", sampling]
        if run.main([*config, "--split", "all", *(["--fresh"] if args.fresh else [])]) != 0:
            not_run.append(name)
    print()
    code = metrics.main(["--output", str(METRICS_PATH)])
    if not_run:
        print("\nNot finished, see the messages above: " + ", ".join(not_run), file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
