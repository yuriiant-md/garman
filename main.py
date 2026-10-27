"""Stage 1 in one command: every model x prompt x sampling on all texts, then the metrics table.

    make run                                        # the same as the next line
    uv run --env-file .env python main.py

Stage 1.3 matrix, on the schema with grades: Claude Haiku 5.5 (the main model), and Claude
Haiku 4.5 and GPT-4o (through OpenRouter) to compare with. Haiku 5.5 rejects temperature/top_p,
so it runs with the default sampling only. Earlier matrices stay in results/stage1/ (with the local Lapa) and results/stage1_2/ (with
the local Qwen, on the schema before grades); the local models left the code in stage 1.3.
Every launch asks the models and saves everything in a new folder results/stage1_3/<start time>/:
one file per configuration (model, prompt version, sampling) and the two metrics tables built
from them. Earlier folders are never touched, so launches can be compared; `make metrics`
rebuilds the tables of the latest folder without an API key. A configuration whose
model is unavailable is skipped with the reason. `--models haiku-5.5` runs only the named models.
"""

import argparse
import sys
from itertools import product
from pathlib import Path

from src.extraction import metrics, run
from src.extraction.sampling import DEFAULT_SAMPLING

ROOT = Path(__file__).resolve().parent
MODELS = run.MODELS  # the sections of configs/models.yaml: the main model first, then the others
PROMPTS = (ROOT / "prompts" / "extract_v1.md", ROOT / "prompts" / "extract_v2.md")
SAMPLINGS = ("precise", "varied")  # for models that accept them; the others run DEFAULT_SAMPLING
RESULTS_DIR = ROOT / "results" / "stage1_3"
SMOKE_DIR = ROOT / "results" / "smoke"  # quick checks; kept apart so `make metrics` ignores them
METRICS_FILE = "metrics.md"
BY_FIELD_FILE = "metrics_by_field.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the whole stage-1 matrix and its metrics.")
    parser.add_argument(
        "--per-source",
        type=run.positive_int,
        help="Quick check: only the first N texts of each source, saved in results/smoke/.",
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=MODELS, help="Run only these models."
    )
    return parser.parse_args(argv)


def configurations(models: tuple[str, ...]) -> list[tuple[str, Path, str]]:
    """Model, prompt and sampling of every run; a model without sampling runs the default."""
    return [
        (model, prompt, sampling)
        for model in models
        for prompt, sampling in product(
            PROMPTS, SAMPLINGS if run.takes_sampling(model) else (DEFAULT_SAMPLING,)
        )
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = RESULTS_DIR if args.per_source is None else SMOKE_DIR
    limit = [] if args.per_source is None else ["--per-source", str(args.per_source)]
    run_dir = run.run_dir_path(results_dir, run.current_time())
    not_run = []
    for model, prompt, sampling in configurations(tuple(args.models)):
        name = f"{model} · {prompt.stem} · {sampling}"
        print(f"\n== {name}")
        config = ["--model", model, "--prompt", str(prompt), "--sampling", sampling]
        if run.main([*config, "--split", "all", "--run-dir", str(run_dir), *limit]) != 0:
            not_run.append(name)
    print()
    code = metrics.main(
        [
            "--results-dir", str(run_dir),
            "--output", str(run_dir / METRICS_FILE),
            "--by-field-output", str(run_dir / BY_FIELD_FILE),
        ]
    )
    if not_run:
        print("\nNot finished, see the messages above: " + ", ".join(not_run), file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
