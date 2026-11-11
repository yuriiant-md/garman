"""Write the sheet for the author's manual marks of a saved run: free, no model, no search.

The sheet holds what the judge judges in the run, without its verdicts (src/rag/calibration.py):
the author fills it, then scripts/calibrate_judge.py compares the marks with judge.jsonl.
    uv run python -m scripts.judge_sheet --seen q01,q05                # the newest run
    uv run python -m scripts.judge_sheet results/stage2/answers/2026-11-10T10-08-19

--seen names the questions whose verdicts the author saw before marking: their agreement is
counted apart. A sheet that exists is kept: it may hold marks; --force writes over it.

Exit code: 0 — the sheet is written; 2 — it is not (the sheet exists, an unknown --seen id, or a
run that cannot be judged: the same refusals as scripts/judge_answers.py).
"""

import argparse
import sys
from pathlib import Path

from scripts.ask_rag import RESULTS_DIR, SETUP_ERRORS
from scripts.judge_answers import load_pairs
from scripts.score_answers import latest_run
from src.rag.answer_eval import ANSWER_QUESTIONS_PATH, SEARCH_QUESTIONS_PATH
from src.rag.calibration import MARKS_PATH, build_sheet, dump_sheet, unmarked


def parse_seen(value: str) -> frozenset[str]:
    """«q01,q05» → {q01, q05}; an empty id is a typo, not «nothing»."""
    if not value:
        return frozenset()
    ids = [part.strip() for part in value.split(",")]
    if "" in ids:
        raise ValueError(f"--seen has an empty id: {value!r}")
    return frozenset(ids)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", type=Path, help="a run's folder; the newest by default")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--questions", type=Path, default=ANSWER_QUESTIONS_PATH)
    parser.add_argument("--search-questions", type=Path, default=SEARCH_QUESTIONS_PATH)
    parser.add_argument("--out", type=Path, default=MARKS_PATH)
    parser.add_argument("--seen", default="", help="ids whose verdicts were seen, comma-separated")
    parser.add_argument("--force", action="store_true", help="write over a sheet that exists")
    args = parser.parse_args(argv)
    args.limit = None  # load_pairs takes every question
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.out.exists() and not args.force:
            raise ValueError(f"{args.out} exists and may hold marks; --force writes over it")
        seen = parse_seen(args.seen)
        run_dir: Path = args.run or latest_run(args.results_dir)
        sheet = build_sheet(run_dir.name, load_pairs(args, run_dir), seen)
    except SETUP_ERRORS as error:
        print(f"cannot write the sheet: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump_sheet(sheet, args.out)
    claims = sum(len(entry.claims) for entry in sheet.answers)
    print(f"{len(sheet.answers)} answers, {len(unmarked(sheet))} marks to fill "
          f"({claims} claims), {len(seen)} seen → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
