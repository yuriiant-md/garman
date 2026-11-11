"""Calibrate the stage 2 judge on the author's manual marks: free, no model, no search.

Reads the filled sheet (data/rag/judge_marks.yaml, written by scripts/judge_sheet.py) and
judge.jsonl of the run it names, compares every mark with the judge's verdict on the same thing
(src/rag/calibration.py) and writes calibration.md next to judge.jsonl: the agreement on correct
and faithful, over all answers and over unseen and seen apart, and every disagreement with the
judge's reason. «The judge says yes, the author no» is a missed error, the dangerous kind: it is
listed apart from «the judge says no, the author yes», a judge too strict.

    uv run python -m scripts.calibrate_judge                 # the run named in the sheet
    uv run python -m scripts.calibrate_judge results/stage2/answers/2026-11-10T10-08-19

Exit code: 0 — both agreements reach agreement_min of configs/judge.yaml: the judge may be used;
1 — some does not; 2 — nothing is compared (an empty mark, a run other than the sheet's, no
judge.jsonl or no record in it for a marked answer, a claim count that differs).
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from scripts.ask_rag import MARKS, RESULTS_DIR, SETUP_ERRORS, _table
from scripts.judge_answers import JUDGEMENTS_NAME, JudgeRecord
from src.rag.answer_scoring import Gate, Ratio
from src.rag.calibration import (
    MARKS_PATH,
    Check,
    Compared,
    JudgeVerdicts,
    MarkSheet,
    agreement,
    calibrate,
    calibration_gates,
    load_sheet,
)
from src.rag.documents import read_jsonl
from src.rag.judge import JUDGE_CONFIG_PATH, Judgement, load_judge_config

CALIBRATION_NAME = "calibration.md"
GROUPS = ((None, "усі"), (False, "не бачені"), (True, "бачені"))  # seen filter, column name

type Key = tuple[str, Check, int | None]  # answer id, check, claim number (None for correct)


def verdicts_of(record: JudgeRecord) -> JudgeVerdicts:
    """The judge's verdicts on one answer; None for each thing it gave no verdict on."""
    correct = record.correct.verdict if record.correct else None
    faithful = record.faithful.verdict if record.faithful else None
    by_n = {item.n: item.supported for item in faithful.verdicts} if faithful else {}
    return JudgeVerdicts(
        correct=correct.correct if correct else None,
        supported=tuple(by_n.get(n) for n in range(1, record.claims + 1)),
    )


def _failure(judgement: Judgement) -> str:
    return f"{judgement.status.value}: {judgement.error or 'без пояснення'}"


def reasons_of(records: Sequence[JudgeRecord]) -> dict[Key, str]:
    """Why the judge said what it said, or why it said nothing, for every thing it judged."""
    found: dict[Key, str] = {}
    for record in records:
        if record.correct:
            verdict = record.correct.verdict
            found[(record.id, Check.CORRECT, None)] = (
                verdict.reason if verdict else _failure(record.correct)
            )
        if record.faithful:
            for n in range(1, record.claims + 1):  # a claim the verdict skips keeps the failure
                found[(record.id, Check.FAITHFUL, n)] = _failure(record.faithful)
            if record.faithful.verdict:
                found.update({
                    (record.id, Check.FAITHFUL, item.n): item.reason
                    for item in record.faithful.verdict.verdicts
                })
    return found


@dataclass(frozen=True)
class Calibrated:
    sheet: MarkSheet
    marks: Path
    records: tuple[JudgeRecord, ...]
    items: tuple[Compared, ...]
    gates: tuple[Gate, ...]
    limit: float


def _ratio(ratio: Ratio) -> str:
    return f"{ratio.hit} з {ratio.total} ({ratio.rate:.0%})" if ratio.total else "—"


def _judge_line(records: Sequence[JudgeRecord]) -> str:
    judgements = [j for r in records for j in (r.correct, r.faithful) if j is not None]
    if not judgements:
        return "Суддя: —"
    versions = sorted({j.prompt_version for j in judgements})
    return f"Суддя: {judgements[0].model} · версії промптів: {', '.join(versions)}"


def _item_line(item: Compared, reasons: dict[Key, str]) -> str:
    seen = " (бачене)" if item.seen else ""
    what = "correct" if item.n is None else f"твердження {item.n}"
    reason = reasons.get((item.id, item.check, item.n), "—").replace("\n", " ")
    return f"- {item.id}{seen}, {what}: {reason}"


def _section(title: str, chosen: Sequence[Compared], reasons: dict[Key, str]) -> list[str]:
    return [f"## {title}", "", *([_item_line(i, reasons) for i in chosen] or ["Немає."]), ""]


def _disagreements(calibrated: Calibrated) -> list[str]:
    items, reasons = calibrated.items, reasons_of(calibrated.records)
    missed = [i for i in items if i.judge is True and not i.author]
    strict = [i for i in items if i.judge is False and i.author]
    silent = [i for i in items if i.judge is None]
    return [
        *_section("Суддя «так», автор «ні»: пропущена помилка", missed, reasons),
        *_section("Суддя «ні», автор «так»: надто суворий", strict, reasons),
        *_section("Без вердикту судді", silent, reasons),
    ]


def render_calibration(calibrated: Calibrated) -> str:
    sheet, items = calibrated.sheet, calibrated.items
    seen = sum(entry.seen for entry in sheet.answers)
    agreement_rows = [
        [str(check), *(_ratio(agreement(items, check, group)) for group, _ in GROUPS)]
        for check in Check
    ]
    gate_rows = [
        [gate.name, f"{gate.measured:.0%}", f"≥ {gate.limit:.0%}", MARKS[gate.passed]]
        for gate in calibrated.gates
    ]
    return "\n".join([
        f"# Калібрування судді: прогін {sheet.run}",
        "",
        _judge_line(calibrated.records),
        "",
        (
            f"Ручна розмітка: {calibrated.marks} · {len(sheet.answers)} відповідей, з них "
            f"{seen} бачені (вердикт судді автор бачив до розмітки) · поріг збігу "
            f"≥ {calibrated.limit:.0%} на обох перевірках"
        ),
        "",
        "## Збіг із ручною розміткою",
        "",
        *_table(["Перевірка", *(name for _, name in GROUPS)], agreement_rows),
        "",
        *_table(["Метрика", "Виміряно", "Вимога", "Пройдено"], gate_rows),
        "",
        *_disagreements(calibrated),
    ])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", type=Path, help="a run's folder; the sheet's by default")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--marks", type=Path, default=MARKS_PATH)
    parser.add_argument("--config", type=Path, default=JUDGE_CONFIG_PATH)
    return parser.parse_args(argv)


def load_calibrated(args: argparse.Namespace) -> tuple[Path, Calibrated]:
    """The run folder and every mark next to the judge's verdict; raises what main reports."""
    sheet = load_sheet(args.marks)
    run_dir: Path = args.run or args.results_dir / sheet.run
    if run_dir.name != sheet.run:
        raise ValueError(f"{args.marks} holds the marks of {sheet.run}, not of {run_dir.name}")
    thresholds = load_judge_config(args.config).thresholds
    records = read_jsonl(run_dir / JUDGEMENTS_NAME, JudgeRecord)
    items = calibrate(sheet, {record.id: verdicts_of(record) for record in records})
    gates = calibration_gates(items, thresholds)
    calibrated = Calibrated(sheet, args.marks, records, items, gates, thresholds.agreement_min)
    return run_dir, calibrated


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir, calibrated = load_calibrated(args)
    except SETUP_ERRORS as error:
        print(f"cannot calibrate: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    out = run_dir / CALIBRATION_NAME
    out.write_text(render_calibration(calibrated), encoding="utf-8")
    for check in Check:
        cells = (f"{name} {_ratio(agreement(calibrated.items, check, group))}"
                 for group, name in GROUPS)
        print(f"{check}: " + " · ".join(cells))
    for gate in calibrated.gates:
        print(f"{MARKS[gate.passed]} {gate.name}: {gate.measured:.2f} (limit {gate.limit:g})")
    print(f"calibration: {out}")
    return 0 if all(gate.passed for gate in calibrated.gates) else 1


if __name__ == "__main__":
    sys.exit(main())
