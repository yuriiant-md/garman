"""Check the hand-labeled golden files in data/golden/ before they become ground truth.

Each file names its text, and its "deal" must pass the Deal schema with every key written out.
Pydantic silently drops an unknown key, so a typo ("quantity" for "quantity_t") would turn a
labeled value into null; a missing key would hide a field the labeler never looked at."""

import argparse
import json
from pathlib import Path
from types import NoneType
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ValidationError

from src.schemas.deal import Deal

ROOT = Path(__file__).resolve().parents[1]
# split and source are read by src/extraction/metrics.py: without split it stops, and without
# source a golden contract would be counted as synthetic
REQUIRED_KEYS = ("id", "text_path", "split", "source", "deal")
GOLDEN_SPLIT = "test"  # golden deals only measure; prompts are tuned on the dev split


def check_file(path: Path, root: Path = ROOT) -> list[str]:
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        return [f"not valid JSON: {error}"]
    if not isinstance(record, dict):
        return ["file must hold a JSON object"]
    missing = [f"missing key: {key}" for key in REQUIRED_KEYS if key not in record]
    if missing:
        return missing
    if not isinstance(record["deal"], dict):
        return ["deal must be a JSON object"]
    errors = []
    if record["id"] != path.stem:
        errors.append(f"id {record['id']!r} does not match file name {path.stem!r}")
    if record["split"] != GOLDEN_SPLIT:
        errors.append(f"split must be {GOLDEN_SPLIT!r} for a golden file, not {record['split']!r}")
    if not (root / record["text_path"]).is_file():
        errors.append(f"text_path {record['text_path']} does not exist")
    errors.extend(key_errors(record["deal"], Deal, "deal"))
    errors.extend(schema_errors(record["deal"]))
    return errors


def key_errors(data: dict[str, Any], model: type[BaseModel], path: str) -> list[str]:
    errors = [f"{path}.{key}: unknown key" for key in data if key not in model.model_fields]
    for name, field in model.model_fields.items():
        if name not in data:
            errors.append(f"{path}.{name}: missing (write null if the text has no value)")
            continue
        nested, is_list = _nested_model(field.annotation)
        value = data[name]
        if nested is None or value is None:
            continue
        if is_list and isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    errors.extend(key_errors(item, nested, f"{path}.{name}[{index}]"))
        elif not is_list and isinstance(value, dict):
            errors.extend(key_errors(value, nested, f"{path}.{name}"))
    return errors


def _nested_model(annotation: Any) -> tuple[type[BaseModel] | None, bool]:
    """Return the model in "Price | None" or "list[Penalty] | None" and whether it is a list."""
    for arg in get_args(annotation) or (annotation,):
        if arg is NoneType:
            continue
        if get_origin(arg) is list:
            inner = get_args(arg)[0]
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                return inner, True
        elif isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg, False
    return None, False


def schema_errors(deal: Any) -> list[str]:
    try:
        Deal.model_validate(deal)
    except ValidationError as error:
        return [f"{_field_path(issue['loc'])}: {issue['msg']}" for issue in error.errors()]
    return []


def _field_path(loc: tuple[int | str, ...]) -> str:
    return "deal" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in loc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check hand-labeled golden files against Deal.")
    parser.add_argument("--root", type=Path, default=ROOT, help="project root with data/golden/")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    golden_dir = args.root / "data" / "golden"
    paths = sorted(golden_dir.glob("*.json"))
    if not paths:
        print(f"no golden files yet in {golden_dir}")
        return 0
    failed = 0
    for path in paths:
        errors = check_file(path, args.root)
        print(f"{'FAIL' if errors else 'OK  '} {path.stem}")
        for error in errors:
            print(f"    - {error}")
        failed += bool(errors)
    print(f"{len(paths) - failed} of {len(paths)} golden files are valid")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
