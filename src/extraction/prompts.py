"""Versioned extraction prompts: a YAML header with the version, then the prompt text."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

HEADER_FENCE = "---"


@dataclass(frozen=True)
class Prompt:
    version: str
    text: str
    sha256: str  # hash of the whole file, so an edit without a version bump is still visible


def load_prompt(path: Path) -> Prompt:
    content = path.read_text(encoding="utf-8")
    header, text = _split_header(content, path)
    version = header.get("version")
    if not version:
        raise ValueError(f"{path}: header has no version")
    if not text:
        raise ValueError(f"{path}: prompt text is empty")
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return Prompt(version=str(version), text=text, sha256=sha256)


def _split_header(content: str, path: Path) -> tuple[dict[str, object], str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != HEADER_FENCE:
        raise ValueError(f"{path}: prompt must start with a '{HEADER_FENCE}' header")
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == HEADER_FENCE)
    except StopIteration:
        raise ValueError(f"{path}: header is not closed with '{HEADER_FENCE}'") from None
    header = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(header, dict):
        raise ValueError(f"{path}: header must be a YAML mapping")
    return header, "\n".join(lines[end + 1 :]).strip()
