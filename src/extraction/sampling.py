"""Named sampling configurations from configs/sampling.yaml."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import yaml

SAMPLING_PATH = Path(__file__).resolve().parents[2] / "configs" / "sampling.yaml"
DEFAULT_SAMPLING = "default"  # sends nothing: for models that reject temperature/top_p


@dataclass(frozen=True)
class Sampling:
    name: str
    temperature: float | None = None
    top_p: float | None = None


def load_sampling(name: str, path: Path = SAMPLING_PATH) -> Sampling:
    configs = yaml.safe_load(path.read_text(encoding="utf-8"))
    if name not in configs:
        raise KeyError(f"unknown sampling '{name}'; known: {', '.join(configs)}")
    values = configs[name]
    return Sampling(name=name, temperature=values.get("temperature"), top_p=values.get("top_p"))


class SamplingParams(TypedDict, total=False):
    """The sampling values sent to a model; a value that is not set is left out."""

    temperature: float
    top_p: float


def sampling_options(sampling: Sampling) -> SamplingParams:
    """The values that are set, both as configured: for OpenAI, which accepts temperature and
    top_p together (Claude takes one of them, see anthropic_extractor)."""
    values = {"temperature": sampling.temperature, "top_p": sampling.top_p}
    kept = {name: value for name, value in values.items() if value is not None}
    return cast(SamplingParams, kept)  # the keys above are exactly those of SamplingParams
