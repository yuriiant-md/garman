"""Named sampling configurations from configs/sampling.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml

SAMPLING_PATH = Path(__file__).resolve().parents[2] / "configs" / "sampling.yaml"


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


def sampling_options(sampling: Sampling) -> dict[str, float]:
    """The values that are set, both as configured: for Ollama and OpenAI, which accept
    temperature and top_p together (Claude takes one of them, see anthropic_extractor)."""
    values = {"temperature": sampling.temperature, "top_p": sampling.top_p}
    return {name: value for name, value in values.items() if value is not None}
