"""configs/answer.yaml: which model answers, with which prompt, from how much context.

The model is a section of configs/models.yaml, so its id, timeout and prices stay in one place.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from src.rag.context import ContextSettings

ANSWER_CONFIG_PATH = Path("configs/answer.yaml")


class AnswerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str  # a section of configs/models.yaml
    prompt: Path
    context: ContextSettings


def load_answer_config(path: Path = ANSWER_CONFIG_PATH) -> AnswerConfig:
    """The config, or a ValueError that names the file and what is wrong in it."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return AnswerConfig.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error
