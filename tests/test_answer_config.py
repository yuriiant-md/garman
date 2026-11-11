from pathlib import Path

import pytest

from src.extraction.prompts import load_prompt
from src.rag.answer_config import ANSWER_CONFIG_PATH, load_answer_config

VALID = """\
model: haiku-5.5
prompt: prompts/answer_v1.md
context:
  contract_k: 4
  law_k: 3
  decision_k: 3
  max_tokens: 4000
  min_score: -1.7
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "answer.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_config_names_the_model_the_prompt_and_the_context_settings(tmp_path: Path) -> None:
    config = load_answer_config(write(tmp_path, VALID))

    assert config.model == "haiku-5.5"
    assert config.prompt == Path("prompts/answer_v1.md")
    assert (config.context.contract_k, config.context.law_k, config.context.decision_k) == (4, 3, 3)
    assert (config.context.max_tokens, config.context.min_score) == (4000, -1.7)


def test_the_repository_config_loads_and_its_prompt_has_a_version() -> None:
    config = load_answer_config(ANSWER_CONFIG_PATH)

    assert config.model == "haiku-5.5"
    assert load_prompt(config.prompt).version


def test_an_unknown_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="answer.yaml(.|\\n)*min_scor"):
        load_answer_config(write(tmp_path, VALID.replace("min_score", "min_scor")))


def test_an_impossible_value_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        load_answer_config(write(tmp_path, VALID.replace("max_tokens: 4000", "max_tokens: 0")))


def test_a_file_that_is_not_a_mapping_names_the_file(tmp_path: Path) -> None:
    path = write(tmp_path, "- haiku-5.5\n")

    with pytest.raises(ValueError, match="answer.yaml"):
        load_answer_config(path)
