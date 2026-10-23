import json
from types import SimpleNamespace

import httpx
import ollama
import pytest

from src.extraction.ollama_extractor import DEAL_FORMAT, OllamaConfig, OllamaExtractor
from src.extraction.prompts import Prompt
from src.extraction.result import ExtractionStatus
from src.extraction.sampling import Sampling
from src.schemas.deal import Crop, Deal

CONFIG = OllamaConfig(model="lapa-v0.1.2-q4km", timeout_s=120, num_ctx=16384)
PROMPT = Prompt(version="v1", text="Витягни умови угоди.", sha256="abc")
PRECISE = Sampling(name="precise", temperature=0.0, top_p=1.0)


def answer(content: str, done_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        done_reason=done_reason,
        prompt_eval_count=325,
        eval_count=400,
    )


class FakeClient:
    """A list of outcomes is served one per call, the last one repeating; a single outcome
    answers every call."""

    def __init__(self, outcome: object) -> None:
        self.calls: list[dict[str, object]] = []
        self._outcomes = outcome if isinstance(outcome, list) else [outcome]

    def chat(self, **kwargs: object) -> object:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def extract(outcome: object) -> tuple[object, FakeClient]:
    client = FakeClient(outcome)
    result = OllamaExtractor(CONFIG, client=client).extract("пш 3кл 500т", PROMPT, PRECISE)
    return result, client


def objects_with_properties(node: object) -> list[dict[str, object]]:
    if isinstance(node, list):
        return [found for item in node for found in objects_with_properties(item)]
    if not isinstance(node, dict):
        return []
    nested = [found for value in node.values() for found in objects_with_properties(value)]
    return ([node] if "properties" in node else []) + nested


def test_valid_answer_becomes_ok_result_with_deal_and_token_counts():
    result, _ = extract(answer(json.dumps({"crop": "wheat", "quantity_t": 500})))

    assert result.status is ExtractionStatus.OK
    assert result.deal.crop is Crop.WHEAT
    assert (result.input_tokens, result.output_tokens) == (325, 400)
    assert (result.model, result.prompt_version, result.sampling) == ("lapa-v0.1.2-q4km", "v1", "precise")


def test_request_sends_system_prompt_text_schema_and_both_sampling_values():
    _, client = extract(answer("{}"))
    call = client.calls[0]

    assert call["model"] == "lapa-v0.1.2-q4km"
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][0]["content"].startswith(PROMPT.text)
    assert "пш 3кл 500т" in call["messages"][1]["content"]
    assert call["format"] == DEAL_FORMAT
    assert call["options"] == {"temperature": 0.0, "top_p": 1.0, "num_ctx": 16384}


def test_system_message_carries_the_schema_text_the_grammar_alone_does_not_convey():
    # The grammar only forbids tokens; field meanings reach the model through the prompt.
    _, client = extract(answer("{}"))

    assert json.dumps(DEAL_FORMAT, ensure_ascii=False) in client.calls[0]["messages"][0]["content"]


def test_format_requires_every_key_so_an_empty_object_is_not_a_valid_answer():
    # With optional keys Ollama's grammar accepts `{}`, and a local model takes that shortcut.
    objects = objects_with_properties(DEAL_FORMAT)

    assert objects
    assert all(obj["required"] == list(obj["properties"]) for obj in objects)
    assert "required" not in Deal.model_json_schema()  # the Deal schema itself stays untouched


def test_result_logs_both_sampling_values_sent():
    result, _ = extract(answer("{}"))

    assert (result.temperature, result.top_p) == (0.0, 1.0)


def test_content_validator_failure_that_stays_is_failed_and_keeps_raw_output():
    raw = json.dumps({"delivery": {"date_from": "2026-11-20", "date_to": "2026-11-01"}})

    result, _ = extract(answer(raw))

    assert result.status is ExtractionStatus.FAILED
    assert result.deal is None
    assert result.raw_output == raw
    assert "date_from" in result.error


def test_cut_off_answer_is_failed_and_names_the_done_reason():
    result, _ = extract(answer('{"crop": "wh', done_reason="length"))

    assert result.status is ExtractionStatus.FAILED
    assert "length" in result.error


BAD_RAW = json.dumps({"delivery": {"date_from": "2026-11-20", "date_to": "2026-11-01"}})


def test_valid_first_answer_is_asked_once():
    result, client = extract(answer(json.dumps({"crop": "wheat"})))

    assert (result.status, result.attempts, len(client.calls)) == (ExtractionStatus.OK, 1, 1)


def test_invalid_answer_gets_one_fix_with_its_own_text_and_the_error_in_the_chat():
    result, client = extract([answer(BAD_RAW), answer(json.dumps({"crop": "wheat"}))])
    *_, assistant, user = client.calls[1]["messages"]

    assert result.status is ExtractionStatus.OK
    assert result.attempts == 2
    assert result.deal.crop is Crop.WHEAT
    assert assistant == {"role": "assistant", "content": BAD_RAW}
    assert user["role"] == "user"
    assert "date_from" in user["content"]


def test_both_attempts_are_paid_for_in_tokens():
    result, _ = extract([answer(BAD_RAW), answer(json.dumps({"crop": "wheat"}))])

    assert (result.input_tokens, result.output_tokens) == (650, 800)


def test_answer_that_is_still_invalid_after_the_fix_is_failed_with_the_last_answer():
    last = json.dumps({"delivery": {"date_from": "2026-12-20", "date_to": "2026-11-01"}})

    result, client = extract([answer(BAD_RAW), answer(last)])

    assert (result.status, result.attempts, len(client.calls)) == (ExtractionStatus.FAILED, 2, 2)
    assert result.raw_output == last


def test_no_answer_is_not_fixed_because_there_is_nothing_to_fix():
    result, client = extract(httpx.ReadTimeout("timed out"))

    assert (result.status, result.attempts, len(client.calls)) == (ExtractionStatus.ERROR, 1, 1)


def test_timeout_becomes_error_result():
    result, _ = extract(httpx.ReadTimeout("timed out"))

    assert result.status is ExtractionStatus.ERROR
    assert result.deal is None
    assert "120" in result.error


@pytest.mark.parametrize(
    "error",
    [ollama.ResponseError("model 'lapa' not found", 404), ConnectionError("Failed to connect")],
)
def test_missing_model_or_server_stops_the_run(error):
    with pytest.raises(type(error)):
        extract(error)
