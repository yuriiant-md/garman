import inspect
import json
from types import SimpleNamespace

import httpx2
import openai
import pytest

from src.extraction.openrouter_extractor import (
    API_KEY_VARIABLE,
    BASE_URL,
    DEAL_SCHEMA,
    RESPONSE_FORMAT,
    OpenRouterConfig,
    OpenRouterExtractor,
)
from src.extraction.prompts import Prompt
from src.extraction.result import ExtractionStatus
from src.extraction.sampling import Sampling
from src.schemas.deal import Crop

CONFIG = OpenRouterConfig(model="openai/gpt-4o", timeout_s=60, max_retries=3, max_tokens=4096)
PROMPT = Prompt(version="v1", text="Витягни умови угоди.", sha256="abc")
PRECISE = Sampling(name="precise", temperature=0.0, top_p=1.0)
REQUEST = httpx2.Request("POST", f"{BASE_URL}/chat/completions")
BAD_RAW = json.dumps({"delivery": {"date_from": "2026-11-20", "date_to": "2026-11-01"}})


def answer(
    content: str | None, finish_reason: str = "stop", refusal: str | None = None
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=300),
    )


class FakeClient:
    """A list of outcomes is served one per call, the last one repeating; a single outcome
    answers every call."""

    def __init__(self, outcome: object) -> None:
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._outcomes = outcome if isinstance(outcome, list) else [outcome]

    def _create(self, **kwargs: object) -> object:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def status_error(error_class: type[openai.APIStatusError], code: int) -> Exception:
    return error_class("boom", response=httpx2.Response(code, request=REQUEST), body=None)


def extract(outcome: object) -> tuple[object, FakeClient]:
    client = FakeClient(outcome)
    result = OpenRouterExtractor(CONFIG, client=client).extract("пш 3кл 500т", PROMPT, PRECISE)
    return result, client


def objects(node: object) -> list[dict[str, object]]:
    """Every object schema inside a JSON schema, the root included."""
    if isinstance(node, list):
        return [found for item in node for found in objects(item)]
    if not isinstance(node, dict):
        return []
    inner = [found for value in node.values() for found in objects(value)]
    return [node, *inner] if "properties" in node else inner


def test_strict_schema_requires_every_key_and_forbids_extra_ones_in_every_object():
    found = objects(DEAL_SCHEMA)

    assert len(found) > 1  # the root and the nested objects in $defs
    assert all(schema["required"] == list(schema["properties"]) for schema in found)
    assert all(schema["additionalProperties"] is False for schema in found)


def test_strict_schema_drops_default_which_strict_mode_does_not_list():
    assert '"default"' not in json.dumps(DEAL_SCHEMA)


def test_response_format_asks_for_the_strict_deal_schema():
    assert RESPONSE_FORMAT["type"] == "json_schema"
    assert RESPONSE_FORMAT["json_schema"]["strict"] is True
    assert RESPONSE_FORMAT["json_schema"]["schema"] is DEAL_SCHEMA


def test_valid_answer_becomes_ok_result_with_deal_and_usage():
    result, _ = extract(answer(json.dumps({"crop": "wheat", "quantity_t": 500})))

    assert result.status is ExtractionStatus.OK
    assert result.deal.crop is Crop.WHEAT
    assert result.deal.quantity_t == 500
    assert (result.input_tokens, result.output_tokens) == (1200, 300)
    assert result.model == "openai/gpt-4o"
    assert (result.prompt_version, result.sampling) == ("v1", "precise")


def test_request_sends_system_prompt_schema_and_both_sampling_values():
    _, client = extract(answer("{}"))
    call = client.calls[0]
    system, user = call["messages"]

    assert call["model"] == "openai/gpt-4o"
    assert system == {"role": "system", "content": PROMPT.text}
    assert user["role"] == "user"
    assert "пш 3кл 500т" in user["content"]
    assert call["response_format"] == RESPONSE_FORMAT
    assert (call["temperature"], call["top_p"]) == (0.0, 1.0)
    assert call["max_tokens"] == 4096


def test_openrouter_routes_only_to_providers_that_honour_every_parameter():
    # Otherwise a provider without structured outputs could get the request and ignore the schema
    _, client = extract(answer("{}"))

    assert client.calls[0]["extra_body"] == {"provider": {"require_parameters": True}}


def test_request_arguments_exist_in_the_installed_sdk_signature():
    _, client = extract(answer("{}"))
    accepted = inspect.signature(openai.resources.chat.Completions.create).parameters

    assert set(client.calls[0]) <= set(accepted)


def test_unset_sampling_values_are_not_sent():
    client = FakeClient(answer("{}"))
    sampling = Sampling(name="temperature_only", temperature=0.7)

    result = OpenRouterExtractor(CONFIG, client=client).extract("текст", PROMPT, sampling)

    assert "top_p" not in client.calls[0]
    assert (result.temperature, result.top_p) == (0.7, None)


def test_result_logs_both_sampling_values_sent():
    result, _ = extract(answer("{}"))

    assert (result.temperature, result.top_p) == (0.0, 1.0)


def test_content_validator_failure_that_stays_is_failed_and_keeps_raw_output():
    result, _ = extract(answer(BAD_RAW))

    assert result.status is ExtractionStatus.FAILED
    assert result.deal is None
    assert result.raw_output == BAD_RAW
    assert "date_from" in result.error


def test_cut_off_answer_is_failed_and_names_the_finish_reason():
    # A cut-off answer is incomplete even when what arrived happens to parse
    result, _ = extract(answer("{}", finish_reason="length"))

    assert result.status is ExtractionStatus.FAILED
    assert result.deal is None
    assert "length" in result.error


def test_refusal_is_failed_and_keeps_the_refusal_text():
    result, _ = extract(answer(None, refusal="I can't help with that."))

    assert result.status is ExtractionStatus.FAILED
    assert result.raw_output == "I can't help with that."
    assert "refus" in result.error


def test_valid_first_answer_is_asked_once():
    result, client = extract(answer(json.dumps({"crop": "wheat"})))

    assert (result.status, result.attempts, len(client.calls)) == (ExtractionStatus.OK, 1, 1)


def test_invalid_answer_gets_one_fix_with_its_own_text_and_the_error_in_the_chat():
    result, client = extract([answer(BAD_RAW), answer(json.dumps({"crop": "wheat"}))])
    _, _, assistant, user = client.calls[1]["messages"]

    assert (result.status, result.attempts) == (ExtractionStatus.OK, 2)
    assert result.deal.crop is Crop.WHEAT
    assert assistant == {"role": "assistant", "content": BAD_RAW}
    assert user["role"] == "user"
    assert "date_from" in user["content"]


def test_both_attempts_are_paid_for_in_tokens():
    result, _ = extract([answer(BAD_RAW), answer(json.dumps({"crop": "wheat"}))])

    assert (result.input_tokens, result.output_tokens) == (2400, 600)


def test_answer_that_is_still_invalid_after_the_fix_is_failed_with_the_last_answer():
    last = json.dumps({"delivery": {"date_from": "2026-12-20", "date_to": "2026-11-01"}})

    result, client = extract([answer(BAD_RAW), answer(last)])

    assert (result.status, result.attempts, len(client.calls)) == (ExtractionStatus.FAILED, 2, 2)
    assert result.raw_output == last


def test_no_answer_is_not_fixed_because_there_is_nothing_to_fix():
    result, client = extract(openai.APITimeoutError(request=REQUEST))

    assert (result.status, result.attempts, len(client.calls)) == (ExtractionStatus.ERROR, 1, 1)


@pytest.mark.parametrize(
    "error",
    [
        openai.APITimeoutError(request=REQUEST),
        openai.APIConnectionError(request=REQUEST),
        status_error(openai.RateLimitError, 429),
        status_error(openai.InternalServerError, 500),
    ],
)
def test_errors_left_after_sdk_retries_become_error_results(error):
    result, _ = extract(error)

    assert result.status is ExtractionStatus.ERROR
    assert result.deal is None
    assert result.error


@pytest.mark.parametrize(
    ("error_class", "code"),
    [(openai.AuthenticationError, 401), (openai.BadRequestError, 400)],
)
def test_configuration_errors_stop_the_run(error_class, code):
    with pytest.raises(error_class):
        extract(status_error(error_class, code))


def test_default_client_talks_to_openrouter_with_its_own_key_and_the_configured_limits(
    monkeypatch,
):
    monkeypatch.setenv(API_KEY_VARIABLE, "sk-or-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-be-used")

    client = OpenRouterExtractor(CONFIG)._client

    assert str(client.base_url).rstrip("/") == BASE_URL
    assert client.api_key == "sk-or-test"
    assert (client.timeout, client.max_retries) == (60, 3)
