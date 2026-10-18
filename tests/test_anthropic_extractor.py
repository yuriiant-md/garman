import inspect
import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from src.extraction.anthropic_extractor import (
    DEAL_TOOL,
    AnthropicConfig,
    AnthropicExtractor,
    sampling_params,
)
from src.extraction.prompts import Prompt
from src.extraction.result import ExtractionStatus
from src.extraction.sampling import Sampling
from src.schemas.deal import Crop

CONFIG = AnthropicConfig(model="claude-haiku-4-5", timeout_s=30, max_retries=3, max_tokens=4096)
PROMPT = Prompt(version="v1", text="Витягни умови угоди.", sha256="abc")
PRECISE = Sampling(name="precise", temperature=0.0, top_p=1.0)
REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def reply(content: list[SimpleNamespace], stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
    )


def tool_reply(deal_input: dict[str, object], stop_reason: str = "tool_use") -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", name=DEAL_TOOL["name"], input=deal_input)
    return reply([block], stop_reason)


def text_reply(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return reply([SimpleNamespace(type="text", text=text)], stop_reason)


class FakeClient:
    def __init__(self, outcome: object) -> None:
        self.calls: list[dict[str, object]] = []
        self.messages = SimpleNamespace(create=self._create)
        self._outcome = outcome

    def _create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def status_error(error_class: type[anthropic.APIStatusError], code: int) -> Exception:
    return error_class("boom", response=httpx2.Response(code, request=REQUEST), body=None)


def extract(outcome: object) -> tuple[object, FakeClient]:
    client = FakeClient(outcome)
    result = AnthropicExtractor(CONFIG, client=client).extract("пш 3кл 500т", PROMPT, PRECISE)
    return result, client


def test_valid_tool_call_becomes_ok_result_with_deal_and_usage():
    result, _ = extract(tool_reply({"crop": "wheat", "quantity_t": 500}))

    assert result.status is ExtractionStatus.OK
    assert result.deal.crop is Crop.WHEAT
    assert result.deal.quantity_t == 500
    assert (result.input_tokens, result.output_tokens) == (1200, 300)
    assert (result.model, result.prompt_version, result.sampling) == ("claude-haiku-4-5", "v1", "precise")


def test_request_forces_the_deal_tool_with_system_prompt_and_temperature_only():
    _, client = extract(tool_reply({}))
    call = client.calls[0]

    assert call["model"] == "claude-haiku-4-5"
    assert call["system"] == PROMPT.text
    assert call["tools"] == [DEAL_TOOL]
    assert call["tool_choice"] == {"type": "tool", "name": DEAL_TOOL["name"]}
    assert call["extra_body"] == {"temperature": 0.0}
    assert "пш 3кл 500т" in call["messages"][0]["content"]


def test_deal_tool_is_not_compiled_into_a_strict_grammar():
    # Strict schemas allow 24 optional and 16 nullable fields in total; Deal has 33 of each.
    _, client = extract(tool_reply({}))

    assert "strict" not in DEAL_TOOL
    assert "output_config" not in client.calls[0]


def test_request_arguments_exist_in_the_installed_sdk_signature():
    _, client = extract(tool_reply({}))
    accepted = inspect.signature(anthropic.resources.messages.Messages.create).parameters

    assert set(client.calls[0]) <= set(accepted)


def test_result_logs_the_sampling_values_actually_sent():
    result, _ = extract(tool_reply({}))

    assert (result.temperature, result.top_p) == (0.0, None)


def test_content_validator_failure_is_invalid_and_keeps_raw_output():
    deal_input = {"delivery": {"date_from": "2026-11-20", "date_to": "2026-11-01"}}

    result, _ = extract(tool_reply(deal_input))

    assert result.status is ExtractionStatus.INVALID
    assert result.deal is None
    assert json.loads(result.raw_output) == deal_input
    assert "date_from" in result.error


def test_truncated_tool_call_is_invalid_even_if_its_input_parses():
    # A cut-off call can arrive with an empty input, and an empty Deal would otherwise pass.
    result, _ = extract(tool_reply({}, stop_reason="max_tokens"))

    assert result.status is ExtractionStatus.INVALID
    assert result.deal is None
    assert "max_tokens" in result.error


def test_reply_without_the_tool_call_is_invalid_and_keeps_the_text():
    result, _ = extract(text_reply("Не можу знайти умови угоди."))

    assert result.status is ExtractionStatus.INVALID
    assert result.raw_output == "Не можу знайти умови угоди."
    assert DEAL_TOOL["name"] in result.error


@pytest.mark.parametrize(
    "error",
    [
        anthropic.APITimeoutError(request=REQUEST),
        anthropic.APIConnectionError(request=REQUEST),
        status_error(anthropic.RateLimitError, 429),
        status_error(anthropic.InternalServerError, 500),
    ],
)
def test_errors_left_after_sdk_retries_become_error_results(error):
    result, _ = extract(error)

    assert result.status is ExtractionStatus.ERROR
    assert result.deal is None
    assert result.error


@pytest.mark.parametrize(
    ("error_class", "code"),
    [(anthropic.AuthenticationError, 401), (anthropic.BadRequestError, 400)],
)
def test_configuration_errors_stop_the_run(error_class, code):
    with pytest.raises(error_class):
        extract(status_error(error_class, code))


@pytest.mark.parametrize(
    ("sampling", "expected"),
    [
        (Sampling(name="both", temperature=0.7, top_p=0.9), {"temperature": 0.7}),
        (Sampling(name="top_p", top_p=0.9), {"top_p": 0.9}),
        (Sampling(name="none"), {}),
    ],
)
def test_claude_gets_temperature_or_top_p_never_both(sampling, expected):
    assert sampling_params(sampling) == expected
