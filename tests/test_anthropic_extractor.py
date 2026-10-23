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
    block = SimpleNamespace(type="tool_use", id="toolu_1", name=DEAL_TOOL["name"], input=deal_input)
    return reply([block], stop_reason)


def text_reply(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return reply([SimpleNamespace(type="text", text=text)], stop_reason)


class FakeClient:
    """A list of outcomes is served one per call, the last one repeating; a single outcome
    answers every call."""

    def __init__(self, outcome: object) -> None:
        self.calls: list[dict[str, object]] = []
        self.messages = SimpleNamespace(create=self._create)
        self._outcomes = outcome if isinstance(outcome, list) else [outcome]

    def _create(self, **kwargs: object) -> object:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


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


def test_content_validator_failure_that_stays_is_failed_and_keeps_raw_output():
    deal_input = {"delivery": {"date_from": "2026-11-20", "date_to": "2026-11-01"}}

    result, _ = extract(tool_reply(deal_input))

    assert result.status is ExtractionStatus.FAILED
    assert result.deal is None
    assert json.loads(result.raw_output) == deal_input
    assert "date_from" in result.error


def test_truncated_tool_call_is_failed_even_if_its_input_parses():
    # A cut-off call can arrive with an empty input, and an empty Deal would otherwise pass.
    result, _ = extract(tool_reply({}, stop_reason="max_tokens"))

    assert result.status is ExtractionStatus.FAILED
    assert result.deal is None
    assert "max_tokens" in result.error


def test_reply_without_the_tool_call_is_failed_and_keeps_the_text():
    result, _ = extract(text_reply("Не можу знайти умови угоди."))

    assert result.status is ExtractionStatus.FAILED
    assert result.raw_output == "Не можу знайти умови угоди."
    assert DEAL_TOOL["name"] in result.error


BAD_INPUT = {"delivery": {"date_from": "2026-11-20", "date_to": "2026-11-01"}}


def test_valid_first_answer_is_asked_once():
    result, client = extract(tool_reply({"crop": "wheat"}))

    assert (result.status, result.attempts, len(client.calls)) == (ExtractionStatus.OK, 1, 1)


def test_invalid_answer_gets_one_fix_that_returns_the_error_as_a_failed_tool_result():
    result, client = extract([tool_reply(BAD_INPUT), tool_reply({"crop": "wheat"})])
    _, assistant, user = client.calls[1]["messages"]

    assert result.status is ExtractionStatus.OK
    assert result.attempts == 2
    assert result.deal.crop is Crop.WHEAT
    assert assistant["content"][0]["id"] == "toolu_1"
    assert assistant["content"][0]["input"] == BAD_INPUT
    assert user["content"][0]["type"] == "tool_result"
    assert user["content"][0]["tool_use_id"] == "toolu_1"
    assert user["content"][0]["is_error"] is True
    assert "date_from" in user["content"][0]["content"]


def test_both_attempts_are_paid_for_in_tokens_and_latency():
    result, _ = extract([tool_reply(BAD_INPUT), tool_reply({"crop": "wheat"})])

    assert (result.input_tokens, result.output_tokens) == (2400, 600)


def test_answer_that_is_still_invalid_after_the_fix_is_failed_with_the_last_answer():
    last = {"delivery": {"date_from": "2026-12-20", "date_to": "2026-11-01"}}

    result, client = extract([tool_reply(BAD_INPUT), tool_reply(last)])

    assert (result.status, result.attempts, len(client.calls)) == (ExtractionStatus.FAILED, 2, 2)
    assert json.loads(result.raw_output) == last


def test_text_reply_is_fixed_with_a_plain_user_message():
    result, client = extract([text_reply("Не можу."), tool_reply({"crop": "wheat"})])
    _, assistant, user = client.calls[1]["messages"]

    assert (result.status, result.attempts) == (ExtractionStatus.OK, 2)
    assert assistant == {"role": "assistant", "content": "Не можу."}
    assert user["role"] == "user"
    assert DEAL_TOOL["name"] in user["content"]


def test_no_answer_is_not_fixed_because_there_is_nothing_to_fix():
    result, client = extract(anthropic.APITimeoutError(request=REQUEST))

    assert (result.status, result.attempts, len(client.calls)) == (ExtractionStatus.ERROR, 1, 1)


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
