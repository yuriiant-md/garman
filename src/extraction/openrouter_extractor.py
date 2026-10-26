"""GPT extractor through OpenRouter: strict structured output by the Deal schema, validated locally.

OpenRouter speaks the OpenAI API, so the official SDK talks to it with another base URL and key.
Strict mode compiles the schema into a grammar, as Ollama does, but only for schemas where every
key is required and no extra keys are allowed; nullable values still let the model skip a term.

Retries and timeouts are the SDK's own: it retries 408/409/429/5xx and connection errors with
exponential backoff. Whatever is left after those retries becomes an `error` result; errors that
mean the request itself is wrong (bad key, rejected schema, unknown model) stop the run instead.
"""

import os
import time
from dataclasses import dataclass
from typing import cast

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema

from src.extraction.messages import user_message
from src.extraction.prompts import Prompt
from src.extraction.result import ExtractionResult, ExtractionStatus, make_result
from src.extraction.retry import is_retryable
from src.extraction.sampling import Sampling, SamplingParams, sampling_options
from src.extraction.validation import repair_request, require_every_key, validate_deal
from src.schemas.deal import Deal

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_VARIABLE = "OPENROUTER_API_KEY"
# Without it OpenRouter may pick a provider that silently drops response_format or sampling.
ROUTING = {"provider": {"require_parameters": True}}


def strict_schema(node: object) -> object:
    """Copy of a JSON schema that OpenAI strict mode accepts: no extra keys in any object, no
    `default`, which its list of supported keywords does not name, and nothing next to a `$ref`:
    a field that is a plain enum (QualityIndicator.limit) loses its description there."""
    if isinstance(node, list):
        return [strict_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        return {"$ref": node["$ref"]}
    copy = {key: strict_schema(value) for key, value in node.items() if key != "default"}
    if "properties" in copy:
        copy["additionalProperties"] = False
    return copy


# strict_schema and require_every_key turn a dict into a dict
DEAL_SCHEMA = cast(dict[str, object], strict_schema(require_every_key(Deal.model_json_schema())))
RESPONSE_FORMAT: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": {"name": "deal", "strict": True, "schema": DEAL_SCHEMA},
}


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str
    timeout_s: float
    max_retries: int
    max_tokens: int


def read_answer(response: ChatCompletion) -> tuple[str, Deal | None, str | None]:
    """Raw output, the validated Deal or None, and the reason it is invalid."""
    choice = response.choices[0]
    if choice.message.refusal:
        return choice.message.refusal, None, "the model refused to answer"
    raw = choice.message.content or ""
    if choice.finish_reason != "stop":
        return raw, None, f"finish_reason={choice.finish_reason}: the answer is incomplete"
    deal, error = validate_deal(raw)
    return raw, deal, error


class OpenRouterExtractor:
    def __init__(self, config: OpenRouterConfig, client: openai.OpenAI | None = None) -> None:
        self.config = config
        # The key is passed explicitly: left alone, the SDK would read OPENAI_API_KEY instead.
        self._client = client or openai.OpenAI(
            base_url=BASE_URL,
            api_key=os.environ[API_KEY_VARIABLE],
            timeout=config.timeout_s,
            max_retries=config.max_retries,
        )

    def extract(self, text: str, prompt: Prompt, sampling: Sampling) -> ExtractionResult:
        params = sampling_options(sampling)  # OpenAI accepts temperature and top_p together
        started = time.perf_counter()
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": prompt.text},
            {"role": "user", "content": user_message(text)},
        ]
        try:
            responses = self._ask_with_one_fix(messages, params)
        except openai.APITimeoutError:
            error = f"timed out after {self.config.max_retries} retries"
        except openai.APIConnectionError as exc:
            error = f"connection failed after retries: {exc}"
        except openai.APIStatusError as exc:
            if not is_retryable(exc.status_code):
                raise
            error = f"API error {exc.status_code} after retries: {exc.message}"
        else:
            return self._answer_result(responses, prompt, sampling, params, started)
        return make_result(
            self.config.model, prompt, sampling, params, started, ExtractionStatus.ERROR,
            error=error,
        )

    def _ask(
        self, messages: list[ChatCompletionMessageParam], params: SamplingParams
    ) -> ChatCompletion:
        return self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            response_format=RESPONSE_FORMAT,
            max_tokens=self.config.max_tokens,
            extra_body=ROUTING,
            **params,
        )

    def _ask_with_one_fix(
        self, messages: list[ChatCompletionMessageParam], params: SamplingParams
    ) -> list[ChatCompletion]:
        """The answers in order: one, or two when the first was invalid and was sent back."""
        first = self._ask(messages, params)
        raw, _, error = read_answer(first)
        if error is None:
            return [first]
        fix: list[ChatCompletionMessageParam] = [
            *messages,
            {"role": "assistant", "content": raw},
            {"role": "user", "content": repair_request(error)},
        ]
        return [first, self._ask(fix, params)]

    def _answer_result(
        self,
        responses: list[ChatCompletion],
        prompt: Prompt,
        sampling: Sampling,
        params: SamplingParams,
        started: float,
    ) -> ExtractionResult:
        raw, deal, error = read_answer(responses[-1])
        status = ExtractionStatus.OK if deal is not None else ExtractionStatus.FAILED
        return make_result(
            self.config.model,
            prompt,
            sampling,
            params,
            started,
            status,
            deal=deal,
            raw_output=raw,
            error=error,
            attempts=len(responses),
            input_tokens=sum(tokens(response)[0] for response in responses),
            output_tokens=sum(tokens(response)[1] for response in responses),
        )


def tokens(response: ChatCompletion) -> tuple[int, int]:
    """Input and output tokens. OpenRouter may leave usage out: such an answer then counts as
    free rather than stopping the whole run."""
    usage = response.usage
    return (0, 0) if usage is None else (usage.prompt_tokens, usage.completion_tokens)
