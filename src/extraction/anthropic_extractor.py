"""Claude extractor: structured output constrained to the Deal JSON schema, validated locally.

Retries and timeouts are the SDK's own: it retries 408/409/429/5xx and connection errors with
exponential backoff. Whatever is left after those retries becomes an `error` result; errors that
mean the request itself is wrong (bad key, bad schema, unknown model) stop the run instead.
"""

import time
from dataclasses import dataclass

import anthropic

from src.extraction.prompts import Prompt
from src.extraction.result import ExtractionResult, ExtractionStatus
from src.extraction.sampling import Sampling
from src.extraction.validation import validate_deal
from src.schemas.deal import Deal

# transform_schema moves constraints the API cannot enforce (gt, le) into field descriptions;
# validate_deal checks them locally afterwards.
DEAL_OUTPUT_FORMAT = {"type": "json_schema", "schema": anthropic.transform_schema(Deal)}
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})


@dataclass(frozen=True)
class AnthropicConfig:
    model: str
    timeout_s: float
    max_retries: int
    max_tokens: int


def sampling_params(sampling: Sampling) -> dict[str, float]:
    """Claude 4.x accepts temperature or top_p, not both; temperature wins when both are set."""
    if sampling.temperature is not None:
        return {"temperature": sampling.temperature}
    if sampling.top_p is not None:
        return {"top_p": sampling.top_p}
    return {}


def user_message(text: str) -> str:
    return f"Текст угоди:\n<text>\n{text}\n</text>"


def is_retryable(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES or status_code >= 500


class AnthropicExtractor:
    def __init__(self, config: AnthropicConfig, client: anthropic.Anthropic | None = None) -> None:
        self.config = config
        self._client = client or anthropic.Anthropic(
            timeout=config.timeout_s, max_retries=config.max_retries
        )

    def extract(self, text: str, prompt: Prompt, sampling: Sampling) -> ExtractionResult:
        params = sampling_params(sampling)
        started = time.perf_counter()
        try:
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=prompt.text,
                messages=[{"role": "user", "content": user_message(text)}],
                output_config={"format": DEAL_OUTPUT_FORMAT},
                # SDK 1.x dropped sampling arguments from its signatures; models that still
                # accept them (Haiku 4.5) get them merged into the request body as-is.
                extra_body=params or None,
            )
        except anthropic.APITimeoutError:
            error = f"timed out after {self.config.max_retries} retries"
        except anthropic.APIConnectionError as exc:
            error = f"connection failed after retries: {exc}"
        except anthropic.APIStatusError as exc:
            if not is_retryable(exc.status_code):
                raise
            error = f"API error {exc.status_code} after retries: {exc.message}"
        else:
            return self._answer_result(response, prompt, sampling, params, started)
        return self._result(prompt, sampling, params, started, ExtractionStatus.ERROR, error=error)

    def _answer_result(
        self,
        response: anthropic.types.Message,
        prompt: Prompt,
        sampling: Sampling,
        params: dict[str, float],
        started: float,
    ) -> ExtractionResult:
        raw = next((block.text for block in response.content if block.type == "text"), "")
        deal, error = validate_deal(raw)
        if error and response.stop_reason != "end_turn":
            error = f"stop_reason={response.stop_reason}: {error}"
        status = ExtractionStatus.OK if deal is not None else ExtractionStatus.INVALID
        return self._result(
            prompt,
            sampling,
            params,
            started,
            status,
            deal=deal,
            raw_output=raw,
            error=error,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def _result(
        self,
        prompt: Prompt,
        sampling: Sampling,
        params: dict[str, float],
        started: float,
        status: ExtractionStatus,
        **fields: object,
    ) -> ExtractionResult:
        return ExtractionResult(
            model=self.config.model,
            prompt_version=prompt.version,
            prompt_sha256=prompt.sha256,
            sampling=sampling.name,
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            status=status,
            latency_s=round(time.perf_counter() - started, 3),
            **fields,
        )
