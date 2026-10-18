"""Claude extractor: a forced tool call shaped by the Deal JSON schema, validated locally.

Retries and timeouts are the SDK's own: it retries 408/409/429/5xx and connection errors with
exponential backoff. Whatever is left after those retries becomes an `error` result; errors that
mean the request itself is wrong (bad key, bad schema, unknown model) stop the run instead.
"""

import json
import time
from dataclasses import dataclass

import anthropic

from src.extraction.prompts import Prompt
from src.extraction.result import ExtractionResult, ExtractionStatus
from src.extraction.sampling import Sampling
from src.extraction.validation import validate_deal
from src.schemas.deal import Deal

# A forced tool call instead of output_config.format: strict schemas compile into a grammar that
# allows 24 optional and 16 nullable fields in total, and every Deal field is both (33 of each).
# Without strict the schema guides the model but is not enforced, so validate_deal decides.
DEAL_TOOL = {
    "name": "record_deal",
    "description": "Record the deal terms found in the text; use null for terms it does not state.",
    "input_schema": Deal.model_json_schema(),
}
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


def read_answer(response: anthropic.types.Message) -> tuple[str, Deal | None, str | None]:
    """Raw output, the validated Deal or None, and the reason it is invalid."""
    call = next(
        (b for b in response.content if b.type == "tool_use" and b.name == DEAL_TOOL["name"]), None
    )
    if call is None:
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, None, f"no {DEAL_TOOL['name']} call"
    raw = json.dumps(call.input, ensure_ascii=False)
    if response.stop_reason == "max_tokens":
        return raw, None, "tool call cut off"
    deal, error = validate_deal(raw)
    return raw, deal, error


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
                tools=[DEAL_TOOL],
                tool_choice={"type": "tool", "name": DEAL_TOOL["name"]},
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
        raw, deal, error = read_answer(response)
        if error and response.stop_reason != "tool_use":
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
