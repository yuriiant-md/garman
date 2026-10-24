"""Local extractor (Lapa through Ollama): JSON constrained by the Deal schema, validated locally.

Ollama compiles `format` into a grammar the model cannot leave. Every Deal field is optional, so the
shortest valid answer is `{}`, and a local model takes it. The schema sent here therefore requires
every key while values stay nullable: the model has to decide on each field.

No retries: a local model that ran out of time once will run out again. A timeout becomes an
`error` result; a missing model or a stopped Ollama server stops the run instead.
"""

import json
import time
from dataclasses import dataclass

import httpx
import ollama

from src.extraction.messages import user_message
from src.extraction.prompts import Prompt
from src.extraction.result import ExtractionResult, ExtractionStatus
from src.extraction.sampling import Sampling
from src.extraction.validation import repair_request, validate_deal
from src.schemas.deal import Deal


def require_every_key(node: object) -> object:
    """Copy of a JSON schema in which every object lists all its properties as required."""
    if isinstance(node, list):
        return [require_every_key(item) for item in node]
    if not isinstance(node, dict):
        return node
    copy = {key: require_every_key(value) for key, value in node.items()}
    if "properties" in copy:
        copy["required"] = list(copy["properties"])
    return copy


DEAL_FORMAT = require_every_key(Deal.model_json_schema())
# The grammar only forbids tokens; what each field means reaches the model through the prompt.
# Without it Lapa nulled whole objects it could have filled. Claude sees the same schema as its
# tool definition.
SCHEMA_TEXT = json.dumps(DEAL_FORMAT, ensure_ascii=False)


def system_text(prompt: Prompt) -> str:
    return f"{prompt.text}\n\nResponse JSON schema:\n{SCHEMA_TEXT}"


@dataclass(frozen=True)
class OllamaConfig:
    model: str
    timeout_s: float
    num_ctx: int
    think: bool | None = None  # False for models that reason first (Qwen3); None: not sent


def sampling_options(sampling: Sampling) -> dict[str, float]:
    """Ollama accepts temperature and top_p together, so both go out as configured."""
    values = {"temperature": sampling.temperature, "top_p": sampling.top_p}
    return {name: value for name, value in values.items() if value is not None}


class OllamaExtractor:
    def __init__(self, config: OllamaConfig, client: ollama.Client | None = None) -> None:
        self.config = config
        self._client = client or ollama.Client(timeout=config.timeout_s)

    def extract(self, text: str, prompt: Prompt, sampling: Sampling) -> ExtractionResult:
        params = sampling_options(sampling)
        started = time.perf_counter()
        messages = [
            {"role": "system", "content": system_text(prompt)},
            {"role": "user", "content": user_message(text)},
        ]
        try:
            responses = self._ask_with_one_fix(messages, params)
        except httpx.TimeoutException:  # the ollama client lets httpx timeouts through unwrapped
            error = f"timed out after {self.config.timeout_s:g} s"
            return self._result(prompt, sampling, params, started, ExtractionStatus.ERROR, error=error)
        response = responses[-1]
        raw = response.message.content or ""
        deal, error = validate_deal(raw)
        if error and response.done_reason != "stop":
            error = f"done_reason={response.done_reason}: {error}"
        status = ExtractionStatus.OK if deal is not None else ExtractionStatus.FAILED
        return self._result(
            prompt,
            sampling,
            params,
            started,
            status,
            deal=deal,
            raw_output=raw,
            error=error,
            attempts=len(responses),
            input_tokens=sum(r.prompt_eval_count or 0 for r in responses),
            output_tokens=sum(r.eval_count or 0 for r in responses),
        )

    def _ask(self, messages: list[dict[str, str]], params: dict[str, float]) -> ollama.ChatResponse:
        extra = {} if self.config.think is None else {"think": self.config.think}
        return self._client.chat(
            model=self.config.model,
            messages=messages,
            format=DEAL_FORMAT,
            options={**params, "num_ctx": self.config.num_ctx},
            **extra,
        )

    def _ask_with_one_fix(
        self, messages: list[dict[str, str]], params: dict[str, float]
    ) -> list[ollama.ChatResponse]:
        """The answers in order: one, or two when the first was invalid and was sent back."""
        first = self._ask(messages, params)
        raw = first.message.content or ""
        deal, error = validate_deal(raw)
        if deal is not None:
            return [first]
        fix = [
            *messages,
            {"role": "assistant", "content": raw},
            {"role": "user", "content": repair_request(error)},
        ]
        return [first, self._ask(fix, params)]

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
