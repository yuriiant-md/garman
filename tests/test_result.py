import time

from src.extraction.prompts import Prompt
from src.extraction.result import ExtractionStatus, make_result
from src.extraction.sampling import Sampling

PROMPT = Prompt(version="v2", text="Витягни умови угоди.", sha256="abc")


def test_result_logs_model_prompt_and_the_sampling_values_actually_sent():
    started = time.perf_counter() - 2.0

    result = make_result(
        "claude-haiku-4-5",
        PROMPT,
        Sampling(name="precise", temperature=0.0, top_p=0.9),
        {"temperature": 0.0},  # Claude sends one of the two configured values
        started,
        ExtractionStatus.FAILED,
        error="bad json",
        attempts=2,
    )

    assert (result.model, result.prompt_version, result.prompt_sha256, result.sampling) == (
        "claude-haiku-4-5",
        "v2",
        "abc",
        "precise",
    )
    assert (result.temperature, result.top_p) == (0.0, None)
    assert 2.0 <= result.latency_s < 3.0
    assert result.status == ExtractionStatus.FAILED
    assert (result.error, result.attempts) == ("bad json", 2)
