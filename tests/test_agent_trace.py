import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agents.limits import StopReason, Usage
from src.agents.trace import ToolCall, Trace, TraceStep, append_trace, trace_path
from src.rag.documents import read_jsonl

STARTED = datetime(2026, 11, 17, 14, 5, 9, tzinfo=timezone(timedelta(hours=2)))

SEARCH = ToolCall(
    id="toolu_1",
    name="search_corpus",
    arguments={"query": "строк оплати", "contract_id": "own_003"},
    result="S1: оплата протягом 5 банківських днів",
    latency_s=1.9,
)
STEP_WITH_TOOL = TraceStep(
    step=1,
    input_tokens=800,
    output_tokens=60,
    cost_usd=0.0001,
    latency_s=1.2,
    tool_calls=(SEARCH,),
)
FINAL_STEP = TraceStep(
    step=2,
    input_tokens=1200,
    output_tokens=150,
    cost_usd=0.0002,
    latency_s=2.0,
    text="Оплата протягом 5 банківських днів [S1].",
)


def make_trace(stop: StopReason, answer: str | None = "Оплата протягом 5 днів [S1].") -> Trace:
    return Trace(
        request_id="r1",
        started_at=STARTED,
        model="haiku-5.5",
        question="Коли мені заплатять?",
        steps=(STEP_WITH_TOOL, FINAL_STEP),
        usage=Usage(steps=2, input_tokens=2000, output_tokens=210, cost_usd=0.0003),
        stop_reason=stop,
        answer=answer,
        latency_s=5.1,
    )


# --- tool calls


def test_a_tool_call_keeps_its_result_or_its_error() -> None:
    failed = SEARCH.model_copy(update={"result": None, "error": "timeout after 20 s"})

    assert SEARCH.result and SEARCH.error is None
    assert failed.error == "timeout after 20 s"


@pytest.mark.parametrize("result, error", [(None, None), ("S1", "timeout")])
def test_a_tool_call_needs_exactly_one_of_result_and_error(
    result: str | None, error: str | None
) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ToolCall(
            id="t", name="search_corpus", arguments={}, result=result, error=error, latency_s=0
        )


# --- partial answers


def test_an_answered_trace_is_not_partial() -> None:
    assert make_trace(StopReason.ANSWERED).partial is False


@pytest.mark.parametrize(
    "stop", [StopReason.MAX_STEPS, StopReason.BUDGET, StopReason.TIMEOUT, StopReason.ERROR]
)
def test_a_trace_stopped_by_a_limit_or_an_error_is_partial(stop: StopReason) -> None:
    assert make_trace(stop).partial is True


# --- file


def test_the_trace_file_is_named_after_the_start_of_the_run(tmp_path: Path) -> None:
    assert trace_path(STARTED, tmp_path) == tmp_path / "2026-11-17T14-05-09.jsonl"


def test_append_writes_one_line_per_request_and_keeps_the_earlier_ones(tmp_path: Path) -> None:
    path = tmp_path / "traces" / "run.jsonl"  # the folder does not exist yet
    first, second = make_trace(StopReason.ANSWERED), make_trace(StopReason.BUDGET, answer=None)

    append_trace(first, path)
    append_trace(second, path)

    assert read_jsonl(path, Trace) == (first, second)


def test_a_trace_line_shows_the_stop_reason_the_usage_and_every_tool_argument(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"

    append_trace(make_trace(StopReason.BUDGET), path)
    line = json.loads(path.read_text(encoding="utf-8"))

    assert line["stop_reason"] == "budget"
    assert line["usage"] == {
        "steps": 2,
        "input_tokens": 2000,
        "output_tokens": 210,
        "cost_usd": 0.0003,
    }
    assert line["steps"][0]["tool_calls"][0]["arguments"] == {
        "query": "строк оплати",
        "contract_id": "own_003",
    }
    assert "строк оплати" in path.read_text(encoding="utf-8")  # Ukrainian stays readable
