import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agents.limits import StopReason, Usage
from src.agents.trace import SummaryCall, ToolCall, Trace, TraceStep, append_trace, trace_path
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
        thread_id="t1",
        started_at=STARTED,
        model="haiku-5.5",
        prompt_version="v1",
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


def test_a_tool_call_written_before_tool_versions_still_reads() -> None:
    line = '{"id": "t", "name": "search_corpus", "arguments": {}, "result": "S1", "latency_s": 1}'

    call = ToolCall.model_validate_json(line)

    assert call.tool_version is None


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


def test_a_request_costs_its_model_calls_and_the_summary_made_before_it() -> None:
    trace = make_trace(StopReason.ANSWERED)
    summary = SummaryCall(prompt_version="v1", folded_turns=3, cost_usd=0.0004)

    with_summary = trace.model_copy(update={"summary": summary})

    assert trace.cost_usd == trace.usage.cost_usd  # no summary: the model calls alone
    assert with_summary.cost_usd == pytest.approx(trace.usage.cost_usd + 0.0004)


def test_a_trace_with_its_summary_reads_back_and_an_older_line_without_one_too(
    tmp_path: Path,
) -> None:
    summary = SummaryCall(prompt_version="v1", folded_turns=3, error="empty summary")
    trace = make_trace(StopReason.ANSWERED).model_copy(update={"summary": summary})
    path = tmp_path / "traces.jsonl"
    append_trace(trace, path)
    older = json.loads(make_trace(StopReason.ANSWERED).model_dump_json())
    del older["summary"]  # lines of 25.11 have no summary field
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(older) + "\n")

    first, second = read_jsonl(path, Trace)

    assert first.summary == summary
    assert second.summary is None
