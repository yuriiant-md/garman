from datetime import date
from pathlib import Path

import pytest

from scripts.run_agent_tasks import load_run, main, parse_args, select_tasks
from src.agents.tasks import Task, TaskKind


def task(task_id: str) -> Task:
    return Task(id=task_id, kind=TaskKind.ANSWER, question="?")


TASKS = (task("t01"), task("t02"), task("t03"))


def write_tasks(path: Path) -> Path:
    path.write_text(
        "today: 2026-11-25\n"
        "tasks:\n"
        "  - {id: t01, kind: answer, question: 'Мої договори?'}\n"
        "  - {id: t02, kind: no_tool, question: 'Привіт'}\n",
        encoding="utf-8",
    )
    return path


def test_the_named_tasks_run_in_the_order_of_the_set() -> None:
    assert [t.id for t in select_tasks(TASKS, "t03, t01")] == ["t01", "t03"]
    assert select_tasks(TASKS, None) == TASKS


@pytest.mark.parametrize(("ids", "problem"), [(" , ", "names no tasks"), ("t01,t09", "t09")])
def test_ids_that_name_no_task_stop_the_start(ids: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        select_tasks(TASKS, ids)


def test_without_an_api_key_it_stops_before_loading_anything(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert main([]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_the_run_takes_the_model_prompt_and_prices_of_the_config(tmp_path: Path) -> None:
    args = parse_args(["--tasks", str(write_tasks(tmp_path / "tasks.yaml")), "--ids", "t02"])

    run = load_run(args)

    assert run.config.model == "haiku-5.5"
    assert run.model.model == "claude-haiku-5-5"
    assert run.prompt.version == "v2"
    assert run.prices == (0.1, 0.5)
    assert [t.id for t in run.tasks] == ["t02"]
    assert run.today == date(2026, 11, 25)


def test_a_model_that_is_not_an_anthropic_one_is_refused(tmp_path: Path) -> None:
    args = parse_args(["--tasks", str(write_tasks(tmp_path / "t.yaml")), "--model", "gpt-4o"])

    with pytest.raises(ValueError, match="gpt-4o is not an anthropic model"):
        load_run(args)


@pytest.mark.parametrize("extra", [["--ids", "t09"], ["--model", "no-such-model"]])
def test_a_bad_start_is_a_message_and_exit_code_2_before_any_server_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    tasks = write_tasks(tmp_path / "tasks.yaml")

    assert main(["--tasks", str(tasks), "--qdrant-url", "http://127.0.0.1:9", *extra]) == 2
    assert "cannot start" in capsys.readouterr().err


def test_a_short_set_is_named_when_it_runs_whole(tmp_path: Path) -> None:
    tasks = str(write_tasks(tmp_path / "tasks.yaml"))

    assert "2 tasks, at least 15 needed" in load_run(parse_args(["--tasks", tasks])).warnings
    # a few tasks on purpose: no warning
    assert load_run(parse_args(["--tasks", tasks, "--ids", "t01"])).warnings == ()


def test_a_bad_limit_of_any_task_stops_the_start_before_the_first_paid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text(
        "today: 2026-11-25\n"
        "tasks:\n"
        "  - {id: t01, kind: answer, question: 'Мої договори?'}\n"
        "  - {id: t02, kind: budget, question: 'Порівняй', limits: {tool_timeout_s: 1}}\n",
        encoding="utf-8",
    )

    assert main(["--tasks", str(tasks), "--qdrant-url", "http://127.0.0.1:9"]) == 2
    assert "t02" in capsys.readouterr().err
