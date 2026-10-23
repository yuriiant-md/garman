from datetime import datetime, timedelta, timezone

import pytest
import yaml

import main as stage1

STARTED = datetime(2026, 10, 23, 14, 5, 33, tzinfo=timezone(timedelta(hours=3)))


def test_every_configuration_runs_on_all_texts_into_one_folder_with_the_metrics(
    monkeypatch, tmp_path
):
    runs: list[list[str]] = []
    tables: list[list[str]] = []
    monkeypatch.setattr("src.extraction.run.main", lambda argv: runs.append(argv) or 0)
    monkeypatch.setattr("src.extraction.metrics.main", lambda argv: tables.append(argv) or 0)
    monkeypatch.setattr("src.extraction.run.current_time", lambda: STARTED)
    monkeypatch.setattr("main.RESULTS_DIR", tmp_path)
    run_dir = str(tmp_path / "2026-10-23T14-05-33")

    code = stage1.main([])

    assert code == 0
    assert len(runs) == len(stage1.MODELS) * len(stage1.PROMPTS) * len(stage1.SAMPLINGS)
    assert all(argv[argv.index("--split") + 1] == "all" for argv in runs)
    assert all(argv[argv.index("--run-dir") + 1] == run_dir for argv in runs)
    assert tables == [
        [
            "--results-dir", run_dir,
            "--output", f"{run_dir}/metrics.md",
            "--by-field-output", f"{run_dir}/metrics_by_field.md",
        ]
    ]


def test_the_matrix_covers_every_sampling_configuration_of_the_config_file():
    path = stage1.ROOT / "configs" / "sampling.yaml"
    configured = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert set(stage1.SAMPLINGS) == set(configured)
    assert len(configured) >= 2


def test_a_configuration_that_cannot_run_is_reported_and_the_metrics_are_still_saved(
    monkeypatch, capsys
):
    tables: list[list[str]] = []
    monkeypatch.setattr("src.extraction.run.main", lambda argv: 2 if "lapa" in argv else 0)
    monkeypatch.setattr("src.extraction.metrics.main", lambda argv: tables.append(argv) or 0)

    code = stage1.main([])

    assert code == 1
    assert len(tables) == 1
    assert "lapa" in capsys.readouterr().err


def test_there_is_no_cache_to_skip_so_fresh_is_not_an_option():
    with pytest.raises(SystemExit):
        stage1.main(["--fresh"])
