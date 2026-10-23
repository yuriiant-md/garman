import pytest

import main as stage1


def test_every_configuration_runs_on_all_texts_then_the_metrics_are_saved(monkeypatch):
    runs: list[list[str]] = []
    tables: list[list[str]] = []
    monkeypatch.setattr("src.extraction.run.main", lambda argv: runs.append(argv) or 0)
    monkeypatch.setattr("src.extraction.metrics.main", lambda argv: tables.append(argv) or 0)

    code = stage1.main([])

    assert code == 0
    assert len(runs) == len(stage1.MODELS) * len(stage1.PROMPTS) * len(stage1.SAMPLINGS)
    assert all(argv[argv.index("--split") + 1] == "all" for argv in runs)
    assert tables == [
        ["--output", str(stage1.METRICS_PATH), "--by-field-output", str(stage1.BY_FIELD_PATH)]
    ]


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
