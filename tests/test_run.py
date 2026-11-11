import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx2
import openai
import pytest

from src.extraction import metrics, run
from src.extraction.anthropic_extractor import AnthropicConfig, AnthropicExtractor
from src.extraction.openrouter_extractor import OpenRouterExtractor
from src.extraction.prompts import Prompt
from src.extraction.result import ExtractionResult, ExtractionStatus
from src.extraction.run import (
    build_extractor,
    configured_model,
    load_anthropic_config,
    load_openrouter_config,
    load_prices,
    main,
    model_provider,
    parse_args,
    result_path,
    run_dir_path,
    run_extraction,
    summarize,
    takes_sampling,
    text_paths,
)
from src.extraction.sampling import Sampling, load_sampling

ROOT = Path(__file__).resolve().parents[1]
PROMPT = Prompt(version="v1", text="Витягни умови.", sha256="abc")
PRECISE = Sampling(name="precise", temperature=0.0, top_p=1.0)
STARTED = datetime(2026, 10, 23, 14, 5, 33, tzinfo=timezone(timedelta(hours=3)))


class StubExtractor:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def extract(self, text: str, prompt: Prompt, sampling: Sampling) -> ExtractionResult:
        self.texts.append(text)
        status = ExtractionStatus.ERROR if "збій" in text else ExtractionStatus.OK
        return ExtractionResult(
            model="stub",
            prompt_version=prompt.version,
            prompt_sha256=prompt.sha256,
            sampling=sampling.name,
            temperature=sampling.temperature,
            top_p=None,
            status=status,
            deal={} if status is ExtractionStatus.OK else None,
            latency_s=2.0,
            input_tokens=1000,
            output_tokens=200,
        )


def make_texts(texts_dir: Path) -> None:
    texts_dir.mkdir()
    (texts_dir / "synthetic_002.txt").write_text("пш 3кл", encoding="utf-8")
    (texts_dir / "synthetic_001.txt").write_text("збій", encoding="utf-8")
    (texts_dir / "prozorro_001.txt").write_text("Договір поставки", encoding="utf-8")


@pytest.mark.parametrize(
    ("split", "expected"),
    [
        ("dev", ["synthetic_001", "synthetic_002"]),
        ("test", ["prozorro_001"]),
        ("all", ["prozorro_001", "synthetic_001", "synthetic_002"]),
    ],
)
def test_text_paths_select_documents_by_split_in_sorted_order(tmp_path, split, expected):
    make_texts(tmp_path / "texts")

    assert [path.stem for path in text_paths(tmp_path / "texts", split)] == expected


def test_text_paths_can_take_the_first_n_documents_of_each_source(tmp_path):
    make_texts(tmp_path / "texts")
    (tmp_path / "texts" / "prozorro_002.txt").write_text("Додаток", encoding="utf-8")

    paths = text_paths(tmp_path / "texts", "all", per_source=1)

    assert [path.stem for path in paths] == ["prozorro_001", "synthetic_001"]


def test_per_source_keeps_every_document_when_a_source_has_fewer_than_n(tmp_path):
    make_texts(tmp_path / "texts")

    assert len(text_paths(tmp_path / "texts", "all", per_source=5)) == 3


def test_per_source_is_off_by_default_and_read_from_the_command_line():
    assert parse_args([]).per_source is None
    assert parse_args(["--per-source", "2"]).per_source == 2


@pytest.mark.parametrize("count", ["0", "-1"])
def test_per_source_must_be_a_positive_number(count, capsys):
    with pytest.raises(SystemExit) as stop:
        parse_args(["--per-source", count])

    assert stop.value.code == 2
    assert "--per-source" in capsys.readouterr().err


def test_run_dir_is_named_with_the_start_time(tmp_path):
    assert run_dir_path(tmp_path, STARTED) == tmp_path / "2026-10-23T14-05-33"


def test_result_path_names_model_prompt_and_sampling_inside_the_run_dir(tmp_path):
    path = result_path(tmp_path, "claude-haiku-4-5", "v1", "precise")

    assert path == tmp_path / "claude-haiku-4-5__v1__precise.jsonl"


def test_run_extraction_writes_one_json_line_per_document_with_doc_id_and_run_time(tmp_path):
    make_texts(tmp_path / "texts")
    output = tmp_path / "results" / "out.jsonl"

    records = run_extraction(
        StubExtractor(), text_paths(tmp_path / "texts", "dev"), PROMPT, PRECISE, output, STARTED
    )

    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [line["doc_id"] for line in lines] == ["synthetic_001", "synthetic_002"]
    assert [line["status"] for line in lines] == ["error", "ok"]
    assert {line["run_at"] for line in lines} == {"2026-10-23T14:05:33+03:00"}
    assert lines == records


class FailingExtractor(StubExtractor):
    def extract(self, text: str, prompt: Prompt, sampling: Sampling) -> ExtractionResult:
        if len(self.texts) == 1:
            raise RuntimeError("401 invalid key")
        return super().extract(text, prompt, sampling)


def test_failed_run_leaves_no_results_file_for_the_metrics(tmp_path):
    make_texts(tmp_path / "texts")
    output = tmp_path / "out.jsonl"
    paths = text_paths(tmp_path / "texts", "dev")

    with pytest.raises(RuntimeError):
        run_extraction(FailingExtractor(), paths, PROMPT, PRECISE, output, STARTED)

    assert not output.exists()
    assert output.with_name("out.jsonl.partial").exists()  # what was answered, kept for a look


def test_summary_counts_statuses_tokens_median_latency_and_cost():
    records = [
        {"status": "ok", "latency_s": 1.0, "input_tokens": 1000, "output_tokens": 100},
        {"status": "ok", "latency_s": 3.0, "input_tokens": 1000, "output_tokens": 100},
        {"status": "failed", "latency_s": 2.0, "input_tokens": 1000, "output_tokens": 100},
    ]

    summary = summarize(records, price_input=1.0, price_output=5.0)

    assert summary["statuses"] == {"ok": 2, "failed": 1, "error": 0}
    assert summary["median_latency_s"] == 2.0
    assert summary["cost_usd"] == pytest.approx(3000 / 1e6 + 5 * 300 / 1e6)
    assert summary["cost_per_1000_docs_usd"] == pytest.approx(summary["cost_usd"] / 3 * 1000)


def test_cost_per_1000_docs_ignores_error_results_without_tokens():
    records = [
        {"status": "ok", "latency_s": 1.0, "input_tokens": 1000, "output_tokens": 100},
        {"status": "error", "latency_s": 30.0, "input_tokens": 0, "output_tokens": 0},
    ]

    summary = summarize(records, price_input=1.0, price_output=5.0)

    assert summary["statuses"]["error"] == 1
    assert summary["cost_per_1000_docs_usd"] == pytest.approx(summary["cost_usd"] * 1000)


def test_summary_counts_answers_that_came_without_token_counts():
    # OpenRouter may leave usage out: such an answer costs money but adds no tokens
    records = [
        {"status": "ok", "latency_s": 1.0, "input_tokens": 1000, "output_tokens": 100},
        {"status": "ok", "latency_s": 1.0, "input_tokens": 0, "output_tokens": 0},
        {"status": "failed", "latency_s": 1.0, "input_tokens": 0, "output_tokens": 0},
        {"status": "error", "latency_s": 30.0, "input_tokens": 0, "output_tokens": 0},
    ]

    summary = summarize(records, price_input=1.0, price_output=5.0)

    assert summary["answers_without_usage"] == 2  # an error has no answer, so it is not one


def test_project_configs_load_the_provider_and_both_samplings():
    config = load_anthropic_config(ROOT / "configs" / "models.yaml", section="haiku-4.5")

    assert config.model == "claude-haiku-4-5"
    assert config.timeout_s > 0 and config.max_retries >= 1
    assert load_prices(ROOT / "configs" / "models.yaml", section="haiku-4.5") == (1.0, 5.0)
    assert load_sampling("precise").temperature == 0
    assert load_sampling("varied").temperature > 0


def test_local_models_are_no_longer_offered():
    # Lapa and Qwen ran in stages 1.1 and 1.2: their answers stay in results/, their code in tag stage-1
    path = ROOT / "configs" / "models.yaml"

    for model in ("lapa", "qwen"):
        with pytest.raises(SystemExit):
            parse_args(["--model", model])
    for section in ("ollama", "qwen"):
        with pytest.raises(KeyError):
            load_prices(path, section=section)


def test_project_configs_load_gpt_through_openrouter_with_its_token_prices():
    path = ROOT / "configs" / "models.yaml"

    config = load_openrouter_config(path, section="gpt-4o")

    assert config.model == "openai/gpt-4o"
    assert config.timeout_s > 0 and config.max_retries >= 1 and config.max_tokens > 0
    assert load_prices(path, section="gpt-4o") == (2.5, 10.0)


def test_results_are_named_after_the_chosen_model():
    # the --model keys name the models, so one cannot be taken for another
    assert run.MODELS == ("haiku-5.5", "haiku-4.5", "gpt-4o")
    assert configured_model("haiku-5.5") == "claude-haiku-5-5"
    assert configured_model("haiku-4.5") == "claude-haiku-4-5"
    assert configured_model("gpt-4o") == "openai/gpt-4o"


def test_a_slash_in_the_model_id_does_not_open_a_subfolder(tmp_path):
    # metrics reads only the *.jsonl files directly inside a run folder
    path = result_path(tmp_path, "openai/gpt-4o", "v1", "precise")

    assert path == tmp_path / "openai_gpt-4o__v1__precise.jsonl"


def test_gpt_is_built_as_the_openrouter_extractor(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    assert isinstance(build_extractor("gpt-4o"), OpenRouterExtractor)


def test_every_model_names_a_provider_that_picks_its_extractor_and_api_key():
    providers = {model: model_provider(model) for model in run.MODELS}

    assert providers == {"haiku-5.5": "anthropic", "haiku-4.5": "anthropic", "gpt-4o": "openrouter"}
    assert {run.API_KEY_VARIABLES[name] for name in providers.values()} == {
        "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"
    }


def test_a_new_section_is_read_by_its_provider_not_by_its_name(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "gpt-5:\n  provider: openrouter\n  model: openai/gpt-5\n"
        "  timeout_s: 60\n  max_retries: 3\n  max_tokens: 4096\n",
        encoding="utf-8",
    )

    assert model_provider("gpt-5", path) == "openrouter"
    assert load_openrouter_config(path, section="gpt-5").model == "openai/gpt-5"


@pytest.mark.parametrize("line", ["  provider: ollama\n", ""])
def test_a_section_without_a_known_provider_is_rejected(tmp_path, line):
    path = tmp_path / "models.yaml"
    path.write_text(f"local:\n{line}  model: qwen3:14b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="provider"):
        model_provider("local", path)


def test_project_configs_load_haiku_55_with_its_effort_and_without_sampling():
    path = ROOT / "configs" / "models.yaml"

    config = load_anthropic_config(path, section="haiku-5.5")

    assert config == AnthropicConfig(
        model="claude-haiku-5-5", timeout_s=60, max_retries=3, max_tokens=8192,
        effort="medium", accepts_sampling=False,
    )
    assert load_prices(path, section="haiku-5.5") == (0.1, 0.5)
    haiku_45 = load_anthropic_config(path, section="haiku-4.5")
    assert haiku_45.accepts_sampling and haiku_45.effort is None  # Haiku 4.5 keeps both samplings


def test_haiku_55_is_the_main_model_by_default():
    path = ROOT / "configs" / "models.yaml"

    assert parse_args([]).model == run.MAIN_MODEL == "haiku-5.5"
    assert load_anthropic_config(path).model == "claude-haiku-5-5"
    assert load_prices(path) == (0.1, 0.5)


def test_haiku_55_is_built_as_a_claude_extractor_and_names_its_results(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    extractor = build_extractor("haiku-5.5")

    assert isinstance(extractor, AnthropicExtractor)
    assert extractor.config.model == configured_model("haiku-5.5") == "claude-haiku-5-5"
    assert takes_sampling("haiku-4.5") and not takes_sampling("haiku-5.5")


def test_a_model_without_sampling_runs_only_with_the_default_sampling(
    tmp_path, monkeypatch, capsys
):
    make_texts(tmp_path / "texts")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("src.extraction.run.ROOT", tmp_path)
    monkeypatch.setattr("src.extraction.run.build_extractor", lambda model: StubExtractor())
    run_dir = tmp_path / "chosen"
    texts = ["--texts-dir", str(tmp_path / "texts"), "--run-dir", str(run_dir)]

    refused = main(["--model", "haiku-5.5", "--sampling", "precise", *texts])
    accepted = main(["--model", "haiku-5.5", "--sampling", "default", *texts])

    assert (refused, accepted) == (2, 0)
    assert "--sampling default" in capsys.readouterr().err
    assert [p.name for p in run_dir.glob("*.jsonl")] == ["claude-haiku-5-5__v2__default.jsonl"]


def test_without_a_sampling_option_a_model_gets_precise_or_its_default(tmp_path, monkeypatch):
    make_texts(tmp_path / "texts")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("src.extraction.run.ROOT", tmp_path)
    monkeypatch.setattr("src.extraction.run.build_extractor", lambda model: StubExtractor())
    run_dir = tmp_path / "chosen"
    texts = ["--texts-dir", str(tmp_path / "texts"), "--run-dir", str(run_dir)]

    codes = [main(["--model", model, *texts]) for model in ("haiku-4.5", "haiku-5.5")]

    assert codes == [0, 0]
    assert sorted(path.name for path in run_dir.glob("*.jsonl")) == [
        "claude-haiku-4-5__v2__precise.jsonl", "claude-haiku-5-5__v2__default.jsonl"
    ]


def test_runs_without_a_run_dir_land_where_make_metrics_looks():
    assert run.RESULTS_DIR == metrics.RESULTS_DIR


def test_main_saves_into_the_given_run_dir(tmp_path, monkeypatch):
    make_texts(tmp_path / "texts")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("src.extraction.run.ROOT", tmp_path)
    monkeypatch.setattr("src.extraction.run.configured_model", lambda model: "stub")
    monkeypatch.setattr("src.extraction.run.build_extractor", lambda model: StubExtractor())
    run_dir = tmp_path / "chosen"

    code = main(["--texts-dir", str(tmp_path / "texts"), "--run-dir", str(run_dir)])

    assert code == 0
    assert [path.name for path in run_dir.glob("*.jsonl")] == ["stub__v2__default.jsonl"]


def test_main_accepts_a_run_dir_given_relative_to_the_working_directory(tmp_path, monkeypatch):
    make_texts(tmp_path / "texts")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("src.extraction.run.ROOT", tmp_path)
    monkeypatch.setattr("src.extraction.run.configured_model", lambda model: "stub")
    monkeypatch.setattr("src.extraction.run.build_extractor", lambda model: StubExtractor())
    monkeypatch.chdir(tmp_path)

    code = main(["--texts-dir", "texts", "--run-dir", "results/smoke/check"])

    assert code == 0
    assert (tmp_path / "results" / "smoke" / "check" / "stub__v2__default.jsonl").exists()


def test_main_prints_the_full_path_of_a_run_dir_outside_the_project(
    tmp_path, monkeypatch, capsys
):
    make_texts(tmp_path / "texts")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("src.extraction.run.ROOT", tmp_path / "project")
    monkeypatch.setattr("src.extraction.run.configured_model", lambda model: "stub")
    monkeypatch.setattr("src.extraction.run.build_extractor", lambda model: StubExtractor())
    run_dir = tmp_path / "elsewhere"

    code = main(["--texts-dir", str(tmp_path / "texts"), "--run-dir", str(run_dir)])

    assert code == 0
    assert f"to {run_dir.resolve() / 'stub__v2__default.jsonl'}\n" in capsys.readouterr().out


def test_main_saves_every_run_to_its_own_folder_named_with_the_time(tmp_path, monkeypatch):
    make_texts(tmp_path / "texts")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("src.extraction.run.ROOT", tmp_path)
    monkeypatch.setattr("src.extraction.run.RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("src.extraction.run.configured_model", lambda model: "stub")
    extractor = StubExtractor()
    monkeypatch.setattr("src.extraction.run.build_extractor", lambda model: extractor)
    times = iter((STARTED, STARTED + timedelta(minutes=7)))
    monkeypatch.setattr("src.extraction.run.current_time", lambda: next(times))

    first = main(["--texts-dir", str(tmp_path / "texts")])
    second = main(["--texts-dir", str(tmp_path / "texts")])

    names = sorted(
        f"{path.parent.name}/{path.name}" for path in (tmp_path / "results").glob("*/*.jsonl")
    )
    assert (first, second) == (0, 0)
    assert names == [
        "2026-10-23T14-05-33/stub__v2__default.jsonl",
        "2026-10-23T14-12-33/stub__v2__default.jsonl",
    ]
    assert len(extractor.texts) == 4  # no answer is reused: both runs asked the model


def test_main_does_not_overwrite_the_results_of_the_same_prompt_version(
    tmp_path, monkeypatch, capsys
):
    make_texts(tmp_path / "texts")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("src.extraction.run.ROOT", tmp_path)
    monkeypatch.setattr("src.extraction.run.configured_model", lambda model: "stub")
    extractor = StubExtractor()
    monkeypatch.setattr("src.extraction.run.build_extractor", lambda model: extractor)
    args = ["--texts-dir", str(tmp_path / "texts"), "--run-dir", str(tmp_path / "run")]
    results = tmp_path / "run" / "stub__v2__default.jsonl"
    first = main(args)
    saved = results.read_text(encoding="utf-8")

    second = main(args)

    assert (first, second) == (0, 2)
    assert results.read_text(encoding="utf-8") == saved
    assert len(extractor.texts) == 2  # the second run did not ask the model
    assert "already exists" in capsys.readouterr().err


def test_main_reports_a_bad_prompt_version_without_traceback(tmp_path, monkeypatch, capsys):
    make_texts(tmp_path / "texts")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("src.extraction.run.build_extractor", lambda model: StubExtractor())
    prompt = tmp_path / "copied.md"
    prompt.write_text("---\nversion: v2/fix\n---\n\nВитягни умови.\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    code = main(
        ["--texts-dir", str(tmp_path / "texts"), "--prompt", str(prompt), "--run-dir", str(run_dir)]
    )

    assert code == 2
    assert "version" in capsys.readouterr().err
    assert not run_dir.exists()


def test_main_without_api_key_stops_before_writing_results(tmp_path, monkeypatch, capsys):
    make_texts(tmp_path / "texts")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("src.extraction.run.RESULTS_DIR", tmp_path / "results")

    code = main(["--texts-dir", str(tmp_path / "texts")])

    assert code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_gpt_needs_the_openrouter_key_not_the_anthropic_one(tmp_path, monkeypatch, capsys):
    make_texts(tmp_path / "texts")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("src.extraction.run.RESULTS_DIR", tmp_path / "results")

    code = main(["--model", "gpt-4o", "--texts-dir", str(tmp_path / "texts")])

    assert code == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_gpt_runs_without_the_anthropic_key(tmp_path, monkeypatch):
    make_texts(tmp_path / "texts")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("src.extraction.run.ROOT", tmp_path)
    monkeypatch.setattr("src.extraction.run.RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("src.extraction.run.build_extractor", lambda model: StubExtractor())

    code = main(["--model", "gpt-4o", "--texts-dir", str(tmp_path / "texts")])

    assert code == 0


def test_main_reports_a_rejected_openrouter_request_without_traceback(
    tmp_path, monkeypatch, capsys
):
    make_texts(tmp_path / "texts")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("src.extraction.run.RESULTS_DIR", tmp_path / "results")
    request = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

    class Rejected(StubExtractor):
        def extract(self, text: str, prompt: Prompt, sampling: Sampling) -> ExtractionResult:
            response = httpx2.Response(400, request=request)
            raise openai.BadRequestError("Invalid schema", response=response, body=None)

    monkeypatch.setattr("src.extraction.run.build_extractor", lambda model: Rejected())

    code = main(["--model", "gpt-4o", "--texts-dir", str(tmp_path / "texts")])

    assert code == 1
    assert "400" in capsys.readouterr().err


def test_unknown_sampling_name_is_rejected():
    with pytest.raises(KeyError):
        load_sampling("creative")
