import json
from pathlib import Path

import pytest

from src.extraction.prompts import Prompt, load_prompt
from src.extraction.result import ExtractionResult, ExtractionStatus
from src.extraction.run import (
    DEFAULT_PROMPT,
    configured_model,
    load_anthropic_config,
    load_cached,
    load_ollama_config,
    load_prices,
    main,
    result_path,
    run_extraction,
    summarize,
    text_paths,
)
from src.extraction.sampling import Sampling, load_sampling

ROOT = Path(__file__).resolve().parents[1]
PROMPT = Prompt(version="v1", text="Витягни умови.", sha256="abc")
PRECISE = Sampling(name="precise", temperature=0.0, top_p=1.0)


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


def stored(doc_id: str, prompt: Prompt = PROMPT, **changes: object) -> dict[str, object]:
    """A saved answer as run_extraction writes it, with some fields changed."""
    result = StubExtractor().extract("пш", prompt, PRECISE)
    return {"doc_id": doc_id, **result.model_dump(mode="json"), **changes}


def write_results(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text("".join(lines), encoding="utf-8")


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


def test_result_path_names_model_prompt_and_sampling(tmp_path):
    path = result_path(tmp_path, "claude-haiku-4-5", "v1", "precise")

    assert path == tmp_path / "claude-haiku-4-5__v1__precise.jsonl"


def test_run_extraction_writes_one_json_line_per_document_with_doc_id(tmp_path):
    make_texts(tmp_path / "texts")
    output = tmp_path / "results" / "out.jsonl"

    records = run_extraction(
        StubExtractor(), text_paths(tmp_path / "texts", "dev"), PROMPT, PRECISE, output
    )

    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [line["doc_id"] for line in lines] == ["synthetic_001", "synthetic_002"]
    assert [line["status"] for line in lines] == ["error", "ok"]
    assert lines == records


def test_run_extraction_replaces_results_of_a_previous_run(tmp_path):
    make_texts(tmp_path / "texts")
    output = tmp_path / "out.jsonl"
    paths = text_paths(tmp_path / "texts", "dev")

    run_extraction(StubExtractor(), paths, PROMPT, PRECISE, output)
    run_extraction(StubExtractor(), paths, PROMPT, PRECISE, output)

    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


class FailingExtractor(StubExtractor):
    def extract(self, text: str, prompt: Prompt, sampling: Sampling) -> ExtractionResult:
        if len(self.texts) == 1:
            raise RuntimeError("401 invalid key")
        return super().extract(text, prompt, sampling)


def test_failed_rerun_keeps_the_previous_results(tmp_path):
    make_texts(tmp_path / "texts")
    output = tmp_path / "out.jsonl"
    paths = text_paths(tmp_path / "texts", "dev")
    run_extraction(StubExtractor(), paths, PROMPT, PRECISE, output)
    previous = output.read_text(encoding="utf-8")

    with pytest.raises(RuntimeError):
        run_extraction(FailingExtractor(), paths, PROMPT, PRECISE, output)

    assert output.read_text(encoding="utf-8") == previous


def test_load_cached_keeps_answers_for_the_same_prompt_and_sampling(tmp_path):
    path = tmp_path / "stub__v1__precise.jsonl"
    write_results(
        path,
        [
            stored("synthetic_001"),
            stored("synthetic_002", prompt_sha256="edited prompt"),
            stored("synthetic_003", sampling="varied"),
            stored("synthetic_004", status="error"),
            stored("synthetic_005", status="invalid"),
        ],
    )

    cached = load_cached(path, PROMPT, PRECISE)

    assert sorted(cached) == ["synthetic_001", "synthetic_005"]


def test_load_cached_without_a_results_file_is_empty(tmp_path):
    assert load_cached(tmp_path / "missing.jsonl", PROMPT, PRECISE) == {}


def test_run_extraction_sends_only_uncached_documents_to_the_model(tmp_path):
    make_texts(tmp_path / "texts")
    paths = text_paths(tmp_path / "texts", "all")
    output = tmp_path / "results" / "stub__v1__precise.jsonl"
    extractor = StubExtractor()

    records = run_extraction(
        extractor, paths, PROMPT, PRECISE, output, cached={"synthetic_002": stored("synthetic_002")}
    )

    assert extractor.texts == ["Договір поставки", "збій"]
    assert [record["doc_id"] for record in records] == [path.stem for path in paths]


def test_run_extraction_keeps_cached_answers_of_documents_outside_the_split(tmp_path):
    make_texts(tmp_path / "texts")
    output = tmp_path / "results" / "stub__v1__precise.jsonl"
    cached = {"prozorro_001": stored("prozorro_001")}

    run_extraction(
        StubExtractor(), text_paths(tmp_path / "texts", "dev"), PROMPT, PRECISE, output, cached
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["doc_id"] for line in lines] == [
        "synthetic_001",
        "synthetic_002",
        "prozorro_001",
    ]


def test_summary_counts_statuses_tokens_median_latency_and_cost():
    records = [
        {"status": "ok", "latency_s": 1.0, "input_tokens": 1000, "output_tokens": 100},
        {"status": "ok", "latency_s": 3.0, "input_tokens": 1000, "output_tokens": 100},
        {"status": "invalid", "latency_s": 2.0, "input_tokens": 1000, "output_tokens": 100},
    ]

    summary = summarize(records, price_input=1.0, price_output=5.0)

    assert summary["statuses"] == {"ok": 2, "invalid": 1, "error": 0}
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


def test_project_configs_load_the_provider_and_both_samplings():
    config = load_anthropic_config(ROOT / "configs" / "models.yaml")

    assert config.model == "claude-haiku-4-5"
    assert config.timeout_s > 0 and config.max_retries >= 1
    assert load_prices(ROOT / "configs" / "models.yaml") == (1.0, 5.0)
    assert load_sampling("precise").temperature == 0
    assert load_sampling("varied").temperature > 0


def test_project_configs_load_the_local_model_without_token_prices():
    config = load_ollama_config(ROOT / "configs" / "models.yaml")

    assert config.model == "lapa-v0.1.2-q4km"
    assert config.timeout_s > 0 and config.num_ctx >= 8192
    assert load_prices(ROOT / "configs" / "models.yaml", section="ollama") == (0.0, 0.0)


def test_main_with_local_model_needs_no_api_key(tmp_path, monkeypatch, capsys):
    make_texts(tmp_path / "texts")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("src.extraction.run.ROOT", tmp_path)  # main prints paths relative to ROOT
    monkeypatch.setattr("src.extraction.run.RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("src.extraction.run.build_extractor", lambda model: StubExtractor())
    monkeypatch.setattr("src.extraction.run.configured_model", lambda model: "stub")

    code = main(["--model", "lapa", "--texts-dir", str(tmp_path / "texts")])

    assert code == 0
    assert (tmp_path / "results" / "stub__v2__precise.jsonl").exists()
    assert "ANTHROPIC_API_KEY" not in capsys.readouterr().err


def test_main_with_every_answer_cached_needs_no_api_key_and_no_model(
    tmp_path, monkeypatch, capsys
):
    make_texts(tmp_path / "texts")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("src.extraction.run.ROOT", tmp_path)
    monkeypatch.setattr("src.extraction.run.RESULTS_DIR", tmp_path / "results")
    prompt = load_prompt(DEFAULT_PROMPT)
    output = result_path(tmp_path / "results", configured_model("claude"), "v2", "precise")
    docs = ("prozorro_001", "synthetic_001", "synthetic_002")
    write_results(output, [stored(doc, prompt) for doc in docs])

    def no_model(model: str) -> StubExtractor:
        raise AssertionError("every answer is cached, no model is needed")

    monkeypatch.setattr("src.extraction.run.build_extractor", no_model)

    code = main(["--split", "all", "--texts-dir", str(tmp_path / "texts")])

    assert code == 0
    assert "cached" in capsys.readouterr().out


def test_main_fresh_asks_the_model_again_despite_the_cache(tmp_path, monkeypatch):
    make_texts(tmp_path / "texts")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("src.extraction.run.ROOT", tmp_path)
    monkeypatch.setattr("src.extraction.run.RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("src.extraction.run.configured_model", lambda model: "stub")
    prompt = load_prompt(DEFAULT_PROMPT)
    output = result_path(tmp_path / "results", "stub", "v2", "precise")
    write_results(output, [stored("synthetic_001", prompt), stored("synthetic_002", prompt)])
    extractor = StubExtractor()
    monkeypatch.setattr("src.extraction.run.build_extractor", lambda model: extractor)

    code = main(["--fresh", "--texts-dir", str(tmp_path / "texts")])

    assert code == 0
    assert len(extractor.texts) == 2


def test_main_reports_an_unreachable_ollama_without_traceback(tmp_path, monkeypatch, capsys):
    make_texts(tmp_path / "texts")
    monkeypatch.setattr("src.extraction.run.RESULTS_DIR", tmp_path / "results")

    class Unreachable(StubExtractor):
        def extract(self, text: str, prompt: Prompt, sampling: Sampling) -> ExtractionResult:
            raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr("src.extraction.run.build_extractor", lambda model: Unreachable())

    code = main(["--model", "lapa", "--texts-dir", str(tmp_path / "texts")])

    assert code == 1
    assert "ollama serve" in capsys.readouterr().err


def test_main_without_api_key_stops_before_writing_results(tmp_path, monkeypatch, capsys):
    make_texts(tmp_path / "texts")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("src.extraction.run.RESULTS_DIR", tmp_path / "results")

    code = main(["--texts-dir", str(tmp_path / "texts")])

    assert code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_unknown_sampling_name_is_rejected():
    with pytest.raises(KeyError):
        load_sampling("creative")
