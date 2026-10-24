import json
from pathlib import Path

import pytest

from src.extraction.metrics import (
    FieldGroup,
    Reference,
    bootstrap_ci,
    compute_metrics,
    cost_per_1000_docs,
    cost_usd,
    field_counts,
    format_field_tables,
    format_table,
    latest_run_dir,
    load_price_table,
    load_references,
    main,
    missing_references,
    percentile,
)
from src.extraction.scoring import Counts
from src.schemas.deal import Deal

ROOT = Path(__file__).resolve().parents[1]
DEAL = {"crop": "wheat", "quantity_t": 100, "buyer": "Зерно Поле"}  # 3 filled of 21 fields
REFERENCES = {
    "synthetic_001": Reference(deal=Deal.model_validate(DEAL), split="dev"),
    "synthetic_002": Reference(deal=Deal.model_validate(DEAL), split="dev"),
    "prozorro_001": Reference(deal=Deal.model_validate(DEAL), split="test", source="prozorro"),
    "own_001": Reference(deal=Deal.model_validate(DEAL), split="test", source="own"),
}
PRICES = {"claude-haiku-4-5": (1.0, 5.0)}
RUN_AT = "2026-10-23T10:00:00+03:00"
LATER_RUN_AT = "2026-10-24T09:30:00+03:00"


def record(
    doc_id: str,
    status: str = "ok",
    deal: dict[str, object] = DEAL,
    model: str = "claude-haiku-4-5",
    latency_s: float = 1.0,
    run_at: str = RUN_AT,
    attempts: int = 1,
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "run_at": run_at,
        "model": model,
        "prompt_version": "v1",
        "sampling": "precise",
        "status": status,
        "deal": deal if status == "ok" else None,
        "attempts": attempts,
        "latency_s": latency_s,
        "input_tokens": 1000,
        "output_tokens": 100,
    }


def doc(correct: int, filled: int = 10) -> Counts:
    return Counts(
        filled=filled, filled_correct=correct, empty=0, fabricated=0, total=filled, correct=correct
    )


def test_cost_counts_input_and_output_tokens_at_their_own_prices():
    assert cost_usd(1_000_000, 200_000, price_input=1.0, price_output=5.0) == pytest.approx(2.0)


def test_cost_per_1000_docs_ignores_error_results_without_tokens():
    records = [record("synthetic_001"), {**record("synthetic_002", status="error"),
                                         "input_tokens": 0, "output_tokens": 0}]

    assert cost_per_1000_docs(records, 1.0, 5.0) == pytest.approx((1000 + 5 * 100) / 1e6 * 1000)
    assert cost_per_1000_docs(records[1:], 1.0, 5.0) is None


def test_percentile_interpolates_between_ranks():
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)
    assert percentile([7.0], 95) == 7.0


def test_bootstrap_interval_is_reproducible_and_brackets_the_estimate():
    docs = [doc(10), doc(9), doc(5), doc(8), doc(10)]  # 42 of 50 = 84%

    low, high = bootstrap_ci(docs)

    assert bootstrap_ci(docs) == (low, high)
    assert low < 84.0 < high


def test_bootstrap_interval_of_identical_documents_has_zero_width():
    assert bootstrap_ci([doc(8), doc(8), doc(8)]) == (pytest.approx(80.0), pytest.approx(80.0))


def test_bootstrap_interval_is_none_without_filled_fields():
    assert bootstrap_ci([doc(0, filled=0)]) is None


def test_metrics_score_each_configuration_per_split_dev_first():
    records = [
        record("prozorro_001"),
        record("synthetic_001"),
        record("synthetic_002", deal={**DEAL, "buyer": "Агро Світ"}),
    ]

    dev, test = compute_metrics(records, REFERENCES, PRICES)

    assert (dev.split, dev.source, dev.docs) == ("dev", "synthetic", 2)
    assert (test.split, test.source, test.docs) == ("test", "prozorro", 1)
    assert dev.filled_accuracy == pytest.approx(100 * 5 / 6)
    assert dev.overall_pct == pytest.approx(100 * 41 / 42)
    assert test.filled_accuracy == pytest.approx(100.0)


def test_golden_sources_get_their_own_rows_and_one_row_for_all_of_them():
    records = [record("own_001", deal={**DEAL, "buyer": "Агро Світ"}), record("prozorro_001")]

    rows = compute_metrics(records, REFERENCES, PRICES)

    assert [(row.source, row.docs) for row in rows] == [("prozorro", 1), ("own", 1), ("all", 2)]
    assert rows[-1].filled_accuracy == pytest.approx(100 * 5 / 6)


def test_all_row_is_per_configuration():
    records = [record("own_001"), record("prozorro_001"), record("prozorro_001", model="lapa")]

    rows = compute_metrics(records, REFERENCES, PRICES)

    # lapa ran only on Prozorro, so it gets no `all` row of its own
    assert [(row.model, row.source) for row in rows] == [
        ("claude-haiku-4-5", "prozorro"), ("lapa", "prozorro"),
        ("claude-haiku-4-5", "own"), ("claude-haiku-4-5", "all"),
    ]


def test_failed_documents_stay_in_accuracy_and_count_as_failed():
    records = [record("synthetic_001"), record("synthetic_002", status="failed")]

    (metrics,) = compute_metrics(records, REFERENCES, PRICES)

    assert metrics.filled_accuracy == pytest.approx(50.0)
    assert metrics.fabricated_pct == pytest.approx(50.0)  # 18 golden nulls of the failed doc
    assert (metrics.valid_pct, metrics.failed_pct) == (50.0, 50.0)


def test_fixed_share_counts_valid_answers_that_needed_the_second_attempt():
    records = [record("synthetic_001"), record("synthetic_002", attempts=2)]

    (metrics,) = compute_metrics(records, REFERENCES, PRICES)

    assert (metrics.valid_pct, metrics.fixed_pct) == (100.0, 50.0)


def test_answer_that_failed_even_after_the_fix_is_failed_not_fixed():
    records = [record("synthetic_001", status="failed", attempts=2), record("synthetic_002")]

    (metrics,) = compute_metrics(records, REFERENCES, PRICES)

    assert (metrics.failed_pct, metrics.fixed_pct) == (50.0, 0.0)


def test_results_saved_before_the_fix_existed_count_as_one_attempt():
    old = {key: value for key, value in record("synthetic_001").items() if key != "attempts"}

    (metrics,) = compute_metrics([old], REFERENCES, PRICES)

    assert metrics.fixed_pct == 0.0


def test_fabricated_share_counts_values_where_golden_is_null():
    (metrics,) = compute_metrics([record("synthetic_001", deal={**DEAL, "basis": "FCA"})],
                                 REFERENCES, PRICES)

    assert metrics.fabricated_pct == pytest.approx(100 / 18)


def test_latency_percentiles_and_cost_per_configuration():
    records = [record(doc_id, latency_s=latency) for doc_id, latency in
               [("synthetic_001", 1.0), ("synthetic_002", 2.0),
                ("synthetic_001", 3.0), ("synthetic_002", 4.0)]]

    (metrics,) = compute_metrics(records, REFERENCES, PRICES)

    assert metrics.latency_p50_s == pytest.approx(2.5)
    assert metrics.latency_p95_s == pytest.approx(3.85)
    assert metrics.cost_per_1000_docs_usd == pytest.approx(1.5)


def test_only_the_latest_run_of_a_configuration_is_scored_and_its_time_is_shown():
    records = [
        record("synthetic_001", deal={**DEAL, "buyer": "Агро Світ"}, run_at=RUN_AT),
        record("synthetic_001", run_at=LATER_RUN_AT),
    ]

    (metrics,) = compute_metrics(records, REFERENCES, PRICES)

    assert (metrics.docs, metrics.run_at) == (1, LATER_RUN_AT)
    assert metrics.filled_accuracy == pytest.approx(100.0)


def test_a_later_run_on_one_split_does_not_hide_the_earlier_run_on_the_other():
    records = [
        record("prozorro_001", run_at=RUN_AT),
        record("synthetic_001", run_at=LATER_RUN_AT),
    ]

    dev, test = compute_metrics(records, REFERENCES, PRICES)

    assert (dev.split, dev.run_at) == ("dev", LATER_RUN_AT)
    assert (test.split, test.run_at) == ("test", RUN_AT)


def test_field_counts_use_the_latest_run_only():
    records = [
        record("prozorro_001", deal={**DEAL, "buyer": "Хтось інший"}, run_at=RUN_AT),
        record("prozorro_001", run_at=LATER_RUN_AT),
    ]

    counts = field_counts(records, REFERENCES, CONTRACTS)

    assert list(counts) == [("claude-haiku-4-5", "v1", "precise", LATER_RUN_AT)]
    assert counts[("claude-haiku-4-5", "v1", "precise", LATER_RUN_AT)]["buyer"].filled_correct == 1


def test_model_without_a_price_has_no_cost():
    (metrics,) = compute_metrics([record("synthetic_001", model="unknown")], REFERENCES, PRICES)

    assert metrics.cost_per_1000_docs_usd is None


def test_stored_deal_that_no_longer_fits_the_schema_names_the_document():
    records = [record("synthetic_001", deal={**DEAL, "quantity_t": -5})]

    with pytest.raises(ValueError, match="synthetic_001"):
        compute_metrics(records, REFERENCES, PRICES)


def test_missing_references_lists_documents_without_golden():
    records = [record("synthetic_001"), record("trader_001"), record("trader_001")]

    assert missing_references(records, REFERENCES) == ("trader_001",)


def test_documents_without_golden_are_left_out_of_metrics():
    records = [record("synthetic_001"), record("trader_001")]

    (metrics,) = compute_metrics(records, REFERENCES, PRICES)

    assert metrics.docs == 1


def test_references_come_from_the_data_files_with_their_split():
    references = load_references((ROOT / "data" / "synthetic", ROOT / "data" / "golden"))

    sources = [(reference.split, reference.source) for reference in references.values()]
    assert sources.count(("dev", "synthetic")) == 20
    assert sources.count(("test", "prozorro")) == 10
    assert sources.count(("test", "own")) == 7
    assert references["prozorro_001"].deal.buyer is not None


def test_references_stop_on_an_id_found_in_two_directories(tmp_path):
    entry = {"id": "doc_001", "split": "test", "source": "own", "deal": DEAL}
    for name in ("synthetic", "golden"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "doc_001.json").write_text(json.dumps(entry, ensure_ascii=False))

    with pytest.raises(ValueError, match="doc_001.*synthetic.*golden"):
        load_references((tmp_path / "synthetic", tmp_path / "golden"))


def test_price_table_maps_model_ids_from_the_config_to_prices():
    prices = load_price_table(ROOT / "configs" / "models.yaml")

    assert prices["claude-haiku-4-5"] == (1.0, 5.0)
    assert prices["lapa-v0.1.2-q4km"] == (0.0, 0.0)


def test_table_shows_interval_and_a_dash_for_missing_values():
    records = [record("synthetic_001"), record("synthetic_002", model="unknown")]

    table = format_table(compute_metrics(records, REFERENCES, PRICES))

    lines = table.splitlines()
    assert lines[0].startswith("| Модель | Промпт | Семплінг | Прогін | Набір | Док. |")
    assert "| Failed, % | Виправлено, % | p50, с |" in lines[0]
    row = (
        "| claude-haiku-4-5 | v1 | precise | 2026-10-23 10:00 | dev · synthetic | 1 "
        "| 100.0 (100.0–100.0) |"
    )
    assert row in table
    assert lines[-1].endswith("| — |")


def write_inputs(tmp_path: Path, records: list[dict[str, object]]) -> tuple[Path, Path]:
    results_dir, references_dir = tmp_path / "results", tmp_path / "golden"
    results_dir.mkdir()
    references_dir.mkdir()
    lines = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records)
    (results_dir / "claude-haiku-4-5__v1__precise.jsonl").write_text(lines, encoding="utf-8")
    reference = {"id": "synthetic_001", "split": "dev", "deal": DEAL}
    (references_dir / "synthetic_001.json").write_text(json.dumps(reference), encoding="utf-8")
    return results_dir, references_dir


def test_main_prints_the_table_and_warns_about_documents_without_golden(tmp_path, capsys):
    results_dir, references_dir = write_inputs(
        tmp_path, [record("synthetic_001"), record("trader_001")]
    )

    code = main(["--results-dir", str(results_dir), "--references", str(references_dir)])

    output = capsys.readouterr()
    assert code == 0
    row = "| claude-haiku-4-5 | v1 | precise | 2026-10-23 10:00 | dev · synthetic | 1 |"
    assert row in output.out
    assert "trader_001" in output.err


def test_main_also_writes_the_table_to_a_file_when_asked(tmp_path, capsys):
    results_dir, references_dir = write_inputs(tmp_path, [record("synthetic_001")])
    output = tmp_path / "metrics.md"

    code = main(
        ["--results-dir", str(results_dir), "--references", str(references_dir),
         "--output", str(output)]
    )

    assert code == 0
    assert output.read_text(encoding="utf-8") == capsys.readouterr().out


def test_main_reports_a_stored_deal_that_fails_the_schema(tmp_path, capsys):
    results_dir, references_dir = write_inputs(
        tmp_path, [record("synthetic_001", deal={**DEAL, "quantity_t": -5})]
    )

    code = main(["--results-dir", str(results_dir), "--references", str(references_dir)])

    assert code == 1
    assert "synthetic_001" in capsys.readouterr().err


CONFIG = ("claude-haiku-4-5", "v1", "precise", RUN_AT)
CONTRACTS = FieldGroup(title="Угоди", split="test", sources=("prozorro", "own"))
FIELD_RECORDS = [
    record("prozorro_001", deal={**DEAL, "buyer": "Агро Світ", "basis": "FCA"}),
    record("own_001"),
    record("synthetic_001"),  # dev: outside the group
]


def test_field_counts_score_each_field_over_the_documents_of_the_group():
    counts = field_counts(FIELD_RECORDS, REFERENCES, CONTRACTS)[CONFIG]

    assert counts["crop"] == Counts(
        filled=2, filled_correct=2, empty=0, fabricated=0, total=2, correct=2
    )
    assert (counts["buyer"].filled, counts["buyer"].filled_correct) == (2, 1)
    # golden has no basis: one answer invented FCA, the other left it null
    assert (counts["basis"].filled, counts["basis"].empty, counts["basis"].fabricated) == (0, 2, 1)


def test_field_table_shows_accuracy_with_fabricated_values_per_configuration():
    table = format_field_tables(FIELD_RECORDS, REFERENCES, (CONTRACTS,))

    lines = table.splitlines()
    assert lines[0] == "## Угоди (test · prozorro + own)"
    assert lines[2].startswith(
        "| Поле | Заповнено у golden | null у golden "
        "| claude-haiku-4-5 · v1 · precise · 2026-10-23 10:00 |"
    )
    assert "| buyer | 2 | 0 | 50.0 (1/2) · вигад. 0 |" in lines
    assert "| basis | 0 | 2 | — · вигад. 1 |" in lines


def test_field_table_total_row_matches_the_main_accuracy():
    table = format_field_tables(FIELD_RECORDS, REFERENCES, (CONTRACTS,))
    (all_row,) = (row for row in compute_metrics(FIELD_RECORDS, REFERENCES, PRICES)
                  if (row.split, row.source) == ("test", "all"))

    assert all_row.filled_accuracy == pytest.approx(100 * 5 / 6)
    assert "| **Разом** | 6 | 36 | 83.3 (5/6) · вигад. 1 |" in table.splitlines()


def test_field_table_of_a_group_without_results_is_left_out():
    only_dev = [record("synthetic_001")]

    assert format_field_tables(only_dev, REFERENCES, (CONTRACTS,)) == ""


def test_main_writes_the_field_table_when_asked(tmp_path):
    results_dir, references_dir = write_inputs(tmp_path, [record("prozorro_001")])
    reference = {"id": "prozorro_001", "split": "test", "source": "prozorro", "deal": DEAL}
    (references_dir / "prozorro_001.json").write_text(json.dumps(reference), encoding="utf-8")
    output = tmp_path / "metrics_by_field.md"

    code = main(
        ["--results-dir", str(results_dir), "--references", str(references_dir),
         "--by-field-output", str(output)]
    )

    assert code == 0
    assert output.read_text(encoding="utf-8").startswith("## Угоди (test · prozorro + own)")


def test_latest_run_dir_is_the_newest_folder_that_holds_runs(tmp_path):
    for name, has_runs in (
        ("2026-10-23T10-56-10", True), ("2026-10-25T09-10-02", True), ("2026-10-26T08-00-00", False)
    ):
        folder = tmp_path / name
        folder.mkdir()
        if has_runs:
            (folder / "stub__v1__precise.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("not a run folder", encoding="utf-8")

    assert latest_run_dir(tmp_path) == tmp_path / "2026-10-25T09-10-02"


def test_latest_run_dir_is_none_without_run_folders(tmp_path):
    assert latest_run_dir(tmp_path) is None
    assert latest_run_dir(tmp_path / "missing") is None


def test_main_reads_the_latest_run_folder_when_none_is_given(tmp_path, monkeypatch, capsys):
    results_dir, references_dir = write_inputs(tmp_path, [record("synthetic_001")])
    base = tmp_path / "stage1"
    (base / "2026-10-23T10-56-10").mkdir(parents=True)
    results_dir.rename(base / "2026-10-25T09-10-02")
    monkeypatch.setattr("src.extraction.metrics.RESULTS_DIR", base)

    code = main(["--references", str(references_dir)])

    assert code == 0
    assert "| claude-haiku-4-5 | v1 | precise |" in capsys.readouterr().out


def test_main_fails_clearly_without_results(tmp_path, capsys):
    results_dir, references_dir = write_inputs(tmp_path, [])

    code = main(["--results-dir", str(results_dir), "--references", str(references_dir)])

    assert code == 2
    assert "No results" in capsys.readouterr().err
