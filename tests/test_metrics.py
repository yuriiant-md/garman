import json
from pathlib import Path

import pytest

from src.extraction.metrics import (
    Reference,
    bootstrap_ci,
    compute_metrics,
    cost_per_1000_docs,
    cost_usd,
    format_table,
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
    "prozorro_001": Reference(deal=Deal.model_validate(DEAL), split="test"),
}
PRICES = {"claude-haiku-4-5": (1.0, 5.0)}


def record(
    doc_id: str,
    status: str = "ok",
    deal: dict[str, object] = DEAL,
    model: str = "claude-haiku-4-5",
    latency_s: float = 1.0,
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "model": model,
        "prompt_version": "v1",
        "sampling": "precise",
        "status": status,
        "deal": deal if status == "ok" else None,
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

    assert (dev.split, dev.docs, test.split, test.docs) == ("dev", 2, "test", 1)
    assert dev.filled_accuracy == pytest.approx(100 * 5 / 6)
    assert dev.overall_pct == pytest.approx(100 * 41 / 42)
    assert test.filled_accuracy == pytest.approx(100.0)


def test_failed_documents_stay_in_accuracy_and_count_as_failed():
    records = [record("synthetic_001"), record("synthetic_002", status="invalid")]

    (metrics,) = compute_metrics(records, REFERENCES, PRICES)

    assert metrics.filled_accuracy == pytest.approx(50.0)
    assert metrics.fabricated_pct == pytest.approx(50.0)  # 18 golden nulls of the failed doc
    assert (metrics.valid_pct, metrics.failed_pct) == (50.0, 50.0)


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


def test_model_without_a_price_has_no_cost():
    (metrics,) = compute_metrics([record("synthetic_001", model="unknown")], REFERENCES, PRICES)

    assert metrics.cost_per_1000_docs_usd is None


def test_stored_deal_that_no_longer_fits_the_schema_names_the_document():
    records = [record("synthetic_001", deal={**DEAL, "quantity_t": -5})]

    with pytest.raises(ValueError, match="synthetic_001"):
        compute_metrics(records, REFERENCES, PRICES)


def test_missing_references_lists_documents_without_golden():
    records = [record("synthetic_001"), record("own_001"), record("own_001")]

    assert missing_references(records, REFERENCES) == ("own_001",)


def test_documents_without_golden_are_left_out_of_metrics():
    (metrics,) = compute_metrics([record("synthetic_001"), record("own_001")], REFERENCES, PRICES)

    assert metrics.docs == 1


def test_references_come_from_the_data_files_with_their_split():
    references = load_references((ROOT / "data" / "synthetic", ROOT / "data" / "golden"))

    splits = [reference.split for reference in references.values()]
    assert (splits.count("dev"), splits.count("test")) == (20, 10)
    assert references["prozorro_001"].deal.buyer is not None


def test_price_table_maps_model_ids_from_the_config_to_prices():
    prices = load_price_table(ROOT / "configs" / "models.yaml")

    assert prices["claude-haiku-4-5"] == (1.0, 5.0)
    assert prices["lapa-v0.1.2-q4km"] == (0.0, 0.0)


def test_table_shows_interval_and_a_dash_for_missing_values():
    records = [record("synthetic_001"), record("synthetic_002", model="unknown")]

    table = format_table(compute_metrics(records, REFERENCES, PRICES))

    lines = table.splitlines()
    assert lines[0].startswith("| Модель | Промпт | Семплінг | Набір | Док. |")
    assert "| claude-haiku-4-5 | v1 | precise | dev | 1 | 100.0 (100.0–100.0) |" in table
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
        tmp_path, [record("synthetic_001"), record("own_001")]
    )

    code = main(["--results-dir", str(results_dir), "--references", str(references_dir)])

    output = capsys.readouterr()
    assert code == 0
    assert "| claude-haiku-4-5 | v1 | precise | dev | 1 |" in output.out
    assert "own_001" in output.err


def test_main_reports_a_stored_deal_that_fails_the_schema(tmp_path, capsys):
    results_dir, references_dir = write_inputs(
        tmp_path, [record("synthetic_001", deal={**DEAL, "quantity_t": -5})]
    )

    code = main(["--results-dir", str(results_dir), "--references", str(references_dir)])

    assert code == 1
    assert "synthetic_001" in capsys.readouterr().err


def test_main_fails_clearly_without_results(tmp_path, capsys):
    results_dir, references_dir = write_inputs(tmp_path, [])

    code = main(["--results-dir", str(results_dir), "--references", str(references_dir)])

    assert code == 2
    assert "No results" in capsys.readouterr().err
