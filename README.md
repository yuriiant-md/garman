# Garman

English · [Українська](README.uk.md)

A Telegram bot for farmers: it extracts the terms of a grain trader's contract, finds risks with a reference to the contract clause or the law, and helps compare offers. This is a course project that grows in stages; stage 1 is ready for review: extracting deal terms from "dirty" texts and comparing a paid model with a local one.

## Reproduce stage 1

### Requirements

- macOS or Linux with `make` and [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs Python 3.14 by itself on the first `make install`.
- To run the models (step 3): an Anthropic API key for Claude Haiku 4.5 and [Ollama](https://ollama.com) with Lapa 12B (a 7.3 GB download, about 12 GB of free memory while it runs).

### 1. Install

```bash
git clone <repository-url> garman
cd garman
make install    # = uv sync
```

### 2. Test

```bash
make test       # = uv run pytest -q
```

All tests run offline in a few seconds: no API key, no Ollama, no network.

### 3. Run stage 1

```bash
cp .env.example .env    # put your key in it: ANTHROPIC_API_KEY=...
ollama pull hf.co/lapa-llm/lapa-v0.1.2-instruct-GGUF:Q4_K_M
ollama cp hf.co/lapa-llm/lapa-v0.1.2-instruct-GGUF:Q4_K_M lapa-v0.1.2-q4km
make run        # = uv run --env-file .env python main.py
```

`make run` asks the models for every configuration of stage 1 on all 37 texts — 2 models × 2 prompts × the `precise` sampling. Nothing is cached: every run makes live API and Ollama calls. A full run costs about $1 for Claude and takes about 25 minutes for Lapa on an Apple Silicon Mac. The numbers differ a little from run to run: Claude is not fully deterministic even at temperature 0, and prompt v2 contains the run date (`{today}`).

Each launch is saved to a new folder named with its start time, e.g. `results/stage1/2026-10-27T14-05-33/`. It holds one file per configuration, named with the model, the prompt version and the sampling (`claude-haiku-4-5__v2__precise.jsonl`; every line also carries `run_at`), and the two tables built from them: `metrics.md` and `metrics_by_field.md` (accuracy and fabricated values per field and configuration, on the test golden set). Earlier folders are never touched, so launches can be compared folder by folder.

A configuration whose model is needed but missing (no key in `.env`, Ollama not running) is skipped with the reason; the others still run, and `make run` ends with exit code 1 and the list of what it did not finish.

### 4. Rebuild the tables from saved runs

```bash
make metrics    # = uv run python -m src.extraction.metrics
```

This needs neither an API key nor Ollama: it scores the saved runs of the latest folder in `results/stage1/` against the golden labels and prints the metrics table (nothing is written, so a saved table never changes afterwards). Pick another launch with `--results-dir results/stage1/<start time>`; add `--output <file> --by-field-output <file>` to write the tables.

### Separate commands

```bash
uv run --env-file .env python -m src.extraction.run --split all                 # one configuration: Claude, prompt v2, precise
uv run python -m src.extraction.run --model lapa --prompt prompts/extract_v1.md   # Lapa, prompt v1, dev texts
uv run python -m src.extraction.metrics                                          # the metrics table only (= make metrics)
uv run python -m scripts.check_golden                                            # validate the golden labels
```

`--split` is `dev` (synthetic texts), `test` (golden set) or `all`; `--sampling` is a name from `configs/sampling.yaml`. Every run is saved to a new file; none is reused.

Building the golden set from Prozorro contracts (details in `data/README.md`):

```bash
uv run python -m scripts.fetch_prozorro    # candidates in data/raw/prozorro_candidates/
uv run python -m scripts.pdf_to_text       # data/raw/prozorro/*.pdf → data/texts/prozorro_NNN.txt
```

## How the metrics are counted

The rules were written before the first run. Code: `src/extraction/scoring.py` (one document) and `src/extraction/metrics.py` (the table); every rule has its own test in `tests/test_scoring.py` and `tests/test_metrics.py`.

1. **The unit is a leaf field** of the deal (`price.amount_per_t`, `payment.trigger`). An empty nested object equals one whose fields are all `null`. `delivery.period_raw` is not scored.
2. **Comparison:** enums, booleans and dates must match exactly; numbers to the hundredth (9200 = 9200.004, but 0.1 ≠ 0.11); strings after normalisation (case, quotes, apostrophes, the legal form ТОВ, ТзОВ, ПрАТ, АТ, ДП, ПП, ФГ) with similarity ≥ 0.85 (`difflib`). The threshold was set on the OCR noise of the Prozorro golden set: a name corrected by the model scores 0.87–0.99, a name in another grammatical case 0.82, a name in Latin script 0. Known flaw: names that differ only in a digit also match.
3. **Penalties** are paired by (`payer`, `trigger`, `unit`) regardless of order; among several candidates the one that matches the golden one on more of `amount` and `cap_pct` wins. An unpaired golden penalty has all its fields wrong; an extra one has its values counted as fabricated.
4. **Three numbers instead of one:** accuracy on filled fields is the main one — only fields that are not `null` in the golden deal; fabricated, % — the model gave a value where the golden deal has `null`; overall — all fields, for reference. Otherwise an "everything `null`" answer would score high. Accuracy is summed over the fields of all documents.
5. **An invalid answer** (`invalid` or `error`) is not dropped: all its fields are wrong, and the fields that are empty in the golden deal count as fabricated, so a failure never looks better than a mistake.
6. **Validity** is reported separately: valid and failed; "valid after the fix attempt" comes with the fix attempt.
7. **`varied`** — 3 repeats on the golden set, mean and spread; `precise` — one run (comes with the final run).
8. **Separately per set:** synthetic texts (dev), Prozorro and the author's own contracts (test); the model is chosen on all golden documents together.
9. **Noise:** accuracy on filled fields comes with a 95% bootstrap interval — 1000 resamples of documents with replacement, fixed seed. When the intervals of two configurations overlap, no advantage is claimed.
10. **Cost** — tokens of all attempts × the price in `configs/models.yaml`, per 1000 answers without `error`; $0 for the local Lapa.

## Where things are

### Stage 1 requirements

| Requirement | Where to look | Status |
| --- | --- | --- |
| 30 "dirty" texts without personal data | `data/texts/` — 37 texts: `synthetic_001–020.txt` (dev), `prozorro_001–010.txt` and the author's own contracts `own_001–007.txt` (test); how they were anonymised — `data/README.md` | done |
| 10 hand-labelled texts (golden set) | `data/golden/`: `prozorro_001–010.json` and `own_001–007.json`, labelling rules — `data/golden/RULES.md` | Prozorro checked by the author; `own` drafted by Claude, being checked by the author |
| Schema: ≥ 6 fields, ≥ 1 enum, a nested object or a list | `src/schemas/deal.py` (10 fields, 9 enums, 5 nested objects, the `penalties` list), tests — `tests/test_deal.py` | done |
| Provider model through an API with structured output, retry and timeout | `src/extraction/anthropic_extractor.py` (the schema as a forced tool call), settings — `configs/models.yaml` | done |
| Local model through a shared interface | `src/extraction/ollama_extractor.py`; the `Extractor` interface — `src/extraction/run.py` | done |
| temperature and top_p in a config, ≥ 2 configurations | `configs/sampling.yaml` (`precise`, `varied`) | config done; the `varied` run comes next |
| Metrics: per-field accuracy, % valid, p50/p95 latency, cost per 1000 documents | `src/extraction/metrics.py` (table), `src/extraction/scoring.py` (comparison with the golden deal); tests — `tests/test_metrics.py`, `tests/test_scoring.py` | done; numbers — `results/stage1/<start time>/metrics.md` and `metrics_by_field.md` |
| Prompts with a version header, ≥ 2 versions, `CHANGELOG` | `prompts/extract_v1.md`, `prompts/extract_v2.md`, `prompts/CHANGELOG.md` | done: v1 is the baseline, v2 is built from the v1 errors on dev |
| Every answer is validated by the schema; invalid → one fix attempt → `failed` | `src/extraction/validation.py` | schema check done; the fix attempt comes next |
| Every result logs the model, prompt version and sampling | record fields — `src/extraction/result.py`; the records — `results/stage1/*.jsonl` | done |
| `REPORT.md`: table, error analysis, production choice | — | next |
| One command to run: `make run` (= `python main.py`) | `Makefile`, `main.py` | done |

### Folders and files

```
Makefile                   make install / test / run / metrics
main.py                    stage 1 in one command: the whole matrix, then the metrics table
configs/
  models.yaml              models, timeout, retry, prices per 1M tokens
  sampling.yaml            sampling configurations precise and varied
data/
  texts/                   the 37 texts the models get
  synthetic/               golden Deal and "dirtiness" parameters of each synthetic text
  golden/                  golden deals of the Prozorro and own contracts + RULES.md
  prozorro_sources.csv     which Prozorro tender each contract comes from
  own_sources.csv          which PDF and contract number each own contract comes from
  README.md                how the data was collected, anonymised and labelled
  raw/                     original PDFs and masking check files (not in git)
prompts/
  extract_v1.md            baseline prompt, version in the header
  extract_v2.md            prompt with per-field rules; {today} becomes the run date
  CHANGELOG.md             what changed between versions and why
src/schemas/
  deal.py                  the Deal schema
src/extraction/
  anthropic_extractor.py   Claude Haiku 4.5 through the Anthropic API
  ollama_extractor.py      Lapa 12B through Ollama
  run.py                   the shared Extractor interface, one configuration, one saved run
  messages.py              the same user message with the document for both models
  prompts.py               reads a prompt with its version and file hash
  sampling.py              reads configs/sampling.yaml
  validation.py            checks an answer against the Deal schema
  result.py                what is recorded about each answer
  scoring.py               compares one answer with the golden deal
  metrics.py               metrics table model × prompt × sampling
scripts/
  generate_synthetic.py    deals for the synthetic set (+ synthetic_render.py, synthetic_checks.py)
  fetch_prozorro.py        search for contracts in Prozorro
  pdf_to_text.py           PDF → anonymised text
  check_golden.py          validates the golden labels
tests/                     pytest tests: make test
results/stage1/            one folder per launch, named with its start time: a *.jsonl file per configuration (one line per document) and the metrics tables (metrics.md, metrics_by_field.md)
```

The author keeps the work plan locally (`docs/`, outside the repository); its metric rules were moved to "How the metrics are counted".
