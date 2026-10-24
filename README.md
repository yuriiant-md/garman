# Garman

English · [Українська](README.uk.md)

A Telegram bot for farmers: it extracts the terms of a grain trader's contract, finds risks with a reference to the contract clause or the law, and helps compare offers. This is a course project that grows in stages; stage 1 is ready for review: extracting deal terms from "dirty" texts and comparing paid models with a local one.

Stage 1 was run twice. The first matrix compared Claude Haiku 4.5 with the local Lapa 12B; Lapa got about 27% accuracy on filled fields of the real contracts, so stage 1.2 replaced it with Qwen3 14B and added GPT-4o through OpenRouter as a second paid model. `make run` now runs the stage 1.2 matrix into `results/stage1_2/`; the first matrix stays in `results/stage1/` and `make metrics` still reads it with `--results-dir` (step 4).

## Reproduce stage 1

### Requirements

- macOS or Linux with `make` and [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs Python 3.14 by itself on the first `make install`.
- To reproduce our numbers (step 3) you **need** an Anthropic API key for Claude Haiku 4.5, an [OpenRouter](https://openrouter.ai) API key for GPT-4o and [Ollama](https://ollama.com) with Qwen3 14B (a 9.3 GB download, about 12 GB of free memory while it runs). Without them only steps 1, 2 and 4 work: install, unit tests and viewing saved runs.

### Make commands

| Command | What it does | Needs key and Ollama |
| --- | --- | --- |
| `make install` | installs the dependencies (`uv sync`) | no |
| `make test` | unit tests on stub models: offline, a few seconds; they check the code, not the quality of the models | no |
| `make smoke` | live quick check: 12 configurations on the first 2 texts of each source (6 texts), saved in `results/smoke/<start time>/`; about $1 and 20 minutes | yes |
| `make run` | live full run: 12 configurations on all 37 texts, saved in `results/stage1_2/<start time>/` together with the two metrics tables; about $5 and 1.5 hours | yes |
| `make metrics` | prints the metrics table of the latest saved folder in `results/stage1_2/`; calls no model and writes nothing | no |

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

The tests run offline in a few seconds on stub models: no API key, no Ollama, no network. They check the code (parsing, validation, the fix attempt, the metrics arithmetic), not the quality of the models — the numbers come from step 3.

### 3. Run stage 1

```bash
cp .env.example .env    # put your keys in it: ANTHROPIC_API_KEY=... and OPENROUTER_API_KEY=...
ollama pull qwen3:14b
make run        # = uv run --env-file .env python main.py
```

`make run` asks the models for every configuration of stage 1.2 on all 37 texts — 3 models (Claude Haiku 4.5, GPT-4o, Qwen3 14B) × 2 prompts × 2 samplings (`precise`, `varied`) = 12 configurations. Nothing is cached: every run makes live API and Ollama calls. Our full run cost $5 ($1.8 for Claude, $3.2 for GPT-4o) and took about 1.5 hours on an Apple Silicon Mac: 10 minutes for Claude, 6 for GPT-4o and 80 for Qwen. The numbers differ a little from run to run: the paid models are not fully deterministic even at temperature 0, and prompt v2 contains the run date (`{today}`).

GPT-4o gets the schema as strict structured output (`response_format` with `strict: true`), so it cannot answer outside it; Claude gets it as a forced tool call, which the API does not compile strictly for a schema of this size. OpenAI has deprecated GPT-4o, so it is a reference point here, not a candidate for production.

Each launch is saved to a new folder named with its start time, e.g. `results/stage1_2/2026-10-27T14-05-33/`. It holds one file per configuration, named with the model, the prompt version and the sampling (`claude-haiku-4-5__v2__precise.jsonl`, `openai_gpt-4o__v2__precise.jsonl`; every line also carries `run_at`), and the two tables built from them: `metrics.md` and `metrics_by_field.md` (accuracy and fabricated values per field and configuration, on the test golden set). Earlier folders are never touched, so launches can be compared folder by folder.

A configuration whose model is needed but missing (no key in `.env`, Ollama not running) is skipped with the reason; the others still run, and `make run` ends with exit code 1 and the list of what it did not finish.

For a quick check use `make smoke` (= `python main.py --per-source 2`): the same 12 configurations on the first 2 texts of each source (6 texts), about $1 and 20 minutes, most of it Qwen. It saves into `results/smoke/<start time>/`, which git ignores and `make metrics` does not read. The numbers of 6 texts prove nothing — it only shows that everything works together. Run it before the full run to see that your keys, Ollama and the whole chain work.

### 4. View saved runs

```bash
make metrics    # = uv run python -m src.extraction.metrics
```

This only re-reads results that are already saved, so it needs neither an API key nor Ollama — it reproduces nothing by itself, the numbers come from step 3. It scores the saved runs of the latest folder in `results/stage1_2/` against the golden labels and prints the metrics table (nothing is written, so a saved table never changes afterwards). Pick another launch with `--results-dir <folder>`; add `--output <file> --by-field-output <file>` to write the tables. To compare your run with ours, our stage 1.2 run is in `results/stage1_2/2026-10-24T18-41-35/`. The first stage-1 matrix (Claude and Lapa, 8 configurations) is in `results/stage1/2026-10-23T11-46-35/` (`2026-10-23T10-56-10/` is an earlier run of 4 configurations with the `precise` sampling only):

```bash
uv run python -m src.extraction.metrics --results-dir results/stage1/2026-10-23T11-46-35
```

### Separate commands

```bash
uv run --env-file .env python -m src.extraction.run --split all                 # one configuration: Claude, prompt v2, precise
uv run --env-file .env python -m src.extraction.run --model gpt --split all     # GPT-4o, prompt v2, precise
uv run python -m src.extraction.run --model qwen --prompt prompts/extract_v1.md   # Qwen, prompt v1, dev texts
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
5. **A failed answer** (`failed` or `error`) is not dropped: all its fields are wrong, and the fields that are empty in the golden deal count as fabricated, so a failure never looks better than a mistake.
6. **Validity** is reported separately: valid, failed, and "fixed" — the share of documents that became valid only after the one fix attempt (it is part of valid, not of failed).
7. **`varied`** runs once per launch, like `precise`, so its spread shows only by comparing launches; 3 repeats on the golden set with mean and spread are planned for the final run.
8. **Separately per set:** synthetic texts (dev), Prozorro and the author's own contracts (test); the model is chosen on all golden documents together.
9. **Noise:** accuracy on filled fields comes with a 95% bootstrap interval — 1000 resamples of documents with replacement, fixed seed. When the intervals of two configurations overlap, no advantage is claimed.
10. **Cost** — tokens of all attempts × the price in `configs/models.yaml`, per 1000 answers without `error`; $0 for the local models.

## Where things are

### Stage 1 requirements

| Requirement | Where to look | Status |
| --- | --- | --- |
| 30 "dirty" texts without personal data | `data/texts/` — 37 texts: `synthetic_001–020.txt` (dev), `prozorro_001–010.txt` and the author's own contracts `own_001–007.txt` (test); how they were anonymised — `data/README.md` | done |
| 10 hand-labelled texts (golden set) | `data/golden/`: `prozorro_001–010.json` and `own_001–007.json`, labelling rules — `data/golden/RULES.md` | Prozorro checked by the author; `own` drafted by Claude, being checked by the author |
| Schema: ≥ 6 fields, ≥ 1 enum, a nested object or a list | `src/schemas/deal.py` (10 fields, 9 enums, 5 nested objects, the `penalties` list), tests — `tests/test_deal.py` | done |
| Provider model through an API with structured output, retry and timeout | `src/extraction/anthropic_extractor.py` (Claude: the schema as a forced tool call), `src/extraction/openrouter_extractor.py` (GPT-4o: strict structured output), settings — `configs/models.yaml` | done |
| Local model through a shared interface | `src/extraction/ollama_extractor.py` (Qwen3 14B; Lapa 12B in the first matrix); the `Extractor` interface — `src/extraction/run.py` | done |
| temperature and top_p in a config, ≥ 2 configurations | `configs/sampling.yaml` (`precise`, `varied`) | done: both samplings run in every `make run` |
| Metrics: per-field accuracy, % valid, p50/p95 latency, cost per 1000 documents | `src/extraction/metrics.py` (table), `src/extraction/scoring.py` (comparison with the golden deal); tests — `tests/test_metrics.py`, `tests/test_scoring.py` | done; numbers — `results/stage1_2/<start time>/metrics.md` and `metrics_by_field.md` |
| Prompts with a version header, ≥ 2 versions, `CHANGELOG` | `prompts/extract_v1.md`, `prompts/extract_v2.md`, `prompts/CHANGELOG.md` | done: v1 is the baseline, v2 is built from the v1 errors on dev |
| Every answer is validated by the schema; invalid → one fix attempt → `failed` | `src/extraction/validation.py` (the fix request), `anthropic_extractor.py`, `openrouter_extractor.py` and `ollama_extractor.py` (the second call); tests — `tests/test_validation.py` and the extractor tests | done; the "Виправлено, %" column of the metrics table counts the answers saved by it |
| Every result logs the model, prompt version and sampling | record fields — `src/extraction/result.py`; the records — `results/stage1_2/*/*.jsonl` (and `results/stage1/*/*.jsonl`) | done |
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
  openrouter_extractor.py  GPT-4o through OpenRouter (OpenAI SDK)
  ollama_extractor.py      local models through Ollama: Qwen3 14B (Lapa 12B in the first matrix)
  run.py                   the shared Extractor interface, one configuration, one saved run
  messages.py              the same user message with the document for both models
  prompts.py               reads a prompt with its version and file hash
  sampling.py              reads configs/sampling.yaml
  validation.py            checks an answer against the Deal schema; the fix request text
  result.py                what is recorded about each answer
  scoring.py               compares one answer with the golden deal
  metrics.py               metrics table model × prompt × sampling
scripts/
  generate_synthetic.py    deals for the synthetic set (+ synthetic_render.py, synthetic_checks.py)
  fetch_prozorro.py        search for contracts in Prozorro
  pdf_to_text.py           PDF → anonymised text
  check_golden.py          validates the golden labels
tests/                     pytest tests: make test
results/stage1_2/          one folder per launch, named with its start time: a *.jsonl file per configuration (one line per document) and the metrics tables (metrics.md, metrics_by_field.md)
results/stage1/            the same for the first matrix (Claude and Lapa)
```

The author keeps the work plan locally (`docs/`, outside the repository); its metric rules were moved to "How the metrics are counted".
