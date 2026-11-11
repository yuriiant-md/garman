# Garman

English · [Українська](README.uk.md)

A Telegram bot for farmers: it extracts the terms of a grain trader's contract, finds risks with a reference to the contract clause or the law, and helps compare offers. This is a course project that grows in stages; stage 1 is ready for review: extracting deal terms from "dirty" texts and comparing paid models with a local one.

Stage 1 grew in three steps. The first matrix compared Claude Haiku 4.5 with the local Lapa 12B; Lapa got about 27% accuracy on filled fields of the real contracts, so stage 1.2 replaced it with Qwen3 14B and added GPT-4o through OpenRouter as a second paid model. Stage 1.3 changed the schema: a contract may name several grades, each with its own price per tonne and its own quality limits (test weight, gluten, falling number, …), and added Claude Haiku 5.5, which became the main model (`REPORT.md`, section 6). Since stage 1.3 the code has no local model: the course no longer requires one, and the local runs stay in `results/` as history. `make run` now runs the stage 1.3 matrix into `results/stage1_3/`. The earlier matrices stay in `results/stage1/` and `results/stage1_2/`; they were scored with the schema before grades, so they are re-read with the code of the `stage-1` tag (step 4).

Stage 2, RAG with citations, is in progress: the corpus, the chunks, the search, the answers with sources and their checks by rules and by an LLM judge are ready ("How documents are chunked", "How search works", "How answers work" and "How answers are checked" below). The judge is calibrated on the author's manual marks: it agrees with them on 95% of the correct checks and 83% of the claims (80% needed). The stage 2 report, in Ukrainian: `REPORT_STAGE2.md`.

## Reproduce stage 1

### Requirements

- macOS or Linux with `make` and [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs Python 3.14 by itself on the first `make install`.
- To reproduce our numbers (step 3) you **need** an Anthropic API key for Claude Haiku 5.5 and 4.5 and an [OpenRouter](https://openrouter.ai) API key for GPT-4o. Without them only steps 1, 2 and 4 work: install, unit tests and viewing saved runs.

### Make commands

| Command | What it does | Needs a key |
| --- | --- | --- |
| `make install` | installs the dependencies (`uv sync`) | no |
| `make test` | unit tests on stub models: offline, a few seconds; they check the code, not the quality of the models | no |
| `make smoke` | live quick check: 10 configurations on the first 2 texts of each source (6 texts), saved in `results/smoke/<start time>/`; about $1.6 and 8 minutes | yes |
| `make run` | live full run: 10 configurations on all 48 texts, saved in `results/stage1_3/<start time>/` together with the two metrics tables; about $12 and 50 minutes of requests (our stage 1.3 run: $11.9) | yes |
| `make metrics` | prints the metrics table of the latest saved folder in `results/stage1_3/`; calls no model and writes nothing | no |

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

The tests run offline in a few seconds on stub models: no API key, no network. They check the code (parsing, validation, the fix attempt, the metrics arithmetic), not the quality of the models — the numbers come from step 3.

### 3. Run stage 1

```bash
cp .env.example .env    # put your keys in it: ANTHROPIC_API_KEY=... and OPENROUTER_API_KEY=...
make run        # = uv run --env-file .env python main.py
```

`make run` asks the models for every configuration of stage 1.3 on all 48 texts — the main model Claude Haiku 5.5 × 2 prompts with the `default` sampling, plus the two models it is compared with (Claude Haiku 4.5, GPT-4o) × 2 prompts × 2 samplings (`precise`, `varied`) = 10 configurations. Haiku 5.5 rejects any temperature and top_p but its own, so it gets neither `precise` nor `varied`: `default` sends no sampling parameters at all. Nothing is cached: every run makes live API calls. The full stage 1.2 run (12 configurations) cost $5 ($1.8 for Claude Haiku 4.5, $3.2 for GPT-4o) and took about 1.5 hours on an Apple Silicon Mac: 10 minutes for Claude, 6 for GPT-4o and 80 for Qwen. The stage 1.3 run (the same 10 configurations on 48 texts) cost $11.9 ($4.1 for Claude Haiku 4.5, $0.19 for Claude Haiku 5.5, $7.7 for GPT-4o) and took about 50 minutes of requests: 23 for Haiku 4.5, 20 for GPT-4o, 6 for Haiku 5.5. The numbers differ a little from run to run: the paid models are not fully deterministic even at temperature 0, and prompt v2 contains the run date (`{today}`).

To run only some models, name them with `--models`; the keys name the models: `haiku-5.5` (the main one), `haiku-4.5`, `gpt-4o`. They are the sections of `configs/models.yaml`, so a new section is a new key. `uv run --env-file .env python main.py --models haiku-5.5` runs only the main model: 2 configurations, about $0.2 and 6 minutes. `--models` also works with `--per-source` (below).

GPT-4o gets the schema as strict structured output (`response_format` with `strict: true`), so it cannot answer outside it; both Claude models get it as a forced tool call, which the API does not compile strictly for a schema of this size. OpenAI has deprecated GPT-4o, so it is a reference point here, not a candidate for production.

Each launch is saved to a new folder named with its start time, e.g. `results/stage1_3/2026-10-27T14-05-33/`. It holds one file per configuration, named with the model, the prompt version and the sampling (`claude-haiku-4-5__v2__precise.jsonl`, `claude-haiku-5-5__v2__default.jsonl`, `openai_gpt-4o__v2__precise.jsonl`; every line also carries `run_at`), and the two tables built from them: `metrics.md` and `metrics_by_field.md` (accuracy, fabricated values and extra list items per field and configuration, on the test golden set). Earlier folders are never touched, so launches can be compared folder by folder. A results file is never overwritten either: a second run of the same model, prompt version and sampling into the same folder stops with a message, so a new prompt needs its own version in the header (Latin letters, digits, `.`, `_` and `-` only, since the version is part of the file name).

A configuration whose model is needed but missing (no key in `.env`) is skipped with the reason; the others still run, and `make run` ends with exit code 1 and the list of what it did not finish.

For a quick check use `make smoke` (= `python main.py --per-source 2`): the same 10 configurations on the first 2 texts of each source (6 texts), about $1.6 and 8 minutes. It saves into `results/smoke/<start time>/`, which git ignores and `make metrics` does not read. The numbers of 6 texts prove nothing — it only shows that everything works together. Run it before the full run to see that your keys and the whole chain work.

### 4. View saved runs

```bash
make metrics    # = uv run python -m src.extraction.metrics
```

This only re-reads results that are already saved, so it needs no API key — it reproduces nothing by itself, the numbers come from step 3. It scores the saved runs of the latest folder in `results/stage1_3/` against the golden labels and prints the metrics table (nothing is written, so a saved table never changes afterwards). Pick another launch with `--results-dir <folder>`; add `--output <file> --by-field-output <file>` to write the tables. The full stage 1.3 run — 10 configurations of Claude Haiku 4.5, Claude Haiku 5.5 and GPT-4o, without the local model — is in `results/stage1_3/2026-10-26T16-10-58/` (`REPORT.md`, section 5); the first one, Claude Haiku 5.5 only, is in `results/stage1_3/2026-10-25T13-25-34/`.

The earlier matrices were scored with the schema before grades. Their answers keep the price per tonne in `price` and the grade and quality limits in `quality`, fields the current schema no longer has, so the current code reads them as empty and counts them wrong. Re-read them with the code of the `stage-1` tag. Our stage 1.2 run is in `results/stage1_2/2026-10-24T18-41-35/`; the first matrix (Claude and Lapa, 8 configurations) is in `results/stage1/2026-10-23T11-46-35/` (`2026-10-23T10-56-10/` is an earlier run of 4 configurations with the `precise` sampling only):

```bash
git checkout stage-1
uv run python -m src.extraction.metrics --results-dir results/stage1_2/2026-10-24T18-41-35
git checkout main    # back to the current code
```

### Separate commands

```bash
uv run --env-file .env python -m src.extraction.run --split all                  # one configuration: the main model Haiku 5.5, prompt v2
uv run --env-file .env python -m src.extraction.run --model gpt-4o --split all   # GPT-4o, prompt v2, precise
uv run --env-file .env python -m src.extraction.run --model haiku-4.5 --sampling varied --split all   # Haiku 4.5, prompt v2, varied
uv run --env-file .env python -m src.extraction.run --model haiku-4.5 --prompt prompts/extract_v1.md   # Haiku 4.5, prompt v1, precise, dev texts
uv run python -m src.extraction.metrics                                          # the metrics table only (= make metrics)
uv run python -m scripts.check_golden                                            # validate the golden labels
```

`--model` is `haiku-5.5` (the default), `haiku-4.5` or `gpt-4o`; `--split` is `dev` (synthetic texts), `test` (golden set) or `all`; `--sampling` is a name from `configs/sampling.yaml`, by default `precise`, or `default` for Haiku 5.5, which takes no other. Every run is saved to a new file; none is reused.

Building the golden set from Prozorro contracts (details in `data/README.md`):

```bash
uv run python -m scripts.fetch_prozorro    # candidates in data/raw/prozorro_candidates/
uv run python -m scripts.pdf_to_text       # data/raw/prozorro/*.pdf → data/texts/prozorro_NNN.txt
```

Collecting the stage 2 corpus (45 documents listed in `data/rag_sources.csv`, details in `data/README.md`):

```bash
uv run python -m scripts.fetch_rag_sources  # laws and Supreme Court decisions → data/raw/rag/; exit 1 while anything is missing
uv run python -m scripts.ingest_rag         # clean, anonymised documents → data/rag/; nothing written while any document has a problem
uv run python -m scripts.chunk_rag          # chunks for the index → data/rag/chunks.jsonl and a comparison with fixed-size chunks
```

The stage 2 search index and its quality (details in "How search works"):

```bash
docker compose up -d qdrant                         # Qdrant 1.20 on 127.0.0.1:6333; the data stays in the qdrant_storage volume
caffeinate -i uv run python -m scripts.index_rag    # both collections, about 7 minutes on CPU; a fresh collection is skipped, --force rebuilds
caffeinate -i uv run python -m scripts.eval_search  # 111 searches, about 2 minutes → results/stage2/search/<start time>/
```

`caffeinate -i` keeps a Mac awake: a sleeping Mac pauses the run. On Linux, drop it. The first run of `index_rag` downloads bge-m3 (2.3 GB), the first run of `eval_search` the reranker (0.57 GB), both into `~/.cache/fastembed`; no key is needed.

The stage 2 answers (paid: Claude Haiku 5.5, about $0.01 for 26 questions; Qdrant must be running with a fresh index; details in "How answers work"):

```bash
caffeinate -i uv run --env-file .env python -m scripts.ask_rag                                # all 26 questions → results/stage2/answers/<start time>/
caffeinate -i uv run --env-file .env python -m scripts.ask_rag --ids q01,t05                  # only these questions
caffeinate -i uv run --env-file .env python -m scripts.ask_rag --prompt prompts/answer_v2.md  # another prompt than the one in configs/answer.yaml
```

Checking a saved run of the answers (no Qdrant, no search; details in "How answers are checked"):

```bash
uv run python -m scripts.score_answers                                            # rules, free: scores.md in the newest run
caffeinate -i uv run --env-file .env python -m scripts.judge_answers --limit 2    # the judge on 2 questions, about $0.02: a check that it works
caffeinate -i uv run --env-file .env python -m scripts.judge_answers              # the judge on the whole run, about $0.19: judge.jsonl and judge.md
uv run python -m scripts.judge_sheet --seen q01,q05                               # free: the sheet for manual marks, data/rag/judge_marks.yaml
uv run python -m scripts.calibrate_judge                                          # free: the marks against judge.jsonl, calibration.md
```

All take a run folder as an argument (`results/stage2/answers/2026-11-10T10-08-19`); without one they take the newest run, and `calibrate_judge` the run named in the sheet. `judge_sheet` keeps a sheet that exists: it may hold marks (`--force` writes over it).

## How the metrics are counted

The rules were written before the first run; the rules for grades and quality indicators were added in stage 1.3. Code: `src/extraction/scoring.py` (one document) and `src/extraction/metrics.py` (the table); every rule has its own test in `tests/test_scoring.py` and `tests/test_metrics.py`.

1. **The unit is a leaf field** of the deal (`grades.amount_per_t`, `payment.trigger`); a quality indicator is one unit as a whole (rule 3). An empty nested object equals one whose fields are all `null`. `delivery.period_raw` is not scored.
2. **Comparison:** enums, booleans and dates must match exactly; numbers to the hundredth (9200 = 9200.004, but 0.1 ≠ 0.11); strings after normalisation (case, quotes, apostrophes, the legal form ТОВ, ТзОВ, ПрАТ, АТ, ДП, ПП, ФГ) with similarity ≥ 0.85 (`difflib`). The threshold was set on the OCR noise of the Prozorro golden set: a name corrected by the model scores 0.87–0.99, a name in another grammatical case 0.82, a name in Latin script 0. Known flaw: names that differ only in a digit also match.
3. **List items are paired first**, regardless of order: penalties by (`payer`, `trigger`, `unit`), grades by their class (`grade`, exact match), quality indicators by name and limit (`min`/`max`); an `other` indicator also needs a similar written name, compared without the explanation in closed brackets after it. Among several candidates the one that matches the golden one on more of the remaining values wins (`amount` and `cap_pct`; the price per tonne; the value and the unit). A quality indicator is one unit, `grades.indicators.<name>`, right when both its value and its unit match. A golden item without a pair has all its fields wrong; a missing grade, its price and every indicator. An extra item, one the golden deal does not have, is one wrong unit of the overall accuracy and is counted in its own column «Зайві елементи»: it is part of neither the main accuracy (golden fields only) nor fabricated (its values do not stand where the golden deal has `null`).
4. **Three numbers instead of one:** accuracy on filled fields is the main one — only fields that are not `null` in the golden deal; fabricated, % — the model gave a value where the golden deal has `null`; overall — all fields, for reference. Otherwise an "everything `null`" answer would score high. Accuracy is summed over the fields of all documents.
5. **A failed answer** (`failed` or `error`) is not dropped: all its fields are wrong, and the fields that are empty in the golden deal count as fabricated, so a failure never looks better than a mistake.
6. **Validity** is reported separately: valid, failed, and "fixed" — the share of documents that became valid only after the one fix attempt (it is part of valid, not of failed).
7. **`varied`** runs once per launch, like `precise`, so its spread shows only by comparing launches; 3 repeats on the golden set with mean and spread are planned for the final run.
8. **Separately per set:** synthetic texts (dev), Prozorro and the author's own contracts (test); the model is chosen on all golden documents together.
9. **Noise:** accuracy on filled fields comes with a 95% bootstrap interval — 1000 resamples of documents with replacement, fixed seed. When the intervals of two configurations overlap, no advantage is claimed.
10. **Cost** — tokens of all attempts × the price in `configs/models.yaml`, per 1000 answers without `error`; $0 for the local models.

## How documents are chunked (stage 2)

`uv run python -m scripts.chunk_rag` (local, free, a few seconds) turns `data/rag/documents.jsonl` into `data/rag/chunks.jsonl`, one chunk per line; a second run writes the same bytes. Chunks follow the structure of the document, not a fixed number of characters, because an answer has to cite a place a person can find: [own_003, п. 7.3], [ЦК, ст. 625]. Code: `src/rag/structure.py` (what an answer can cite) and `src/rag/chunking.py` (sizes and metadata); tests — `tests/test_rag_structure.py`, `tests/test_chunking.py`, `tests/test_chunk_rag.py`.

1. **A unit is what an answer cites.** A contract splits into its preamble, clauses «7.3» and requisites. A clause starts only where its number grows, so «1.1 цього Договору» wrapped inside clause 3.1 does not start a new one. A section line «5. Ціна» counts only when the next clause is 5.x, and everything after the last clause that looks like a party label or requisites goes to the section «Реквізити та додатки». A law splits into whole articles «ст. 625». A decision splits into paragraphs «п. 5.12»; its sections are capital-letter headings («ВСТАНОВИВ:») and «N. Title» lines. Section and heading lines are not part of the chunk text: they are kept in its metadata.
2. **Limits: at most 1500 characters, merge below 200.** They come from the corpus itself: contract clauses have a median of 300 characters, p90 1175, and 106 of 1777 are longer than 1500; law articles have a median of 584; decision paragraphs a median of 411, p90 1677. So 1500 keeps 94% of clauses and nearly every article whole. By the bge-m3 tokenizer the indexed texts (title, section, heading and text) have a median of 147 tokens, p90 316 and at most 544: far below the 8192 that bge-m3 and the reranker take, so nothing is cut.
3. **A long unit is split by its own numbering first:** a clause into subclauses «п. 10.6.1», an article into parts «ст. 30, ч. 1» or Tax Code points «ст. 201, п. 201.1». What is still too long is split between lines, then sentences, then words, and every piece keeps the ref of its unit.
4. **A short unit** (under 200 characters, such as «1.1. Товар — пшениця.») joins its neighbour when both are in the same section or article and fit into 1500 together; the ref becomes a range «п. 1.1–1.3» or a list «преамбула, п. 1.1». Units of 200 characters or more stay one per chunk, so their refs stay exact.
5. **Every chunk carries** `chunk_id` (`own_001#007`), `doc_id`, type, title, short title for the citation, source URL, a date or a version (`version_date`: the date of a contract or a decision; `retrieved_on`: for a law, the day its text was downloaded, as the kodeksy.com.ua mirror gives no edition date), `in_force`, `cites_commercial_code`, the hash of the document text (a changed document means its chunks are stale), section, heading, ref and pages.
6. **Comparison with fixed-size chunks.** The baseline is windows of 1500 characters with an overlap of 200 over the same document text, measured against the same citable units. *Clauses cut*: units with a ref that no single chunk holds whole. *Cross a boundary*: chunks touching two sections of a contract or two articles of a law. *One ref*: chunks that a single ref can cite.

| chunking (05.11.2026, 45 documents) | chunks | median chars | p90 chars | clauses cut | cross a boundary | one ref |
| --- | --- | --- | --- | --- | --- | --- |
| structural ≤ 1500, merge < 200 | 2836 | 468 | 1163 | 3.9% | 0.0% | 67.2% |
| fixed 1500, overlap 200 | 1278 | 1500 | 1500 | 20.7% | 27.6% | 8.1% |

Fixed windows cut every fifth clause and mix two sections or articles in every fourth chunk, so an answer would often see half a clause or cite the wrong one. Structural chunks cut only the 3.9% of units that stay over 1500 characters even after their subclauses, and never cross a boundary. The other 33% of structural chunks are merged ranges or decision paragraphs without a number, cited by the range or the section. The price is twice as many chunks. This table measures only where the boundaries fall; retrieval is measured in "How search works": structural chunks put the right text in the top 5 for 0.93 of the questions, fixed windows for 0.73 (vector search).

## How search works (stage 2)

`src/rag/search.py` searches the Qdrant index that `scripts/index_rag.py` builds, `scripts/eval_search.py` measures it on questions with a known answer. Code: `src/rag/bm25.py`, `embedding.py`, `index.py`, `search.py` and `search_eval.py`; tests — `tests/test_bm25.py`, `test_embedding.py`, `test_rag_index.py`, `test_index_rag.py`, `test_search.py`, `test_search_eval.py`, `test_eval_search.py`, all on an in-memory Qdrant and stub models.

1. **The index.** Every chunk is one Qdrant point with two vectors. `dense` is the bge-m3 embedding (1024 numbers, compared by cosine) of the chunk's title, section, heading and text. `bm25` holds the BM25 weights of the same words; Qdrant multiplies them by IDF itself, so a new document does not re-weight the others. The payload is the whole chunk, with indexes on `doc_id`, `doc_type`, `in_force` and `version_date` for the filters. There are two collections: `chunks` (2836 structural chunks) and `chunks_fixed` (1278 fixed-size windows, the baseline).
2. **A stale index is refused.** After the last point the collection gets a stamp: the hash of its chunks, the model, the vector size and the index version. The version (`INDEX_VERSION` in `src/rag/index.py`) is raised by hand whenever the way a chunk becomes vectors changes (`searchable_text`, the tokenizer, BM25's k1 and b), because the hash of the chunks does not see such a change. A `Searcher` checks the stamp once, when it is created, and stops with "rebuild it" when the index was built from other chunks, by another model or another index version, or did not finish.
3. **The embedder is bge-m3 through fastembed** (ONNX on CPU, no torch, no key). It knows Ukrainian, takes up to 8192 tokens, gives the same vectors as bge-m3 in Ollama (cosine 0.9999 on corpus chunks) and runs the same way in GitHub Actions, where Ollama would not.
4. **Three modes.** `vector`: cosine between the question and each chunk. `hybrid`: 50 chunks by vector and 50 by BM25, fused by RRF (reciprocal rank fusion): a chunk scores by its places in the two lists, not by their scores, which are on different scales. `hybrid_rerank`: the top 20 of hybrid are read again by bge-reranker-v2-m3 (int8 ONNX), a cross-encoder that reads the question and the chunk together; it judges a pair better but is too slow for the whole index.
5. **Filters** (`SearchFilter`). `contract_id`: this contract plus all laws and decisions, never another contract; with `doc_types: [contract]` only this contract. `doc_types`: only these types. `skip_repealed`: drops laws that are no longer in force (only laws carry `in_force`, so contracts and decisions stay). `version_from` / `version_to`: only decisions have a date, so undated chunks pass and dated ones must fall in the range.
6. **Questions.** 15 in `data/rag/search_questions.yaml`, written and checked by the author: 7 about one contract (own_003, own_008, prozorro_009), 6 about the law, 2 about Supreme Court decisions; 4 `exact` (rare words or numbers shared with the answer), 5 `paraphrase` (the same said in other words), 6 `plain`. A found chunk is right when it comes from the question's document and holds its `evidence` word for word (spaces, line breaks and apostrophes compared loosely), so one rule scores structural chunks and fixed windows. A test checks every evidence against `chunks.jsonl`.
7. **Metrics.** R@k: the share of questions with a right chunk among the first k. MRR: the mean of 1/rank of the first right chunk, a miss (not in the first 10) counting 0; it rewards a right chunk at 1 more than one at 5.

| search (06.11.2026, 15 questions, structural chunks) | R@1 | R@3 | R@5 | R@10 | MRR | mean time, s |
| --- | --- | --- | --- | --- | --- | --- |
| vector | 0.67 | 0.93 | 0.93 | 0.93 | 0.79 | 0.03 |
| hybrid | 0.53 | 0.87 | 0.87 | 0.93 | 0.69 | 0.02 |
| hybrid + rerank | 0.73 | 0.93 | 0.93 | 0.93 | 0.81 | 1.99 |

| R@5 / MRR | vector | hybrid | hybrid + rerank |
| --- | --- | --- | --- |
| structural chunks | 0.93 / 0.79 | 0.87 / 0.69 | 0.93 / 0.81 |
| fixed windows | 0.73 / 0.72 | 0.80 / 0.70 | 0.87 / 0.79 |
| 7 contract questions: this contract only | 1.00 / 1.00 | 1.00 / 0.93 | 1.00 / 1.00 |
| the same, with laws and decisions | 1.00 / 0.64 | 1.00 / 0.61 | 1.00 / 0.72 |

The full report, with recall by question kind and the rank of every question — `results/stage2/search/2026-11-06T10-29-42/search_metrics.md`; every search with the chunks that came instead — `ranks.jsonl` next to it.

What the numbers say. Reranking gives the best R@1 and MRR, but plain vector search is one question behind at R@1 and 66 times faster. With 15 questions one question is 0.07, so a difference of one or two questions is not claimed as an advantage. Hybrid alone is below vector here: vector already finds every `exact` question first, so BM25 has nothing to add, and on `plain` questions it pulls up chunks that repeat the question's words ("Що таке пеня?" drops the definition from 3rd to 6th). Structural chunks beat fixed windows at R@5 in every mode (by MRR, hybrid is even: 0.69 and 0.70). A contract question keeps its clause in the top 5 when laws and decisions are allowed, but loses the first place to decisions that quote similar clauses of other contracts, so a question about a contract searches only that contract.

**Decisions made in Е2.4.** The embedder is bge-m3 through fastembed (point 3), so the same model runs locally and in CI. Qdrant runs locally in Docker (`docker-compose.yml`); the tests and CI use the in-memory Qdrant, so CI needs no Qdrant service. Every evaluation run goes to a new folder named by its start time, as in stage 1.

**Known limits and what is left to do:**

- **Filters.** Contract questions search only the contract; searching a contract together with the law, and the other filters, are still to be reworked (one option for stage 3: separate agent tools "search the contract" and "search the law").
- **Whose position a decision paragraph holds.** q15 is missed by every mode, yet every mode finds the right decision: it returns the paragraphs that state the appeal court's view, which the Supreme Court overturned. All paragraphs of a decision sit in one section «ВСТАНОВИВ:», so a chunk does not know whether it holds the lower court's reasoning or the Supreme Court's conclusion. A bot would cite the overturned view; decisions need splitting by their subsections or a mark of whose position a paragraph is.
- **BM25 has no stemming:** «пеня», «пеню» and «пені» are three different words for it.
- **15 questions** tell modes apart only by large differences; there are no confidence intervals yet.

## How answers work (stage 2)

`scripts/ask_rag.py` asks the 26 questions of `data/rag/answer_questions.yaml` and writes each answer with its sources to `results/stage2/answers/<start time>/`. Code: `src/rag/context.py`, `answerer.py`, `answer.py`, `answer_config.py` and `answer_eval.py`; settings — `configs/answer.yaml`; tests — `tests/test_rag_context.py`, `test_rag_answerer.py`, `test_rag_answer.py`, `test_answer_config.py`, `test_answer_eval.py`, `test_ask_rag.py`.

1. **Three searches.** The question's contract (4 chunks), the laws (3) and the decisions (3) are searched apart with `hybrid_rerank` and then interleaved: in one shared top the long decisions pushed the laws out. No other contract is ever searched.
2. **A threshold instead of a guess.** A chunk with a reranker score below −1.7 is noise. With nothing left the answer is «У документах цього немає» and the model is not asked, so it costs nothing.
3. **The context.** Whole chunks, up to about 4000 tokens. Each chunk is a block `<document id="S1">` with its title, type, source and warning. Anything in a chunk or in the question that could close a block or open a fake one is escaped: the text of a contract is data, not instructions.
4. **The answer.** Claude Haiku 5.5 fills the forced tool `record_answer`: claims, each with the labels of its chunks, or `not_in_documents`. An invalid answer is sent back once with the reason; if the fix is invalid too, the result is `failed`.
5. **Python writes the citations and the warnings.** It checks that every label was in the context and builds the citation from the chunk's metadata, so a citation cannot name a clause the chunk does not have; the text of a claim is still the model's. The Commercial Code has not been in force since 28.08.2025: an article of it gets «ГК втратив чинність…», a decision resting on it «Це практика за ГК, що втратив чинність 28.08.2025.». A wrong answer from the model cannot drop them.
6. **Questions.** The 15 search questions and 11 traps: 5 off the corpus (t01, t02, t06–t08), 2 on the repealed code (t03, t04), 1 injection (t05) and 3 on a contract with the same clause but other numbers (s01–s03). The runs in the table below asked the first 20 (q01–q15, t01–t05); t06–t08 and s01–s03 came with the Е2.6 test set. Two checks need no judge: the outcome is the expected one, and the answer cites a chunk that holds the evidence.

| prompt (run) | outcome as expected | cites the evidence | tokens in / out | $ per 1000 questions | latency p50 / p95, s |
| --- | --- | --- | --- | --- | --- |
| v1 (`2026-11-07T20-17-50`) | 19 / 20 | 16 / 18 | 50 436 / 7 248 | 0.43 | 7.1 / 9.3 |
| v2 (`2026-11-08T17-24-00`) | 19 / 20 | 16 / 18 | 62 672 / 5 128 | 0.44 | 6.7 / 9.4 |
| v3 (`2026-11-08T17-58-33`) | 19 / 20 | 16 / 18 | 65 598 / 5 390 | 0.46 | 6.5 / 9.4 |
| v3, final (`2026-11-09T08-46-28`) | 19 / 20 | 16 / 18 | 65 156 / 4 797 | 0.45 | 5.9 / 9.8 |

What the numbers say. The automatic metrics are the same for all three prompts, so they cannot rank them. The one wrong outcome is q03: its right clause scores −5.11, below the threshold. The two missed evidences are q03 and q15. Answers also differ from run to run, so v2 and v3 were compared over 4 tries each with every answer read by hand, and v3 was chosen (`prompts/CHANGELOG.md`, «answer v3»). The traps held in every version.

**Known limits and what is left to do:**

- **Arguments of a party given as the court's view.** q05 and t03 repeat the same error in every try: the model does not tell the section with the complainant's arguments from the court's conclusion. The prompt did not fix it; the code knows the section, so it has to mark or drop such chunks. The judge finds it too (q05, claim 4).
- **Terms of someone else's contract.** A decision may quote another party's contract, and the answer may repeat its terms (q01 in 3 of 4 tries of v3, t03 in all).
- **Foreign words** («half») appear with the «Ukrainian only» rule as often as without it, and Haiku 5.5 accepts only the default sampling.
- **The threshold** was set on 23 questions and loses q03; it is to be recalibrated on the Е2.6 test set.

## How answers are checked (stage 2)

Two scripts check a saved run of the answers; neither searches or needs Qdrant, so a run can be checked again at any time. The questions and their references — `data/rag/answer_questions.yaml`; thresholds — `configs/eval.yaml` (rules) and `configs/judge.yaml` (judge). Code: `src/rag/answer_scoring.py`, `src/rag/judge.py`, `scripts/score_answers.py`, `scripts/judge_answers.py` (the calibration of the judge has its own subsection below); tests — `tests/test_answer_scoring.py`, `test_score_answers.py`, `test_judge.py`, `test_judge_answers.py`.

1. **The test set.** Each of the 26 questions has a reference answer written by a person (`expected_answer`) and the key facts a right answer holds (`key_facts`: a number, a word stem, alternatives split by `|`). The similar-contract traps also have forbidden facts: the numbers of the other contract. Every key fact is in an evidence chunk, and a test keeps them in step.
2. **Rules, free.** `score_answers` looks for the facts in the claims only (citations and warnings carry numbers the model did not say) and writes `scores.md`. Five thresholds: the right source ≥ 85%, key facts ≥ 80%, «У документах цього немає» for every off-corpus question, no forbidden fact, no claim that calls the Commercial Code in force.
3. **The judge, Claude Sonnet 5.5.** A model other than the one that answers, so it does not grade its own habits. Two checks, one call each:
   - **correct** — the answer against the reference: the conclusion, the citation and the numbers, three yes/no; the answer is right when all three are yes. A question left unanswered is wrong and costs nothing. The 5 off-corpus questions are left to the rules.
   - **faithful** — each claim against the documents it cites, with no reference in sight. The answer keeps those documents exactly as the answering model read them (`cited_documents` in `answers.jsonl`), so the verdict does not depend on today's `chunks.jsonl`. A run saved before that is refused before any paid call.

   The verdict is JSON shaped by a schema (structured outputs): Sonnet 5.5 rejects the forced tool call the answers use. A verdict Python cannot accept goes back once with the reason, and then it is `failed`. A judgement with no verdict counts as a miss, so a failure of the judge cannot raise the score. In doubt the judge says no: the farmer acts on the answer.
4. **Results.** In the run's folder: `judge.jsonl` (every verdict with its reasons, the model, the prompt version and hash, tokens, latency) and `judge.md` (the table by question and the claims without support). `--limit N` judges only the first N questions: a check that the judge works, with no thresholds.

The full run `2026-11-10T10-08-19` (answer prompt v3), judged with `judge_correct_v1` and `judge_faithful_v2` (the judging with `judge_faithful_v1` is in commit `644fa89`):

| check | measured | threshold | passed |
| --- | --- | --- | --- |
| rules: cites the right source | 19 / 21 (90%) | ≥ 85% | ✓ |
| rules: key facts | 36 / 40 (90%) | ≥ 80% | ✓ |
| rules: «немає» for off-corpus questions | 5 / 5 | all | ✓ |
| rules: forbidden facts; Commercial Code as in force | 0; 0 | 0 | ✓ |
| judge: correct | 19 / 21 (90%) | ≥ 80% | ✓ |
| judge: faithful claims | 40 / 47 (85%) | ≥ 90% | ✗ |

The judge cost $0.20 for the run (64 602 tokens in, 6 682 out), about $7.5 per 1000 questions; the answers themselves cost $0.01.

What the verdicts say, read by hand against the documents. The two misses on correct are q03 (unanswered: its clause is below the threshold) and s03 (п. 6.1 counts 3 days from the transfer of ownership and the receipt of the documents; the answer drops the documents). Each of the 7 claims without support has a real ground: q01 and t05 widen п. 7.2 from the terms of п. 6.2 and 6.4 to any term of the contract, q05 gives the complainant's argument as the Supreme Court's view, t03 says the court heard the case in 2014 (the contract's period; the decision is of 2026), q13 and s03 drop a condition (the acceptance of the documents of title; the receipt of the documents), q11 adds «лише» to Article 617 of the Civil Code — strict but defensible.

`judge_faithful_v1` also wanted the Commercial Code warning inside q13's claim: it did not know that Python shows that warning below the answer (step 5 of "How answers work"). `judge_faithful_v2` tells it so; the warning in the judge's documents and the one below the answer come from one function, and a test keeps them equal. That reason is gone; q13 stays unsupported for the dropped condition alone. v1 and v2 differ in one verdict of 47: s03 claim 1, supported under v1. Nothing of v2 touches it (no warning there), so it is the spread of the model between two runs. So faithful misses 90% because of the answers, not the judge, and the threshold stays: the fix is on the answering side.

### Calibration of the judge on manual marks

The judge is used only if it agrees with a person. `scripts/judge_sheet.py` writes `data/rag/judge_marks.yaml`: the 20 answers and 47 claims the judge judged in a run, each with the question, the reference, the answer and the documents the claim cites, but without the judge's verdicts. The author fills `correct` and every `supported` by the rules of the judge's prompts (they are on top of the sheet). `scripts/calibrate_judge.py` puts each mark next to the judge's verdict and writes `calibration.md` in the run's folder. Both checks need agreement ≥ 80% (`agreement_min` in `configs/judge.yaml`), or the judge is not used; a verdict the judge did not give counts as a disagreement. Code: `src/rag/calibration.py`; tests — `tests/test_calibration.py`, `test_judge_sheet.py`, `test_calibrate_judge.py`.

The author had read the judge's verdicts on 7 questions (q01, q05, q11, q13, t03, t05, s03) before marking. They stay in the sheet as `seen`: all 6 claims the judge rejected and its only correct ✗ are among them, and without them the agreement would test nothing. So the agreement is counted over all answers and over unseen and seen apart. t05 (the injection) was re-marked before the comparison: its `false` marked the question, not the answer, which did not follow the injection.

Results on run `2026-11-10T10-08-19`:

| check | all | unseen | seen | threshold | passed |
| --- | --- | --- | --- | --- | --- |
| correct | 19 / 20 (95%) | 13 / 13 (100%) | 6 / 7 (86%) | ≥ 80% | ✓ |
| faithful | 39 / 47 (83%) | 27 / 28 (96%) | 12 / 19 (63%) | ≥ 80% | ✓ |

The judge passes and is used (with `judge_faithful_v1` faithful agreed on 40 of 47, 85%). On the blind marks it agrees almost always; on the seen ones the author disagreed with it more often, so the marks did not follow its verdicts. Of the 9 disagreements, 2 are «judge yes, author no», a missed error (q01 claim 2, q04 claim 1), and 7 «judge no, author yes», a judge too strict (q01 claim 3, q05 claim 4, q11 claim 1, q13 claim 2, t03 claim 3, s03 claim 1, s03 correct). Read again against the documents, at least four are the author's errors, not the judge's: q01 claim 2 and q04 claim 1 say what п. 7.2 and п. 6.3 say; q01 claim 3 widens п. 7.2 just as t05 claim 2, which the author marked false; t03 claim 3 dates the hearing by the contract's period. The marks are not changed after reading the judge's reasons: then they would no longer be blind.

**Left to do:** the errors of the answers the judge finds, on the answering side: a party's arguments given as the court's view (q05, t03), a condition widened or dropped (q01, t05, q13, s03).

## CI

GitHub Actions (`.github/workflows/ci.yml`) checks every push and pull request, free and with no key: `uv sync --locked` (exactly the versions of `uv.lock`), `pytest`, `mypy`, the rules on the newest run of the answers (`score_answers`, thresholds of `configs/eval.yaml`) and the calibration of the judge (`calibrate_judge`, agreement ≥ 80%). A threshold that is not met fails the run. The tests need neither Qdrant nor the models: Qdrant runs in memory, the embedder and the reranker are replaced by fakes.

The paid judge runs only by hand: Actions → CI → Run workflow (the `limit` field judges only the first N questions). It needs the `ANTHROPIC_API_KEY` secret in Settings → Secrets and variables → Actions; its `judge.md` and `judge.jsonl` are kept as the run's artifact. For now this job fails: faithful is 85% against 90%, for the errors of the answers above.

The actions are pinned to a commit hash (a tag can be moved), the checkout keeps no GitHub token, and the workflow may only read the repository.

## Where things are

### Stage 1 requirements

| Requirement | Where to look | Status |
| --- | --- | --- |
| 30 "dirty" texts without personal data | `data/texts/` — 48 texts: `synthetic_001–020.txt` (dev), `prozorro_001–010.txt` and the author's own contracts `own_001–018.txt` (test; `own_008–012` written by the author on a general template with fictional parties); how they were anonymised — `data/README.md` | done |
| 10 hand-labelled texts (golden set) | `data/golden/`: `prozorro_001–010.json` and `own_001–018.json`, labelling rules — `data/golden/RULES.md` | done: all 28 checked and approved by the author |
| Schema: ≥ 6 fields, ≥ 1 enum, a nested object or a list | `src/schemas/deal.py` (11 fields, 12 enums, 5 nested objects, the lists `penalties` and `grades`, each grade with its own list of quality indicators), tests — `tests/test_deal.py` | done |
| Provider model through an API with structured output, retry and timeout | `src/extraction/anthropic_extractor.py` (Claude Haiku 4.5 and 5.5: the schema as a forced tool call), `src/extraction/openrouter_extractor.py` (GPT-4o: strict structured output), settings — `configs/models.yaml` | done |
| Local model through a shared interface | stage 1: `src/extraction/ollama_extractor.py` in tag `stage-1` (Qwen3 14B; Lapa 12B in the first matrix), its runs in `results/stage1_2/` and `results/stage1/`; the `Extractor` interface — `src/extraction/run.py` | done in stage 1; since stage 1.3 the code has no local model — the course no longer requires one |
| temperature and top_p in a config, ≥ 2 configurations | `configs/sampling.yaml` (`precise`, `varied`; `default` sends none) | done: both samplings run in every `make run` for the models that accept them; Haiku 5.5 rejects them and runs `default` only |
| Metrics: per-field accuracy, % valid, p50/p95 latency, cost per 1000 documents | `src/extraction/metrics.py` (table), `src/extraction/scoring.py` (comparison with the golden deal); tests — `tests/test_metrics.py`, `tests/test_scoring.py` | done; numbers — `results/stage1_3/<start time>/metrics.md` and `metrics_by_field.md` |
| Prompts with a version header, ≥ 2 versions, `CHANGELOG` | `prompts/extract_v1.md`, `prompts/extract_v2.md`, `prompts/CHANGELOG.md` | done: v1 is the baseline, v2 is built from the v1 errors on dev |
| Every answer is validated by the schema and by content checks (a late-payment penalty paid by the seller, more than 50 000 t — what a 5 000 ha farm can sell; a placeholder buyer becomes `null` without a fix); invalid → one fix attempt → `failed` (an answer cut off at `max_tokens` or a refusal gets none: asking again would bring the same) | `src/extraction/validation.py` (the fix request), `checks.py` (content checks), `anthropic_extractor.py` and `openrouter_extractor.py` (the second call); tests — `tests/test_validation.py`, `tests/test_checks.py` and the extractor tests | done; the "Виправлено, %" column of the metrics table counts the answers saved by it |
| Every result logs the model, prompt version and sampling | record fields — `src/extraction/result.py`; the records — `results/stage1_3/*/*.jsonl` (and `results/stage1_2/`, `results/stage1/`) | done |
| `REPORT.md`: table, error analysis, production choice | `REPORT.md` (in Ukrainian) | done: Claude Haiku 5.5 + prompt v2 (stage 1.3; stage 1.2 chose Claude Haiku 4.5 + v2 + `precise`) |
| One command to run: `make run` (= `python main.py`) | `Makefile`, `main.py` | done |

### Folders and files

```
Makefile                   make install / test / run / metrics
.github/workflows/ci.yml   CI: tests, mypy, rules and judge calibration on every push; the paid judge by hand
docker-compose.yml         Qdrant for the stage 2 search (docker compose up -d qdrant)
main.py                    stage 1 in one command: the whole matrix, then the metrics table
configs/
  models.yaml              models with their provider (anthropic, openrouter), timeout, retry, prices per 1M tokens; a new model is a new section
  sampling.yaml            sampling configurations precise, varied and default
  answer.yaml              stage 2 answers: the model, the prompt and how the context is built
  eval.yaml                stage 2 thresholds of the rule checks
  judge.yaml               the stage 2 LLM judge: the model, its prompts, prices and thresholds
data/
  texts/                   the 48 texts the models get
  synthetic/               golden Deal and "dirtiness" parameters of each synthetic text
  golden/                  golden deals of the Prozorro and own contracts + RULES.md
  prozorro_sources.csv     which Prozorro tender each contract comes from
  own_sources.csv          which PDF and contract number each own contract comes from
  rag_sources.csv          stage 2 corpus: 28 contracts (PDF), 6 laws and 11 Supreme Court decisions (HTML)
  rag/                     the stage 2 corpus after ingestion: documents.jsonl (text by page or article), manifest.jsonl, chunks.jsonl (chunks with metadata) and search_questions.yaml (15 questions with a known answer)
  rag/answer_questions.yaml  the 26 questions the answers are tried on: 15 by id from search_questions.yaml and 11 traps, each with a reference answer and key facts
  rag/judge_marks.yaml     the author's manual marks of the judged answers of one run, for the calibration of the judge
  README.md                how the data was collected, anonymised and labelled
  raw/                     original PDFs, masking check files and the stage 2 HTML pages in raw/rag/ (not in git)
prompts/
  extract_v1.md            baseline prompt, version in the header
  extract_v2.md            prompt with per-field rules; {today} becomes the run date
  answer_v1.md … answer_v3.md  stage 2 answer prompts; v3 is in configs/answer.yaml
  judge_correct_v1.md, judge_faithful_v1.md, judge_faithful_v2.md  the prompts of the stage 2 judge; faithful v2 is in configs/judge.yaml
  CHANGELOG.md             what changed between versions and why
src/schemas/
  deal.py                  the Deal schema: grades with their prices and quality indicators
src/extraction/
  anthropic_extractor.py   Claude Haiku 4.5 and 5.5 through the Anthropic API
  openrouter_extractor.py  GPT-4o through OpenRouter (OpenAI SDK)
  run.py                   the shared Extractor interface, one configuration, one saved run
  retry.py                 which API errors are worth another try, shared by both extractors
  messages.py              the same user message with the document for both models
  prompts.py               reads a prompt with its version and file hash
  sampling.py              reads configs/sampling.yaml
  validation.py            checks an answer against the Deal schema; the fix request text
  checks.py                content checks after the schema: penalty payer, quantity, placeholder buyer
  text.py                  text normalization shared by the content checks and the scoring
  result.py                what is recorded about each answer
  scoring.py               compares one answer with the golden deal
  metrics.py               metrics table model × prompt × sampling
src/rag/
  sources.py               reads and checks data/rag_sources.csv: types, articles of each law, course minimums
  documents.py             an ingested document (parts with a page or an article), its manifest entry, JSONL
  structure.py             citable units of a document: contract clauses, law articles and their parts, decision paragraphs
  chunking.py              units packed into chunks of 200–1500 characters with metadata; fixed-size windows for comparison
  bm25.py                  BM25 word weights for Ukrainian text, kept in Qdrant as a sparse vector
  embedding.py             bge-m3 through fastembed on CPU; texts embedded in batches of similar length
  index.py                 a chunk becomes a Qdrant point with two vectors; the stamp that marks a stale index
  search.py                vector, hybrid and hybrid + rerank search with filters
  search_eval.py           which found chunk is right; recall@k and MRR
  context.py               three searches, the threshold and the document blocks the model reads
  answer.py                the answer schema; Python's citations and warnings
  answerer.py              search, then Claude with a forced tool, the one fix, the result of a question
  answer_config.py         reads configs/answer.yaml
  answer_eval.py           the 26 questions and the checks that need no judge
  answer_scoring.py        rule scoring of a saved run with the thresholds of configs/eval.yaml
  judge.py                 the LLM judge: correct and faithful checks of one saved answer
  calibration.py           the sheet of manual marks and the agreement of the judge with it
scripts/
  generate_synthetic.py    deals for the synthetic set (+ synthetic_render.py, synthetic_checks.py)
  fetch_prozorro.py        search for contracts in Prozorro
  fetch_rag_sources.py     downloads the laws (kodeksy.com.ua) and decisions (reyestr.court.gov.ua) of the stage 2 corpus
  pdf_to_text.py           PDF → anonymised text
  ingest_rag.py            stage 2 ingestion: PDF and HTML → clean, anonymised documents in data/rag/
  chunk_rag.py             stage 2 chunking: data/rag/chunks.jsonl and the comparison with fixed-size chunks
  index_rag.py             builds the two Qdrant collections: structural chunks and fixed-size windows
  eval_search.py           measures the search on the 15 questions: report and ranks in results/stage2/search/
  ask_rag.py               asks the 26 questions: report and answers in results/stage2/answers/
  score_answers.py         rule scoring of a saved run: scores.md, free
  judge_answers.py         the judge on a saved run: judge.jsonl and judge.md, paid
  judge_sheet.py           the sheet for the author's manual marks of a judged run, free
  calibrate_judge.py       the manual marks against judge.jsonl: calibration.md, free
  check_golden.py          validates the golden labels
tests/                     pytest tests: make test
results/stage1_3/          one folder per launch, named with its start time: a *.jsonl file per configuration (one line per document) and the metrics tables (metrics.md, metrics_by_field.md)
results/stage1_2/          the same for stage 1.2, on the schema before grades (read with the stage-1 tag)
results/stage1/            the same for the first matrix (Claude and Lapa)
results/stage2/search/     one folder per search evaluation: search_metrics.md and ranks.jsonl
results/stage2/answers/    one folder per answers run: report.md and answers.jsonl, then scores.md, judge.jsonl, judge.md and calibration.md
```

The author keeps the work plan locally (`docs/`, outside the repository); its metric rules were moved to "How the metrics are counted".
