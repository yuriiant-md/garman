# Commands of stage 1 (install … metrics) and stage 2 (qdrant … ask); see README.md.
ENV_FILE := $(if $(wildcard .env),--env-file .env,)
# Print each line at once even when the output goes through a pipe (`make run | tee run.log`).
export PYTHONUNBUFFERED := 1
# A Mac that falls asleep pauses a long run; caffeinate exists only on macOS.
KEEP_AWAKE := $(if $(shell command -v caffeinate),caffeinate -i,)
QDRANT_READY_URL := http://127.0.0.1:6333/readyz
QDRANT_WAIT_TRIES := 30

.PHONY: install test run smoke metrics qdrant rag answers ask

install:
	uv sync

test:
	uv run pytest -q

run:
	uv run $(ENV_FILE) python main.py

# A quick check on 2 texts of each source, saved in results/smoke/ (ignored by git).
smoke:
	uv run $(ENV_FILE) python main.py --per-source 2

metrics:
	uv run python -m src.extraction.metrics

# Stage 2 needs Docker: Qdrant keeps the search index.
qdrant:
	docker compose up -d qdrant
	@for i in $$(seq $(QDRANT_WAIT_TRIES)); do \
		curl -sf $(QDRANT_READY_URL) >/dev/null && exit 0; sleep 1; \
	done; echo "Qdrant is not ready at $(QDRANT_READY_URL)" >&2; exit 1

# Stage 2 without a key: the index (rebuilt only when stale), search quality, and the checks
# of the saved answers that CI runs too: rules with thresholds and the judge's calibration.
rag: qdrant
	$(KEEP_AWAKE) uv run python -m scripts.index_rag
	$(KEEP_AWAKE) uv run python -m scripts.eval_search
	uv run python -m scripts.score_answers
	uv run python -m scripts.calibrate_judge

# Stage 2, paid (about $0.01): new answers to the 26 questions, then their check by rules.
answers: qdrant
	$(KEEP_AWAKE) uv run python -m scripts.index_rag
	$(KEEP_AWAKE) uv run $(ENV_FILE) python -m scripts.ask_rag
	uv run python -m scripts.score_answers

# Stage 2, paid (cents): answers to a few test questions, e.g. make ask IDS=q01,t05
ask: qdrant
	@test -n "$(IDS)" || { echo "name the questions: make ask IDS=q01,t05" >&2; exit 2; }
	$(KEEP_AWAKE) uv run python -m scripts.index_rag
	$(KEEP_AWAKE) uv run $(ENV_FILE) python -m scripts.ask_rag --ids $(IDS)
