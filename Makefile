# Stage 1 commands; see README.md. `make run` asks the models and saves a new run.
ENV_FILE := $(if $(wildcard .env),--env-file .env,)

.PHONY: install test run smoke metrics

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
