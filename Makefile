# Stage 1 commands; see README.md. `make run` asks the models and saves a new run.
ENV_FILE := $(if $(wildcard .env),--env-file .env,)

.PHONY: install test run metrics

install:
	uv sync

test:
	uv run pytest -q

run:
	uv run $(ENV_FILE) python main.py

metrics:
	uv run python -m src.extraction.metrics
