# Stage 1 commands; see README.md. `make run FRESH=1` asks the models again.
ENV_FILE := $(if $(wildcard .env),--env-file .env,)

.PHONY: install test run metrics

install:
	uv sync

test:
	uv run pytest -q

run:
	uv run $(ENV_FILE) python main.py $(if $(FRESH),--fresh,)

metrics:
	uv run python -m src.extraction.metrics
