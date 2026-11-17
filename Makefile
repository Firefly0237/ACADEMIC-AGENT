# Thin aliases for the cross-platform operations facade. No implicit resource creation.
.DEFAULT_GOAL := help
.PHONY: help doctor stacks status up down logs clean install lock lint format type \
        test test-integration check web-build
PYTHON ?= python
STACK ?=

help:
	$(PYTHON) tools/cq.py --help

doctor:
	$(PYTHON) tools/cq.py doctor --json

stacks:
	$(PYTHON) tools/cq.py stacks

status:
	$(PYTHON) tools/cq.py stack $(STACK) status

up:  ## Existing registered containers only; set STACK explicitly
	$(PYTHON) tools/cq.py stack $(STACK) start

down:  ## Stop selected services, preserve containers and volumes
	$(PYTHON) tools/cq.py stack $(STACK) stop

logs:
	$(PYTHON) tools/cq.py stack $(STACK) logs

clean:  ## Only allowlisted disposable tool caches
	$(PYTHON) tools/cq.py clean-cache --apply

install:  ## Development tools; deploy/core builds use role-specific locks
	uv sync --locked

lock:
	uv lock

lint:
	uv run --frozen ruff format --check src tests scripts tools
	uv run --frozen ruff check src tests scripts tools

format:
	uv run --frozen ruff format src tests scripts tools
	uv run --frozen ruff check --fix src tests scripts tools

type:
	uv run --frozen mypy

test:  ## Use a configured current-role interpreter; no services/models are started here
	$(PYTHON) -m pytest tests/v3 tests/v2/unit -m "not integration and not e2e"

test-integration:  ## Explicitly opt into configured PostgreSQL/CAS integration tests
	$(PYTHON) -m pytest tests/v2/integration -m integration

check:  ## Static source and delivery structure only
	$(PYTHON) -m compileall -q src scripts tools tests
	$(PYTHON) tools/cq.py verify --json

web-build:
	npm --prefix web run build

