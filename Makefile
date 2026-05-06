# Makefile for mentat-session-logger
# ─────────────────────────────────────────────────────────────────────────────
# Usage: make <target>
#   PYTHON=python3.11 make setup   # override the Python interpreter
# ─────────────────────────────────────────────────────────────────────────────

PYTHON   ?= python
PIP      ?= pip

.PHONY: help setup test test-integration lint format typecheck check demo clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Dependencies ──────────────────────────────────────────────────────────────

setup: ## Install core + dev dependencies (pytest, ruff, mypy)
	$(PIP) install -e .[dev]

setup-asr: ## Install optional ASR/diarization backends (torch, whisperx, pyannote)
	$(PIP) install -e .[asr]

setup-all: ## Install everything (dev + asr)
	$(PIP) install -e .[all]

# ── Quality ───────────────────────────────────────────────────────────────────

test: ## Run all unit tests (no Ollama / WhisperX / ffmpeg required)
	$(PYTHON) -m pytest tests/unit/ -q

test-integration: ## Run integration tests (requires a running Ollama instance)
	$(PYTHON) -m pytest tests/integration/ -m integration -v

lint: ## Check code style with ruff
	$(PYTHON) -m ruff check .

format: ## Auto-format code with ruff
	$(PYTHON) -m ruff format .

typecheck: ## Run mypy static type checking
	$(PYTHON) -m mypy src

check: lint typecheck test ## Run lint + typecheck + unit tests (CI-equivalent)

# ── Local runs ────────────────────────────────────────────────────────────────

demo: ## Zero-dependency E2E demo (stub audio + no LLM, outputs in envs/demo/)
	$(PYTHON) scripts/demo_pipeline.py

demo-llm: ## E2E demo with stub LLM responses (full pipeline, no Ollama needed)
	$(PYTHON) scripts/demo_pipeline.py --with-llm

# ── Maintenance ───────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and cache directories
	rm -rf .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
