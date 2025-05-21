# QNNAS Makefile

# Paths
VENV := .venv
PY := $(VENV)/bin/python
UV := uv

# ========================
# ENVIRONMENT SETUP
# ========================

.PHONY: install
install:
	@echo "📦 Installing project in $(VENV)"
	$(UV) venv $(VENV)
	$(UV) pip install .

# ========================
# APP LAUNCH
# ========================

.PHONY: ui
ui:
	@echo "🚀 Launching Gradio UI..."
	@PY=$(PY) $(PY) -u -m ui.app

.PHONY: train
train:
	@echo "🧠 Starting CLI training..."
	$(PY) -m training.train

# ========================
# DEV TOOLS
# ========================

.PHONY: format
format:
	@echo "🎨 Formatting code with black..."
	$(VENV)/bin/black .

.PHONY: lint
lint:
	@echo "🧹 Linting code with ruff..."
	$(VENV)/bin/ruff check .

.PHONY: test
test:
	@echo "🧪 Running tests with pytest..."
	$(VENV)/bin/pytest tests/

# ========================
# CLEANUP
# ========================

.PHONY: clean
clean:
	@echo "🧼 Cleaning up cache and bytecode..."
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache

.PHONY: freeze
freeze:
	@echo "📋 Freezing current environment to requirements.txt"
	$(UV) pip freeze > requirements.txt
