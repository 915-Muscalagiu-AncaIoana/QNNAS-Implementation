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
	@echo "📦 Creating virtual environment in $(VENV)"
	$(UV) venv $(VENV)
	@echo "📦 Installing project with uv pip..."
	$(UV) pip install . --python $(PY)

# ========================
# DATABASE SETUP
# ========================

.PHONY: start-postgres
start-postgres:
	docker-compose -f docker-compose-postgres.yml up -d

.PHONY: stop-postgres
stop-postgres:
	docker-compose -f docker-compose-postgres.yml down

.PHONY: remove-postgres
remove-postgres:
	docker-compose -f docker-compose-postgres.yml down -v

.PHONY: migrate
db-migrate:
	alembic revision --autogenerate -m "$(m)"

.PHONY: upgrade
db-upgrade:
	alembic upgrade head

# ========================
# APP LAUNCH
# ========================

.PHONY: start-backend
start-backend:
	$(PY) -m litestar --app-dir src --app api.backend:app run --reload

.PHONY: start-ui
start-ui:
	@echo "🚀 Launching Gradio UI..."
	PYTHONPATH=src $(PY) -u -m ui.app

.PHONY: train
train:
	@echo "🧠 Starting CLI training..."
	PYTHONPATH=src $(PY) src/training/train.py \
		--dataset=$(DATASET) \
		--gates=$(GATES) \
		--discount=$(DISCOUNT) \
		--lr=$(LR) \
		--max_length=$(MAX_LENGTH) \
		--encoder_path=$(ENCODER_PATH)

# ========================
# DEV TOOLS
# ========================

.PHONY: lint
lint:
	@echo "🧹 Linting code with ruff..."
	$(VENV)/bin/ruff check .

.PHONY: test
test:
	@echo "🧪 Running tests with coverage..."
	PYTHONPATH=src $(VENV)/bin/coverage run --rcfile=.coveragerc -m pytest tests/ --disable-warnings
	$(VENV)/bin/coverage report -m --rcfile=.coveragerc


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
