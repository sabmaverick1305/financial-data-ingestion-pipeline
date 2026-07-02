.DEFAULT_GOAL := help
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help venv install install-dev lint type-check test test-cov clean \
        deploy-infra destroy-infra invoke-lambda infra-status

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## Create virtual environment
	python3 -m venv $(VENV)

install: venv ## Install runtime dependencies
	$(PIP) install -e .

install-dev: venv ## Install dev dependencies
	$(PIP) install -e ".[dev]"

lint: ## Run ruff linter and formatter check
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests

format: ## Auto-format code with ruff
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

type-check: ## Run mypy type checks
	$(VENV)/bin/mypy src

test: ## Run tests
	$(VENV)/bin/pytest

test-cov: ## Run tests with HTML coverage report
	$(VENV)/bin/pytest --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

run: ## Run the pipeline CLI (pass ARGS="..." to forward arguments)
	$(PYTHON) -m financial_pipeline.cli $(ARGS)

clean: ## Remove build artifacts and caches
	rm -rf dist build *.egg-info .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

# ── Infrastructure (AWS CloudFormation) ────────────────────────────────────
deploy-infra: ## Deploy the AMFI pipeline stack (default bucket: mf-finance-kb)
	./infra/deploy.sh deploy $(if $(S3_BUCKET),--bucket "$(S3_BUCKET)") $(if $(STACK),--stack "$(STACK)") $(if $(REGION),--region "$(REGION)")

destroy-infra: ## Tear down the CloudFormation stack (data bucket is retained)
	./infra/deploy.sh destroy $(if $(STACK),--stack "$(STACK)") $(if $(REGION),--region "$(REGION)")

invoke-lambda: ## Manually trigger the AMFI Connector Lambda
	./infra/deploy.sh invoke $(if $(STACK),--stack "$(STACK)") $(if $(REGION),--region "$(REGION)")

infra-status: ## Show CloudFormation stack status and outputs
	./infra/deploy.sh status $(if $(STACK),--stack "$(STACK)") $(if $(REGION),--region "$(REGION)")
