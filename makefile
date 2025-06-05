.PHONY: help install dev test lint format security check clean build docker

PYTHON := python3
PIP := pip3
PROJECT_NAME := brandscope-ai

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PIP) install -e .

dev: ## Install development dependencies
	$(PIP) install -e ".[dev,docs]"
	pre-commit install

test: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage
	pytest tests/ -x -v

lint: ## Run all linting tools
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

security: ## Run security checks
	bandit -r src/
	safety check

check: lint security test ## Run all quality checks

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean ## Build distribution packages
	$(PYTHON) -m build

docker: ## Build Docker image
	docker build -t $(PROJECT_NAME):latest .

docker-dev: ## Build development Docker image
	docker build -f Dockerfile.dev -t $(PROJECT_NAME):dev .

run: ## Run the CLI
	$(PYTHON) -m src.cli

demo: ## Run demo project creation
	$(PYTHON) -m src.cli new --brand "Demo Brand" --category "speakers" --products "Speaker One,Speaker Two"