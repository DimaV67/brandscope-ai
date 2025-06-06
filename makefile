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

test: ## Run tests with coverage
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage
	$(PYTHON) -m pytest tests/ -x -v

lint: ## Run all linting tools
	$(PYTHON) -m black --check src/ tests/
	$(PYTHON) -m isort --check-only src/ tests/
	$(PYTHON) -m flake8 src/ tests/
	$(PYTHON) -m mypy src/

format: ## Format code
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m isort src/ tests/

security: ## Run security checks
	$(PYTHON) -m bandit -r src/
	$(PYTHON) -m safety check

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

run: ## Run the CLI (when entry point is configured)
	$(PYTHON) -m src.main

demo: ## Create a demo project
	@echo "Demo mode - creating sample project"
	@echo "Use: make run"

setup-dev: dev ## Complete development setup
	@echo "Development environment setup complete!"
	@echo "Run 'make check' to verify everything works"

init: ## Initialize project structure
	mkdir -p src/models src/core src/utils src/commands
	mkdir -p config/prompt_templates config/frameworks/category_extensions
	touch src/__init__.py src/models/__init__.py src/core/__init__.py src/utils/__init__.py src/commands/__init__.py
	@echo "Project structure initialized"