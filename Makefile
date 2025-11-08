# Makefile for CSG V4 Development
# ==================================

.PHONY: help install install-dev format lint typecheck test check clean pre-commit-install pre-commit-run

# Default target
help:
	@echo "CSG V4 Development Commands"
	@echo "============================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install              Install package in development mode"
	@echo "  make install-dev          Install package with dev dependencies"
	@echo "  make pre-commit-install   Install pre-commit hooks"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  make format               Format code with Black and isort"
	@echo "  make lint                 Lint code with Ruff"
	@echo "  make typecheck            Type check with mypy"
	@echo "  make check                Run all checks (format, lint, typecheck)"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test                 Run tests with pytest"
	@echo ""
	@echo "Git Hook Commands:"
	@echo "  make pre-commit-run       Run pre-commit on all files"
	@echo ""
	@echo "Cleanup Commands:"
	@echo "  make clean                Remove cache and build files"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

pre-commit-install: install-dev
	pre-commit install

# Code formatting
format:
	@echo "Running isort..."
	isort src/ scripts/ demos/
	@echo ""
	@echo "Running Black..."
	black src/ scripts/ demos/
	@echo ""
	@echo "✓ Code formatting complete!"

# Linting
lint:
	@echo "Running Ruff..."
	ruff check src/ scripts/ demos/
	@echo ""
	@echo "✓ Linting complete!"

# Type checking
typecheck:
	@echo "Running mypy..."
	mypy src/
	@echo ""
	@echo "✓ Type checking complete!"

# Run all checks
check: format lint typecheck
	@echo ""
	@echo "✓ All checks complete!"

# Testing
test:
	@echo "Running pytest..."
	pytest
	@echo ""
	@echo "✓ Tests complete!"

# Pre-commit
pre-commit-run:
	@echo "Running pre-commit on all files..."
	pre-commit run --all-files
	@echo ""
	@echo "✓ Pre-commit checks complete!"

# Cleanup
clean:
	@echo "Cleaning cache and build files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/
	@echo "✓ Cleanup complete!"
