.PHONY: help setup install install-dev train evaluate infer clean test lint format docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make setup          - Create virtual environment and install dependencies"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make train          - Train the model (requires DATA_DIR environment variable)"
	@echo "  make evaluate       - Evaluate the model (requires MODEL and DATA_DIR)"
	@echo "  make infer          - Run inference on an image (requires MODEL and IMAGE)"
	@echo "  make test           - Run all tests"
	@echo "  make lint           - Run code quality checks"
	@echo "  make format         - Auto-format code with black"
	@echo "  make clean          - Remove build artifacts and cache"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements-dev.txt
	@echo "Virtual environment created. Run: source .venv/bin/activate"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

train:
	python -m src.cli train --data-dir $(DATA_DIR) --epochs 10 --batch-size 32

evaluate:
	python -m src.cli evaluate --model $(MODEL) --data-dir $(DATA_DIR)

infer:
	python -m src.cli infer --model $(MODEL) --image $(IMAGE)

test:
	python -m unittest discover -s tests -p 'test_*.py' -v

lint:
	flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics

format:
	black src tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/

docker-build:
	docker build -t face-mask-detector .

docker-run:
	docker run -v $$(pwd)/data:/app/data -v $$(pwd)/models:/app/models face-mask-detector
