.PHONY: install install-dev lint format test test-cov clean run optimize benchmark docker-cpu docker-gpu api help

PYTHON := python
PIP := pip
PROJECT := objatrack-xl

help:
	@echo "ObjaTrack-XL - High-Performance Object Detection & Tracking"
	@echo ""
	@echo "Usage:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make lint           Run linting checks"
	@echo "  make format         Format code"
	@echo "  make test           Run tests"
	@echo "  make test-cov       Run tests with coverage"
	@echo "  make clean          Clean build artifacts"
	@echo "  make run            Run the main pipeline"
	@echo "  make optimize       Run model optimization"
	@echo "  make benchmark      Run performance benchmark"
	@echo "  make docker-cpu     Build CPU Docker image"
	@echo "  make docker-gpu     Build GPU Docker image"
	@echo "  make api            Start the API server"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check src/ tests/ tools/
	mypy src/

format:
	ruff check --fix src/ tests/ tools/
	ruff format src/ tests/ tools/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/

run:
	$(PYTHON) main.py --config configs/default.yaml

optimize:
	$(PYTHON) tools/optimize.py --config configs/optimization.yaml

benchmark:
	$(PYTHON) tools/benchmark.py --config configs/benchmark.yaml

docker-cpu:
	docker build -f docker/Dockerfile.cpu -t $(PROJECT):cpu .

docker-gpu:
	docker build -f docker/Dockerfile.gpu -t $(PROJECT):gpu .

api:
	$(PYTHON) -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
