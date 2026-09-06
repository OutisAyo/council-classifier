.PHONY: install install-dev models run dashboard test lint fmt docker-build docker-run

install:            ## Install API runtime dependencies
	pip install -r requirements.txt

install-dev:        ## Install test + lint dependencies
	pip install -r requirements-dev.txt

models:            ## Pull the trained model artifacts (Git LFS)
	git lfs pull

run:               ## Start the FastAPI service with autoreload
	uvicorn app.main:app --reload

dashboard:         ## Start the Streamlit dashboard
	streamlit run dashboard/app.py

test:              ## Run the test suite
	pytest

lint:              ## Lint with ruff
	ruff check .

fmt:               ## Auto-fix lint issues
	ruff check --fix .

docker-build:      ## Build the container image
	docker build -t council-classifier .

docker-run:        ## Run the container on port 8000
	docker run -p 8000:8000 council-classifier
