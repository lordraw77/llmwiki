# =============================================================================
# Wiki-LLM — Makefile
# -----------------------------------------------------------------------------
# Common developer and Docker workflows. Override variables on the command line:
#
#   make build
#   make publish DOCKER_USER=youruser VERSION=0.1.0
#   make run-local PORT=9000
#
# Variables:
#   DOCKER_USER  Docker Hub namespace (user or org). Default: wikillm
#   IMAGE_NAME   Repository name. Default: wikillm
#   VERSION      Image version tag. Default: project version from pyproject.toml
#   PLATFORMS    Target platforms for multi-arch publish.
# =============================================================================

DOCKER_USER  ?= wikillm
IMAGE_NAME   ?= wikillm
VERSION      ?= $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)
IMAGE        := $(DOCKER_USER)/$(IMAGE_NAME)
PLATFORMS    ?= linux/amd64,linux/arm64
PORT         ?= 8000
PYTHON       ?= python3.12

# Pass WITH_LOCAL_EMBEDDINGS=true to bake in sentence-transformers (large).
WITH_LOCAL_EMBEDDINGS ?= false

.DEFAULT_GOAL := help

# --- Meta --------------------------------------------------------------------
.PHONY: help
help: ## Show this help.
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Local Python ------------------------------------------------------------
.PHONY: venv
venv: ## Create a local virtualenv and install dependencies.
	$(PYTHON) -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt

.PHONY: run-http
run-http: ## Run the HTTP server locally (REST + MCP HTTP/SSE).
	$(PYTHON) run_http.py

.PHONY: run-stdio
run-stdio: ## Run the MCP stdio server locally.
	$(PYTHON) run_stdio.py

.PHONY: test
test: ## Run the HTTP and stdio integration test scripts.
	$(PYTHON) test_http.py
	$(PYTHON) test_stdio.py --skip-tool-calls

# --- Docker: build & run -----------------------------------------------------
.PHONY: build
build: ## Build the Docker image (tags :VERSION and :latest).
	docker build \
		--build-arg WITH_LOCAL_EMBEDDINGS=$(WITH_LOCAL_EMBEDDINGS) \
		-t $(IMAGE):$(VERSION) \
		-t $(IMAGE):latest \
		.

.PHONY: run
run: ## Run the built image (maps PORT, persists data in a named volume).
	docker run --rm -it \
		-p $(PORT):8000 \
		--env-file .env \
		-v wikillm-data:/data \
		--name wikillm \
		$(IMAGE):latest

.PHONY: shell
shell: ## Open a shell inside the built image.
	docker run --rm -it --entrypoint /bin/bash $(IMAGE):latest

# --- Docker Compose ----------------------------------------------------------
.PHONY: up
up: ## Start the stack with docker compose (detached).
	docker compose up -d --build

.PHONY: up-qdrant
up-qdrant: ## Start the stack including the Qdrant server.
	docker compose --profile qdrant up -d --build

.PHONY: down
down: ## Stop and remove the compose stack.
	docker compose down

.PHONY: logs
logs: ## Follow the app container logs.
	docker compose logs -f wikillm

# --- Docker Hub publish ------------------------------------------------------
.PHONY: publish
publish: ## Build & push a multi-arch image to Docker Hub (see scripts/publish.sh).
	DOCKER_USER=$(DOCKER_USER) IMAGE_NAME=$(IMAGE_NAME) VERSION=$(VERSION) \
		PLATFORMS=$(PLATFORMS) WITH_LOCAL_EMBEDDINGS=$(WITH_LOCAL_EMBEDDINGS) \
		./scripts/publish.sh

.PHONY: push
push: ## Push already-built :VERSION and :latest tags (single arch).
	docker push $(IMAGE):$(VERSION)
	docker push $(IMAGE):latest

# --- Housekeeping ------------------------------------------------------------
.PHONY: clean
clean: ## Remove local Python caches.
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info build dist
