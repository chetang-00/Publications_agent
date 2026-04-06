# Research Publications Agent — common tasks. Run `make help`.
COMPOSE  ?= docker compose
NODE     := scripts/node-docker.sh
SEED_CSV ?= data/seed/papers_with_cluster_labels.csv

.DEFAULT_GOAL := help
.PHONY: help up down restart logs ps seed reindex test test-backend test-frontend test-live lint smoke e2e dev-qdrant dev-api dev-web clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.env:
	@cp .env.example .env
	@echo "Created .env from .env.example. Set PORTKEY_API_KEY in it, then run the command again." && exit 1

up: .env ## Build and start the stack (web UI on http://localhost:8080)
	$(COMPOSE) up -d --build --wait
	@echo "Ready: http://localhost:$$(grep -E '^WEB_PORT=' .env | cut -d= -f2 | grep . || echo 8080)"

down: ## Stop the stack (data volumes are kept)
	$(COMPOSE) down

restart: ## Restart the API and web containers
	$(COMPOSE) restart api web

logs: ## Follow logs from all services
	$(COMPOSE) logs -f --tail=200

ps: ## Show service status and health
	$(COMPOSE) ps

seed: ## Load the publications CSV and build the vector index
	@test -f $(SEED_CSV) || (echo "Put the CSV at $(SEED_CSV) first." && exit 1)
	$(COMPOSE) exec api python -m app.cli seed --csv /seed/$(notdir $(SEED_CSV))

reindex: ## Rebuild all vector indexes (after changing EMBEDDING_MODEL)
	$(COMPOSE) exec api python -m app.cli reindex

test: test-backend test-frontend ## Run all offline tests

test-backend: ## Backend tests with coverage
	cd backend && uv run pytest --cov

test-frontend: ## Frontend tests (inside the Node 22 container)
	$(NODE) sh -c "npm ci --no-audit --no-fund && npx vitest run"

test-live: ## Check the real Portkey gateway: streaming, tool calls, embeddings (needs PORTKEY_API_KEY)
	cd backend && RUN_LIVE=1 uv run pytest -m live -v

lint: ## Ruff + TypeScript checks
	cd backend && uv run ruff check app tests && uv run ruff format --check app tests
	$(NODE) npx tsc --noEmit

smoke: ## End-to-end checks against the running stack and the real gateway (needs PORTKEY_API_KEY)
	cd backend && uv run python ../scripts/smoke.py

e2e: ## Edge-case end-to-end suite against the running stack (own sample data; restarts the API twice)
	cd backend && uv run python ../scripts/e2e.py

dev-qdrant: ## Development: run only Qdrant on localhost:6333
	docker run --rm -p 127.0.0.1:6333:6333 -v pa-dev-qdrant:/qdrant/storage qdrant/qdrant:v1.19.1

dev-api: ## Development: API with auto-reload on :8000 (uses the commented dev settings in .env)
	cd backend && uv run uvicorn app.main:app --reload --port 8000

dev-web: ## Development: Vite dev server on :5173, proxying /api to dev-api
	NODE_DOCKER_ARGS="-p 127.0.0.1:5173:5173 -e VITE_API_TARGET=http://host.docker.internal:8000" \
	  $(NODE) sh -c "npm ci --no-audit --no-fund && npx vite --host 0.0.0.0"

clean: ## Stop the stack AND delete all data volumes (asks first)
	@read -p "Delete all conversations, documents and vectors? [y/N] " ans && [ "$$ans" = "y" ]
	$(COMPOSE) down -v
