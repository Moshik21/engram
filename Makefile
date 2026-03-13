.PHONY: up down restart logs status ps health build clean test lint bundle mcp

# Developer/manual full-mode Engram: source-built Docker stack + coherent rework integration enabled
# docker-compose.yml defaults to:
#   - consolidation_profile=standard (worker + consolidation + maturation)
#   - integration_profile=rework (cue layer + cue recall + cue policy +
#     projector path + full recall waves)
#   - worker_enabled=true (background episode processing)
#
# Override via .env or env vars, e.g.:
#   ENGRAM_ACTIVATION__INTEGRATION_PROFILE=off make up
#   ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE=observe make up

up: ## Start full stack — standard consolidation + rework integration preset
	docker compose up -d --build

down: ## Stop full stack
	docker compose down

restart: ## Restart full stack (rebuild)
	docker compose down
	docker compose up -d --build

logs: ## Tail all service logs
	docker compose logs -f

logs-server: ## Tail server logs only
	docker compose logs -f server

status: ## Show service status + health
	@docker compose ps
	@echo ""
	@curl -sf http://localhost:8100/health | python3 -m json.tool 2>/dev/null || echo "Server not responding"

ps: ## Show running containers
	docker compose ps

health: ## Quick health check
	@curl -sf http://localhost:8100/health | python3 -m json.tool

build: ## Rebuild without starting
	docker compose build

bundle: ## Build the public install bundle into dist/install
	python3 scripts/build_install_bundle.py

clean: ## Stop and remove volumes (WARNING: deletes all data)
	docker compose down -v

mcp: ## Start MCP server (streamable HTTP on port 8200, connects to Docker full stack)
	cd server && ENGRAM_MODE=full \
		ENGRAM_FALKORDB__HOST=localhost \
		ENGRAM_FALKORDB__PORT=6380 \
		ENGRAM_FALKORDB__PASSWORD=$${ENGRAM_FALKORDB_PASSWORD:-engram_dev} \
		ENGRAM_REDIS__URL=redis://:$${ENGRAM_REDIS_PASSWORD:-engram_dev}@localhost:6381/0 \
		uv run python -m engram.mcp.server --transport streamable-http

# Development commands

test: ## Run all tests (backend + frontend)
	cd server && uv run pytest -m "not requires_docker" -v
	cd dashboard && pnpm test

test-backend: ## Run backend tests only
	cd server && uv run pytest -m "not requires_docker" -v

test-frontend: ## Run frontend tests only
	cd dashboard && pnpm test

lint: ## Run ruff linter
	cd server && uv run ruff check .

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
