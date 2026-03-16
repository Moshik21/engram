.PHONY: up down restart logs status ps health build clean test lint bundle mcp \
       up-helix down-helix restart-helix logs-helix mcp-helix

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

# HelixDB stack (docker-compose.helix.yml)

HELIX_COMPOSE = docker compose -f docker-compose.helix.yml

up-helix: ## Start Helix stack — HelixDB backend + standard consolidation
	$(HELIX_COMPOSE) up -d --build

down-helix: ## Stop Helix stack
	$(HELIX_COMPOSE) down

restart-helix: ## Restart Helix stack (rebuild)
	$(HELIX_COMPOSE) down
	$(HELIX_COMPOSE) up -d --build

logs-helix: ## Tail all Helix stack logs
	$(HELIX_COMPOSE) logs -f

mcp-helix: ## Start MCP server (streamable HTTP on port 8200, connects to Docker Helix stack)
	cd server && ENGRAM_MODE=helix \
		ENGRAM_HELIX__HOST=localhost \
		ENGRAM_HELIX__PORT=$${ENGRAM_HELIX_PORT:-6969} \
		uv run python -m engram.mcp.server --transport streamable-http

# Native HelixDB (in-process via PyO3)

HELIX_REPO = helixdb-cfg/.helix/dev/helix-repo-copy

patch-helix: ## Re-apply HDB fork changes after helix push dev
	cd $(HELIX_REPO) && git apply ../helix-fork.patch
	@echo "Fork patch applied (batch endpoint, HTTP/2, gRPC, PyO3 crate)."

build-native: ## Build helix_native PyO3 extension (in-process HelixDB)
	cd $(HELIX_REPO)/helix-python && \
	   cargo clean -p helix-python --release 2>/dev/null || true && \
	   VIRTUAL_ENV=$(CURDIR)/server/.venv maturin develop --release
	@echo "helix_native installed."

up-native: build-native ## Start with native in-process HelixDB (no Docker)
	cd server && ENGRAM_MODE=helix \
		ENGRAM_HELIX__TRANSPORT=native \
		uv run engram serve

mcp-native: build-native ## Start MCP server with native in-process HelixDB
	cd server && ENGRAM_MODE=helix \
		ENGRAM_HELIX__TRANSPORT=native \
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
