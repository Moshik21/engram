.PHONY: up down restart logs status ps health build clean test lint

# Full-mode Engram: Docker stack + all features enabled
# docker-compose.yml defaults to standard profile (all features on):
#   - consolidation_profile=standard (LLM triage, PMI inference, transitivity,
#     dream associations, infer+merge LLM validation, Sonnet escalation)
#   - worker_enabled=true (background episode processing)
#   - pressure-triggered consolidation cycles
#
# Override via .env or env vars, e.g.:
#   ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE=observe make up

up: ## Start full stack — all features on (standard profile)
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

clean: ## Stop and remove volumes (WARNING: deletes all data)
	docker compose down -v

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
