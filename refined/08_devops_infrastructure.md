# 08 - DevOps, CI/CD & Infrastructure

## Overview

This document defines the container orchestration, persistence, health-checking, CI/CD pipeline, backup/restore strategy, resource sizing, and development workflow for Engram. The original spec deferred DevOps until Week 7-8; this design moves infrastructure concerns to Week 1 so every component runs on solid foundations from day one.

---

## 1. Docker Compose (Production)

### 1.1 `docker-compose.yml`

```yaml
# docker-compose.yml
# Production-ready Engram stack
# Usage: docker compose up -d

name: engram

services:
  # ---------------------------------------------------------------
  # FalkorDB  (graph store + vector index)
  # Ships as a Redis module, so the container IS a Redis server.
  # ---------------------------------------------------------------
  falkordb:
    image: falkordb/falkordb:v4.4.1          # pin to tested tag
    container_name: engram-falkordb
    restart: unless-stopped
    ports:
      - "${ENGRAM_FALKORDB_PORT:-6380}:6379"
    volumes:
      - falkordb_data:/data
    environment:
      - REDIS_ARGS=--requirepass ${ENGRAM_FALKORDB_PASSWORD:-changeme} --appendonly yes --appendfsync everysec
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${ENGRAM_FALKORDB_PASSWORD:-changeme}", "PING"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 15s
    deploy:
      resources:
        limits:
          memory: 1g
          cpus: "1.0"
        reservations:
          memory: 256m
          cpus: "0.25"
    networks:
      - engram-net

  # ---------------------------------------------------------------
  # Redis  (activation state cache + pub/sub for WS fan-out)
  # ---------------------------------------------------------------
  redis:
    image: redis:7.4-alpine
    container_name: engram-redis
    restart: unless-stopped
    ports:
      - "${ENGRAM_REDIS_PORT:-6381}:6379"
    volumes:
      - redis_data:/data
      - ./config/redis.conf:/usr/local/etc/redis/redis.conf:ro
    command: ["redis-server", "/usr/local/etc/redis/redis.conf"]
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${ENGRAM_REDIS_PASSWORD:-changeme}", "PING"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
    deploy:
      resources:
        limits:
          memory: 512m
          cpus: "0.5"
        reservations:
          memory: 128m
          cpus: "0.1"
    networks:
      - engram-net

  # ---------------------------------------------------------------
  # Engram Server  (FastAPI + MCP)
  # ---------------------------------------------------------------
  server:
    build:
      context: ./server
      dockerfile: Dockerfile
    container_name: engram-server
    restart: unless-stopped
    ports:
      - "${ENGRAM_SERVER_PORT:-8100}:8100"
    depends_on:
      falkordb:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      - ENGRAM_ENV=production
      - ENGRAM_LOG_LEVEL=${ENGRAM_LOG_LEVEL:-info}
      - ENGRAM_SERVER_HOST=0.0.0.0
      - ENGRAM_SERVER_PORT=8100
      # Data stores
      - ENGRAM_FALKORDB_URL=redis://falkordb:6379
      - ENGRAM_FALKORDB_PASSWORD=${ENGRAM_FALKORDB_PASSWORD:-changeme}
      - ENGRAM_REDIS_URL=redis://redis:6379
      - ENGRAM_REDIS_PASSWORD=${ENGRAM_REDIS_PASSWORD:-changeme}
      # AI provider (Vercel AI SDK compatible)
      - ENGRAM_AI_PROVIDER=${ENGRAM_AI_PROVIDER:-anthropic}
      - ENGRAM_ANTHROPIC_API_KEY=${ENGRAM_ANTHROPIC_API_KEY:?ENGRAM_ANTHROPIC_API_KEY is required}
      # Activation engine
      - ENGRAM_ACTIVATION_DECAY_RATE=${ENGRAM_ACTIVATION_DECAY_RATE:-0.05}
      - ENGRAM_ACTIVATION_SPREAD_FACTOR=${ENGRAM_ACTIVATION_SPREAD_FACTOR:-0.3}
      - ENGRAM_ACTIVATION_SPREAD_HOPS=${ENGRAM_ACTIVATION_SPREAD_HOPS:-2}
      # Auth (see 05_security_model.md for full auth design)
      - ENGRAM_AUTH_ENABLED=${ENGRAM_AUTH_ENABLED:-false}
      - ENGRAM_AUTH_SECRET=${ENGRAM_AUTH_SECRET:-}
      - ENGRAM_MASTER_KEY=${ENGRAM_MASTER_KEY:-}
      - ENGRAM_JWT_SECRET=${ENGRAM_JWT_SECRET:-}
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8100/health"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 20s
    deploy:
      resources:
        limits:
          memory: 1g
          cpus: "1.0"
        reservations:
          memory: 256m
          cpus: "0.25"
    networks:
      - engram-net

  # ---------------------------------------------------------------
  # Dashboard  (React SPA served by nginx)
  # ---------------------------------------------------------------
  dashboard:
    build:
      context: ./dashboard
      dockerfile: Dockerfile
      args:
        - VITE_API_URL=${ENGRAM_DASHBOARD_API_URL:-http://localhost:8100}
        - VITE_WS_URL=${ENGRAM_DASHBOARD_WS_URL:-ws://localhost:8100/ws}
    container_name: engram-dashboard
    restart: unless-stopped
    ports:
      - "${ENGRAM_DASHBOARD_PORT:-3000}:80"
    depends_on:
      server:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:80/"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          memory: 128m
          cpus: "0.25"
        reservations:
          memory: 64m
          cpus: "0.05"
    networks:
      - engram-net

volumes:
  falkordb_data:
    name: engram_falkordb_data
  redis_data:
    name: engram_redis_data

networks:
  engram-net:
    name: engram-net
    driver: bridge
```

### 1.2 Redis Configuration (`config/redis.conf`)

```conf
# config/redis.conf
# Activation state Redis -- tuned for low-latency reads + AOF durability

# Auth
requirepass ${ENGRAM_REDIS_PASSWORD:-changeme}

# Persistence -- AOF with everysec fsync
appendonly yes
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# RDB snapshots as secondary safety net
save 900 1
save 300 10
save 60 10000

# Memory policy -- activation state should not silently evict
maxmemory 384mb
maxmemory-policy noeviction

# Performance
hz 10
tcp-keepalive 300
timeout 0
```

**Why AOF with everysec:** Activation state is hot data that changes frequently. AOF captures every write with at most 1 second of data loss on crash, which is acceptable for activation values that will naturally recalculate. The RDB snapshots provide a secondary backup for point-in-time restore.

### 1.3 Startup Ordering & Dependency Chain

```
falkordb (healthy) ─┐
                    ├─> server (healthy) ─> dashboard
redis    (healthy) ─┘
```

The `depends_on` conditions guarantee:
1. FalkorDB and Redis are confirmed healthy (PING succeeds) before the server starts
2. The server health endpoint (`/health`) returns 200 before the dashboard starts
3. All containers use `restart: unless-stopped` so they recover from transient failures but stay down after intentional `docker compose stop`

### 1.4 Activation State Recovery on Startup

The server startup sequence must handle the case where Redis was wiped or restarted while FalkorDB retains node properties.

```python
# Pseudocode for server/engram/activation/recovery.py

async def recover_activation_state(
    graph_store: GraphStore,
    redis_client: Redis,
) -> RecoveryReport:
    """
    Called during server startup, after both stores are confirmed healthy.
    Restores activation state from FalkorDB node properties into Redis
    if Redis keys are missing.
    """
    # 1. Check if Redis has activation keys
    existing_keys = await redis_client.keys("engram:activation:*")

    if existing_keys:
        logger.info("Redis activation state found (%d keys), skipping recovery", len(existing_keys))
        return RecoveryReport(recovered=0, skipped=len(existing_keys))

    # 2. Redis is empty -- load from FalkorDB snapshots
    logger.warning("No activation state in Redis, recovering from FalkorDB")
    nodes = await graph_store.query(
        "MATCH (e:Entity) "
        "RETURN e.name, e.activation_base, e.activation_current, "
        "       e.access_count, e.last_accessed"
    )

    pipeline = redis_client.pipeline()
    for node in nodes:
        key = f"engram:activation:{node['e.name']}"
        pipeline.hset(key, mapping={
            "base_activation": node["e.activation_base"] or 0.1,
            "current_activation": node["e.activation_current"] or 0.0,
            "access_count": node["e.access_count"] or 0,
            "last_accessed": node["e.last_accessed"] or "",
        })
    await pipeline.execute()

    logger.info("Recovered activation state for %d nodes", len(nodes))
    return RecoveryReport(recovered=len(nodes), skipped=0)
```

The `/health` endpoint should report `recovering` status during this phase and only return 200 once recovery completes. This prevents the dashboard from connecting before data is ready.

---

## 2. Docker Compose (Development)

### 2.1 `docker-compose.dev.yml`

```yaml
# docker-compose.dev.yml
# Development overrides -- hot reload, debug ports, relaxed limits
# Usage: docker compose -f docker-compose.yml -f docker-compose.dev.yml up

services:
  falkordb:
    ports:
      - "6380:6379"               # always expose in dev
    environment:
      - REDIS_ARGS=--requirepass devpass --appendonly no
    deploy:
      resources:
        limits:
          memory: 512m

  redis:
    ports:
      - "6381:6379"
    command: ["redis-server", "--requirepass", "devpass", "--appendonly", "no"]
    deploy:
      resources:
        limits:
          memory: 256m

  server:
    build:
      context: ./server
      dockerfile: Dockerfile.dev
    volumes:
      - ./server:/app                # mount source for hot reload
      - /app/.venv                   # exclude venv from mount
    environment:
      - ENGRAM_ENV=development
      - ENGRAM_LOG_LEVEL=debug
      - ENGRAM_FALKORDB_PASSWORD=devpass
      - ENGRAM_REDIS_PASSWORD=devpass
      - ENGRAM_AUTH_ENABLED=false
    command: ["uvicorn", "engram.main:app", "--host", "0.0.0.0", "--port", "8100", "--reload", "--reload-dir", "/app/engram"]
    deploy:
      resources:
        limits:
          memory: 512m

  dashboard:
    build:
      context: ./dashboard
      dockerfile: Dockerfile.dev
    volumes:
      - ./dashboard/src:/app/src     # mount source for Vite HMR
      - ./dashboard/public:/app/public
    ports:
      - "3000:5173"                  # Vite dev server port
    environment:
      - VITE_API_URL=http://localhost:8100
      - VITE_WS_URL=ws://localhost:8100/ws
    command: ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
    deploy:
      resources:
        limits:
          memory: 512m
```

### 2.2 Development Dockerfiles

**`server/Dockerfile.dev`**:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --dev
COPY . .
CMD ["uvicorn", "engram.main:app", "--host", "0.0.0.0", "--port", "8100", "--reload"]
```

**`dashboard/Dockerfile.dev`**:
```dockerfile
FROM node:22-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
```

### 2.3 Dev Workflow

```bash
# Start full dev stack with hot reload
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Or run only infra and develop locally
docker compose up falkordb redis
cd server && uv run uvicorn engram.main:app --reload --port 8100
cd dashboard && npm run dev
```

---

## 3. Production Dockerfiles

### 3.1 Server Dockerfile

```dockerfile
# server/Dockerfile
FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/.venv /app/.venv
COPY engram/ ./engram/
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8100
CMD ["uvicorn", "engram.main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "2"]
```

### 3.2 Dashboard Dockerfile

```dockerfile
# dashboard/Dockerfile
FROM node:22-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_URL
ARG VITE_WS_URL
RUN npm run build

FROM nginx:1.27-alpine
RUN apk add --no-cache curl
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## 4. Health Check Endpoints

### 4.1 Server `/health` Endpoint

```python
# server/engram/api/health.py
from fastapi import APIRouter
from pydantic import BaseModel
from enum import Enum

class ServiceStatus(str, Enum):
    healthy = "healthy"
    degraded = "degraded"
    unhealthy = "unhealthy"
    recovering = "recovering"

class HealthResponse(BaseModel):
    status: ServiceStatus
    version: str
    services: dict[str, ServiceStatus]

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(
    graph_store: GraphStore = Depends(get_graph_store),
    redis_client: Redis = Depends(get_redis),
) -> HealthResponse:
    services = {}

    # FalkorDB
    try:
        await graph_store.ping()
        services["falkordb"] = ServiceStatus.healthy
    except Exception:
        services["falkordb"] = ServiceStatus.unhealthy

    # Redis
    try:
        await redis_client.ping()
        services["redis"] = ServiceStatus.healthy
    except Exception:
        services["redis"] = ServiceStatus.unhealthy

    # Overall status
    if all(s == ServiceStatus.healthy for s in services.values()):
        status = ServiceStatus.healthy
    elif any(s == ServiceStatus.unhealthy for s in services.values()):
        status = ServiceStatus.unhealthy
    else:
        status = ServiceStatus.degraded

    return HealthResponse(
        status=status,
        version=__version__,
        services=services,
    )
```

The Docker healthcheck runs `curl -sf http://localhost:8100/health`. The endpoint returns HTTP 200 for healthy/degraded and HTTP 503 for unhealthy/recovering. Docker treats non-200 as unhealthy, which blocks dependent services from starting.

### 4.2 Health Check Summary

| Service   | Method                              | Interval | Timeout | Retries | Start Period |
|-----------|-------------------------------------|----------|---------|---------|-------------|
| FalkorDB  | `redis-cli -a $PASS PING`          | 10s      | 5s      | 5       | 15s         |
| Redis     | `redis-cli -a $PASS PING`          | 10s      | 5s      | 5       | 10s         |
| Server    | `curl -sf http://localhost:8100/health` | 15s  | 5s      | 5       | 20s         |
| Dashboard | `curl -sf http://localhost:80/`     | 15s      | 5s      | 3       | 10s         |

---

## 5. GitHub Actions CI Pipeline

### 5.1 Main CI Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

env:
  PYTHON_VERSION: "3.12"
  NODE_VERSION: "22"

jobs:
  # ----- Python linting & type checking -----
  lint-server:
    name: Lint Server
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: server
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v5
        with:
          version: "latest"

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: uv sync --frozen --dev

      - name: Ruff lint
        run: uv run ruff check .

      - name: Ruff format check
        run: uv run ruff format --check .

      - name: Mypy type check
        run: uv run mypy engram/

  # ----- Dashboard linting -----
  lint-dashboard:
    name: Lint Dashboard
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: dashboard
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: "npm"
          cache-dependency-path: dashboard/package-lock.json

      - name: Install dependencies
        run: npm ci

      - name: ESLint
        run: npm run lint

      - name: TypeScript type check
        run: npm run typecheck

  # ----- Python unit tests -----
  test-server:
    name: Test Server
    runs-on: ubuntu-latest
    needs: lint-server
    defaults:
      run:
        working-directory: server
    services:
      falkordb:
        image: falkordb/falkordb:v4.4.1
        ports:
          - 6380:6379
        options: >-
          --health-cmd "redis-cli PING"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7.4-alpine
        ports:
          - 6381:6379
        options: >-
          --health-cmd "redis-cli PING"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v5
        with:
          version: "latest"

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: uv sync --frozen --dev

      - name: Run tests
        run: uv run pytest tests/ -v --tb=short --cov=engram --cov-report=xml
        env:
          ENGRAM_FALKORDB_URL: redis://localhost:6380
          ENGRAM_REDIS_URL: redis://localhost:6381
          ENGRAM_ENV: test

      - name: Upload coverage
        if: github.event_name == 'pull_request'
        uses: codecov/codecov-action@v4
        with:
          files: server/coverage.xml
          flags: server

  # ----- Dashboard unit tests -----
  test-dashboard:
    name: Test Dashboard
    runs-on: ubuntu-latest
    needs: lint-dashboard
    defaults:
      run:
        working-directory: dashboard
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: "npm"
          cache-dependency-path: dashboard/package-lock.json

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test -- --run

  # ----- Docker Compose smoke test -----
  smoke-test:
    name: Smoke Test
    runs-on: ubuntu-latest
    needs: [test-server, test-dashboard]
    steps:
      - uses: actions/checkout@v4

      - name: Create .env for CI
        run: |
          cat <<'ENVFILE' > .env
          ENGRAM_FALKORDB_PASSWORD=ci-test-pass
          ENGRAM_REDIS_PASSWORD=ci-test-pass
          ENGRAM_ANTHROPIC_API_KEY=sk-ant-placeholder-for-smoke-test
          ENGRAM_AUTH_ENABLED=false
          ENGRAM_AUTH_SECRET=ci-test-token
          ENGRAM_LOG_LEVEL=warning
          ENVFILE

      - name: Build and start stack
        run: docker compose up -d --build --wait --wait-timeout 120

      - name: Verify all containers healthy
        run: |
          for svc in engram-falkordb engram-redis engram-server engram-dashboard; do
            status=$(docker inspect --format='{{.State.Health.Status}}' "$svc")
            echo "$svc: $status"
            if [ "$status" != "healthy" ]; then
              echo "FAIL: $svc is $status"
              docker logs "$svc" --tail 50
              exit 1
            fi
          done

      - name: Test server health endpoint
        run: |
          response=$(curl -sf http://localhost:8100/health)
          echo "$response"
          echo "$response" | jq -e '.status == "healthy"'

      - name: Test dashboard is reachable
        run: curl -sf http://localhost:3000/ | head -c 200

      - name: Cleanup
        if: always()
        run: docker compose down -v
```

### 5.2 Release Workflow

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - "v*"

permissions:
  contents: write
  packages: write

jobs:
  build-and-push:
    name: Build & Push Images
    runs-on: ubuntu-latest
    strategy:
      matrix:
        include:
          - context: server
            image: ghcr.io/${{ github.repository }}/server
          - context: dashboard
            image: ghcr.io/${{ github.repository }}/dashboard
    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ${{ matrix.image }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - uses: docker/build-push-action@v6
        with:
          context: ${{ matrix.context }}
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  github-release:
    name: GitHub Release
    runs-on: ubuntu-latest
    needs: build-and-push
    steps:
      - uses: actions/checkout@v4
      - uses: softprops/action-gh-release@v2
        with:
          generate_release_notes: true
```

---

## 6. Backup & Restore

### 6.1 Backup Script

```bash
#!/usr/bin/env bash
# scripts/backup.sh
# Creates a timestamped backup of FalkorDB and Redis data.
#
# Usage:
#   ./scripts/backup.sh                     # backup to ./backups/
#   ./scripts/backup.sh /mnt/nas/engram     # backup to custom path
#   ENGRAM_BACKUP_RETAIN=7 ./scripts/backup.sh  # keep last 7 backups

set -euo pipefail

BACKUP_DIR="${1:-./backups}"
RETAIN_COUNT="${ENGRAM_BACKUP_RETAIN:-10}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="${BACKUP_DIR}/${TIMESTAMP}"

echo "=== Engram Backup ==="
echo "Timestamp:  ${TIMESTAMP}"
echo "Target:     ${BACKUP_PATH}"
echo "Retention:  last ${RETAIN_COUNT} backups"
echo ""

mkdir -p "${BACKUP_PATH}"

# 1. Trigger Redis BGSAVE on both instances
echo "[1/4] Triggering BGSAVE on FalkorDB..."
docker exec engram-falkordb redis-cli \
  -a "${ENGRAM_FALKORDB_PASSWORD:-changeme}" BGSAVE
sleep 2  # wait for BGSAVE to start

echo "[2/4] Triggering BGSAVE on Redis..."
docker exec engram-redis redis-cli \
  -a "${ENGRAM_REDIS_PASSWORD:-changeme}" BGSAVE
sleep 2

# Wait for both BGSAVE operations to complete
for container in engram-falkordb engram-redis; do
  pass_var="ENGRAM_FALKORDB_PASSWORD"
  [ "$container" = "engram-redis" ] && pass_var="ENGRAM_REDIS_PASSWORD"
  pass="${!pass_var:-changeme}"

  echo "  Waiting for BGSAVE on ${container}..."
  for i in $(seq 1 60); do
    status=$(docker exec "$container" redis-cli -a "$pass" LASTSAVE 2>/dev/null)
    bg_status=$(docker exec "$container" redis-cli -a "$pass" INFO persistence 2>/dev/null \
      | grep rdb_bgsave_in_progress | tr -d '\r' | cut -d: -f2)
    if [ "$bg_status" = "0" ]; then
      echo "  BGSAVE complete on ${container}"
      break
    fi
    sleep 1
  done
done

# 2. Copy volume data
echo "[3/4] Copying volume data..."
docker run --rm \
  -v engram_falkordb_data:/source:ro \
  -v "$(cd "${BACKUP_PATH}" && pwd)":/backup \
  alpine tar czf /backup/falkordb_data.tar.gz -C /source .

docker run --rm \
  -v engram_redis_data:/source:ro \
  -v "$(cd "${BACKUP_PATH}" && pwd)":/backup \
  alpine tar czf /backup/redis_data.tar.gz -C /source .

# 3. Record metadata
echo "[4/4] Writing metadata..."
cat > "${BACKUP_PATH}/metadata.json" <<METAEOF
{
  "timestamp": "${TIMESTAMP}",
  "falkordb_image": "$(docker inspect --format='{{.Config.Image}}' engram-falkordb)",
  "redis_image": "$(docker inspect --format='{{.Config.Image}}' engram-redis)",
  "server_image": "$(docker inspect --format='{{.Config.Image}}' engram-server 2>/dev/null || echo 'N/A')",
  "falkordb_data_size": "$(du -sh "${BACKUP_PATH}/falkordb_data.tar.gz" | cut -f1)",
  "redis_data_size": "$(du -sh "${BACKUP_PATH}/redis_data.tar.gz" | cut -f1)"
}
METAEOF

cat "${BACKUP_PATH}/metadata.json"

# 4. Prune old backups
echo ""
echo "Pruning old backups (keeping last ${RETAIN_COUNT})..."
ls -dt "${BACKUP_DIR}"/*/ 2>/dev/null | tail -n +$((RETAIN_COUNT + 1)) | while read -r old; do
  echo "  Removing: ${old}"
  rm -rf "${old}"
done

echo ""
echo "Backup complete: ${BACKUP_PATH}"
```

### 6.2 Restore Script

```bash
#!/usr/bin/env bash
# scripts/restore.sh
# Restores Engram from a backup created by backup.sh.
#
# Usage:
#   ./scripts/restore.sh ./backups/20260227_120000

set -euo pipefail

BACKUP_PATH="${1:?Usage: restore.sh <backup_path>}"

if [ ! -f "${BACKUP_PATH}/metadata.json" ]; then
  echo "ERROR: ${BACKUP_PATH}/metadata.json not found. Is this a valid backup?"
  exit 1
fi

echo "=== Engram Restore ==="
echo "Source: ${BACKUP_PATH}"
cat "${BACKUP_PATH}/metadata.json"
echo ""

read -rp "This will REPLACE all current data. Continue? [y/N] " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
  echo "Aborted."
  exit 0
fi

# 1. Stop services that use the volumes
echo "[1/3] Stopping services..."
docker compose stop server dashboard

# 2. Restore volumes
echo "[2/3] Restoring FalkorDB data..."
docker compose stop falkordb
docker run --rm \
  -v engram_falkordb_data:/target \
  -v "$(cd "${BACKUP_PATH}" && pwd)":/backup:ro \
  alpine sh -c "rm -rf /target/* && tar xzf /backup/falkordb_data.tar.gz -C /target"

echo "  Restoring Redis data..."
docker compose stop redis
docker run --rm \
  -v engram_redis_data:/target \
  -v "$(cd "${BACKUP_PATH}" && pwd)":/backup:ro \
  alpine sh -c "rm -rf /target/* && tar xzf /backup/redis_data.tar.gz -C /target"

# 3. Restart everything
echo "[3/3] Restarting stack..."
docker compose up -d

echo ""
echo "Restore complete. Waiting for health checks..."
docker compose up --wait --wait-timeout 120
echo "All services healthy."
```

### 6.3 Cron Automation

```bash
# Example crontab entry -- daily backup at 3 AM
0 3 * * * cd /opt/engram && ./scripts/backup.sh /mnt/backups/engram >> /var/log/engram-backup.log 2>&1
```

---

## 7. Environment Variable Reference

All environment variables use the `ENGRAM_` prefix for consistency, including the Anthropic API key (wrapped as `ENGRAM_ANTHROPIC_API_KEY`).

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_ENV` | `production` | Environment: production, development, test |
| `ENGRAM_LOG_LEVEL` | `info` | Python log level |
| `ENGRAM_SERVER_HOST` | `0.0.0.0` | Server bind address |
| `ENGRAM_SERVER_PORT` | `8100` | Server port |
| `ENGRAM_FALKORDB_URL` | `redis://falkordb:6379` | FalkorDB connection string |
| `ENGRAM_FALKORDB_PASSWORD` | `changeme` | FalkorDB auth password |
| `ENGRAM_FALKORDB_PORT` | `6380` | FalkorDB host-exposed port |
| `ENGRAM_REDIS_URL` | `redis://redis:6379` | Redis connection string |
| `ENGRAM_REDIS_PASSWORD` | `changeme` | Redis auth password |
| `ENGRAM_REDIS_PORT` | `6381` | Redis host-exposed port |
| `ENGRAM_AI_PROVIDER` | `anthropic` | AI provider for extraction |
| `ENGRAM_ANTHROPIC_API_KEY` | *(required)* | Anthropic API key for entity extraction |
| `ENGRAM_ACTIVATION_DECAY_RATE` | `0.05` | Activation decay rate |
| `ENGRAM_ACTIVATION_SPREAD_FACTOR` | `0.3` | Spreading activation factor |
| `ENGRAM_ACTIVATION_SPREAD_HOPS` | `2` | Max hops for spreading activation |
| `ENGRAM_AUTH_ENABLED` | `false` | Enable bearer token authentication |
| `ENGRAM_AUTH_SECRET` | *(empty)* | Self-hosted bearer token (see 05_security_model.md) |
| `ENGRAM_MASTER_KEY` | *(empty)* | 32-byte hex key for PII encryption (AES-256-GCM). Optional, only when `encryption.enabled=true` |
| `ENGRAM_JWT_SECRET` | *(empty)* | JWT signing key (SaaS mode only) |
| `ENGRAM_DASHBOARD_PORT` | `3000` | Dashboard host-exposed port |
| `ENGRAM_DASHBOARD_API_URL` | `http://localhost:8100` | API URL for dashboard build |
| `ENGRAM_DASHBOARD_WS_URL` | `ws://localhost:8100/ws` | WebSocket URL for dashboard |
| `ENGRAM_BACKUP_RETAIN` | `10` | Number of backups to keep |

Every variable that holds a secret also supports the `_FILE` suffix convention for Docker Swarm / Kubernetes secret mounts. For example, setting `ENGRAM_AUTH_SECRET_FILE=/run/secrets/engram_auth_secret` causes the config loader to read the secret from that file instead of the environment variable value. See section 10.4 and 05_security_model.md section 11.3 for details.

### `.env.example`

```env
# .env.example -- Copy to .env and fill in values

# Required -- Claude API key for entity extraction
ENGRAM_ANTHROPIC_API_KEY=sk-ant-your-key-here

# Data store passwords (change from defaults in production)
ENGRAM_FALKORDB_PASSWORD=changeme
ENGRAM_REDIS_PASSWORD=changeme

# Auth -- disabled by default for local dev
# ENGRAM_AUTH_ENABLED=false
# ENGRAM_AUTH_SECRET=      # generate: openssl rand -hex 32
# ENGRAM_MASTER_KEY=       # PII encryption (optional): openssl rand -hex 32
# ENGRAM_JWT_SECRET=       # SaaS mode only: openssl rand -hex 32

# Optional overrides
# ENGRAM_SERVER_PORT=8100
# ENGRAM_DASHBOARD_PORT=3000
# ENGRAM_LOG_LEVEL=info
```

**Security note:** The `.env` file must be listed in `.gitignore` and `.dockerignore`. Only `.env.example` is committed. See 05_security_model.md for the full secrets management strategy.

### `.gitignore` entries (security-critical)

```gitignore
.env
.env.*
!.env.example
config.yaml
config.local.yaml
*.pem
*.key
```

---

## 8. Resource Sizing Guide

### 8.1 Personal Use (Recommended Minimum)

Target: single user, up to 10K entities, 50K episodes.

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU      | 2 cores | 4 cores     |
| RAM      | 2 GB    | 4 GB        |
| Disk     | 2 GB    | 10 GB       |
| Network  | N/A     | N/A (local) |

**Per-container breakdown (recommended):**

| Container | CPU   | RAM    | Disk       |
|-----------|-------|--------|------------|
| FalkorDB  | 1.0   | 1 GB   | 2-5 GB     |
| Redis     | 0.5   | 512 MB | 500 MB     |
| Server    | 1.0   | 1 GB   | 200 MB     |
| Dashboard | 0.25  | 128 MB | 50 MB      |
| **Total** | **2.75** | **~2.6 GB** | **~6 GB** |

### 8.2 Small Team / Power User

Target: 1-5 users, up to 100K entities, 500K episodes.

| Resource | Recommended |
|----------|-------------|
| CPU      | 4-8 cores   |
| RAM      | 8-16 GB     |
| Disk     | 50 GB SSD   |

### 8.3 Scaling Notes

- **FalkorDB** is the primary memory consumer. Each node with properties uses roughly 200-500 bytes. At 100K nodes: ~50 MB in graph memory. FalkorDB handles this easily.
- **Redis activation state** is lightweight. Each activation key is a small hash (~150 bytes). At 100K keys: ~15 MB.
- **Embeddings** are the largest data type. If stored in FalkorDB vector index, 384-dim float32 vectors at 100K entities: ~150 MB.
- **Disk growth** is dominated by AOF/RDB files. AOF rewrite keeps this bounded.
- For true production multi-tenant (hosted SaaS), split FalkorDB and Redis to dedicated hosts. This compose setup targets personal/small-team use.

---

## 9. Monitoring & Observability

### 9.1 Structured Logging

The server uses structured JSON logging in production:

```python
# server/engram/config.py (logging section)
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)
```

### 9.2 Key Metrics to Expose

The `/health` endpoint (section 4) covers liveness. For deeper observability, the server should expose a `/metrics` endpoint or structured log lines for:

| Metric | Type | Description |
|--------|------|-------------|
| `engram_episodes_ingested_total` | counter | Total episodes processed |
| `engram_retrieval_latency_ms` | histogram | Recall query latency |
| `engram_activation_recovery_count` | gauge | Nodes recovered on last startup |
| `engram_entities_total` | gauge | Current entity count |
| `engram_redis_keys_total` | gauge | Activation keys in Redis |
| `engram_extraction_latency_ms` | histogram | Claude API extraction time |
| `engram_extraction_errors_total` | counter | Failed extractions |

These can be consumed by Prometheus/Grafana if users want dashboards beyond the built-in one. For v1, structured log lines are sufficient.

---

## 10. Security Considerations for Docker

### 10.1 Container Hardening

1. **No root in containers.** Production Dockerfiles should use a non-root user:
   ```dockerfile
   RUN addgroup -S engram && adduser -S engram -G engram
   USER engram
   ```

2. **Read-only filesystem where possible.** The dashboard container serves static files and can run with `read_only: true` in compose.

3. **No host networking.** All containers communicate through the `engram-net` bridge network. Only mapped ports are exposed to the host.

4. **Image pinning.** All base images use specific version tags (not `latest`) to prevent supply chain drift.

5. **`.dockerignore` files** in both `server/` and `dashboard/` to exclude `.env`, `.git`, `node_modules`, `__pycache__`, and test fixtures from build context.

### 10.2 Secrets Hierarchy

| Deployment | Secret Storage | Method |
|------------|---------------|--------|
| Local dev  | `.env` file (gitignored) | Docker Compose `env_file` |
| Self-hosted production | `.env` file with restricted permissions (`chmod 600`) | Docker Compose `env_file` |
| Docker Swarm | Docker secrets | `_FILE` suffix convention (see 10.4) |
| Kubernetes | K8s secrets | Mounted as files, read via `_FILE` suffix |
| Hosted SaaS | Cloud secrets manager (e.g., AWS Secrets Manager) | Injected at runtime |

### 10.3 Secret Variables

These variables hold sensitive material and must never be logged or exposed in health endpoints:

- `ENGRAM_ANTHROPIC_API_KEY` -- Claude API key
- `ENGRAM_AUTH_SECRET` -- self-hosted bearer token
- `ENGRAM_MASTER_KEY` -- 32-byte hex AES-256-GCM key for PII field encryption (optional, requires `encryption.enabled=true`)
- `ENGRAM_JWT_SECRET` -- JWT signing key (SaaS mode)
- `ENGRAM_FALKORDB_PASSWORD` -- FalkorDB auth
- `ENGRAM_REDIS_PASSWORD` -- Redis auth

### 10.4 `_FILE` Suffix Convention for Secret Mounts

For production deployments using Docker Swarm or Kubernetes, the config loader supports reading secrets from files instead of environment variables. For any secret variable `ENGRAM_X`, setting `ENGRAM_X_FILE=/run/secrets/engram_x` causes the loader to read the file contents as the value.

**Docker Swarm example:**

```yaml
# docker-compose.swarm.yml (overlay on top of docker-compose.yml)
services:
  server:
    environment:
      - ENGRAM_AUTH_SECRET_FILE=/run/secrets/engram_auth_secret
      - ENGRAM_ANTHROPIC_API_KEY_FILE=/run/secrets/engram_anthropic_api_key
      - ENGRAM_MASTER_KEY_FILE=/run/secrets/engram_master_key
      - ENGRAM_FALKORDB_PASSWORD_FILE=/run/secrets/engram_falkordb_password
      - ENGRAM_REDIS_PASSWORD_FILE=/run/secrets/engram_redis_password
    secrets:
      - engram_auth_secret
      - engram_anthropic_api_key
      - engram_master_key
      - engram_falkordb_password
      - engram_redis_password

secrets:
  engram_auth_secret:
    external: true
  engram_anthropic_api_key:
    external: true
  engram_master_key:
    external: true
  engram_falkordb_password:
    external: true
  engram_redis_password:
    external: true
```

**Config loader implementation** (coordinate with config-agent's Pydantic schema):

```python
import os
from pathlib import Path

def resolve_secret(env_var: str) -> str | None:
    """Read secret from _FILE path if set, otherwise from env var directly."""
    file_path = os.environ.get(f"{env_var}_FILE")
    if file_path:
        return Path(file_path).read_text().strip()
    return os.environ.get(env_var)
```

See 05_security_model.md section 11.3 for the full Docker Swarm secrets configuration.

---

## 11. Coordination Notes

### Config Agent

Environment variable conventions established here:
- All Engram-specific vars use `ENGRAM_` prefix (including `ENGRAM_ANTHROPIC_API_KEY`)
- Connection strings use `_URL` suffix
- Passwords use `_PASSWORD` suffix
- Ports use `_PORT` suffix
- Boolean flags use `true`/`false` strings
- Secret vars support `_FILE` suffix for Docker Swarm / K8s mounts (see 10.4)

The Config agent's Pydantic settings schema should read from these env vars with the same defaults listed in section 7, and implement the `resolve_secret()` pattern from section 10.4 for all secret fields.

### Auth Agent (Resolved)

Both documents now use `ENGRAM_AUTH_SECRET` as the canonical env var name per 05_security_model.md section 16.1:
- `ENGRAM_AUTH_ENABLED` (`false` default) -- toggle for bearer auth. When disabled, TenantContextMiddleware still produces a `TenantContext` with `group_id="default"` so all queries remain tenant-scoped.
- `ENGRAM_AUTH_SECRET` -- self-hosted bearer token (canonical name, confirmed in both docs)
- `ENGRAM_MASTER_KEY` -- 32-byte hex AES-256-GCM key for PII field-level encryption at rest. Optional; only required when `encryption.enabled=true` in config. Not needed for v1 dev default.
- `ENGRAM_JWT_SECRET` -- JWT signing key for SaaS mode (empty/unused in self-hosted)
- `ENGRAM_ANTHROPIC_API_KEY` -- uses `ENGRAM_` prefix (not bare `ANTHROPIC_API_KEY`)
- All secret vars support `_FILE` suffix for production secret mounts (section 10.4)
- `.gitignore` entries aligned with auth agent's requirements (section 7)
- `ENGRAM_FALKORDB_PASSWORD` and `ENGRAM_REDIS_PASSWORD` must differ in production
- Production secrets migration: Tier 1 `.env` (v1) -> Tier 2 Docker Swarm `_FILE` (Week 7) -> Tier 3 pluggable `SecretProvider` for Vault/AWS SM/GCP SM (Month 3 SaaS). See 05_security_model.md section 16.
