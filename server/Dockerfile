# Multistage build for Engram server
# Build context must be the repo root (context: . in docker-compose.yml)

# --- Builder ---
FROM python:3.12-slim AS builder

# Use /app as WORKDIR so venv paths match runtime
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy README to parent so ../README.md resolves from /app
COPY README.md /README.md

# Copy dependency files and source (hatchling needs source to build)
COPY server/pyproject.toml .
COPY server/uv.lock* ./
COPY server/engram ./engram

# Install dependencies (no dev deps, include full extras)
RUN uv sync --frozen --no-dev --extra full --extra local || uv sync --no-dev --extra full --extra local

# --- Runtime ---
FROM python:3.12-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/engram /app/engram

# Non-root user
RUN groupadd --gid 1001 engram && \
    useradd --uid 1001 --gid engram --create-home engram && \
    mkdir -p /home/engram/.engram && \
    chown -R engram:engram /app /home/engram/.engram
USER engram

# Ensure venv is on PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8100

# Engram keeps schedulers, workers, and live event fan-out in-process.
# Running multiple Uvicorn workers duplicates that state and fragments events.
CMD ["uvicorn", "engram.main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
