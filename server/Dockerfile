# Multistage build for Engram server

# --- Builder ---
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock* ./

# Install dependencies (no dev deps, include full extras)
RUN uv sync --frozen --no-dev --extra full || uv sync --no-dev --extra full

# Copy source code
COPY . .

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
    chown -R engram:engram /app
USER engram

# Ensure venv is on PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8100

CMD ["uvicorn", "engram.main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "2"]
