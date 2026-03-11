# Contributing to Engram

Thank you for your interest in contributing to Engram. This guide will help you get started quickly.

## Getting Started

1. **Fork and clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/Engram.git
   cd Engram
   ```

2. **Install backend dependencies:**

   ```bash
   cd server && uv sync
   ```

3. **Install frontend dependencies:**

   ```bash
   cd dashboard && pnpm install
   ```

4. **Run the test suites to confirm everything works:**

   ```bash
   # Backend (lite mode, no Docker required)
   cd server && uv run pytest -m "not requires_docker" -v

   # Frontend
   cd dashboard && pnpm test
   ```

## Development Setup

Engram runs in **lite mode** by default, which uses SQLite and requires no external services. This is the recommended setup for local development.

```bash
# Start the REST API server
cd server && uv run engram serve

# Start the MCP server (stdio transport)
cd server && uv run engram mcp

# Run a one-shot consolidation cycle
cd server && uv run python -m engram.consolidation --profile observe
```

For the full stack with FalkorDB and Redis, use Docker:

```bash
docker compose up -d --build
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with clear, focused commits.
3. Ensure all tests pass and the linter is clean (see below).
4. Submit a pull request against `main` with a clear description of what changed and why.
5. A maintainer will review your PR. Please be responsive to feedback.

Keep PRs focused on a single concern. If you are fixing a bug and also want to refactor nearby code, consider splitting those into separate PRs.

## Code Style

**Python:** We use [ruff](https://docs.astral.sh/ruff/) for both formatting and linting. Before submitting, run:

```bash
cd server && uv run ruff check .
```

**TypeScript:** We use ESLint. The dashboard project has its own lint configuration:

```bash
cd dashboard && pnpm lint
```

## Testing

All new features and bug fixes should include tests.

- **Backend:** pytest with pytest-asyncio. Place test files in `server/tests/`.
- **Frontend:** vitest with React Testing Library. Place test files alongside components or in dedicated test directories.

For backend tests, use the lite-mode marker to skip tests that require Docker services:

```bash
cd server && uv run pytest -m "not requires_docker" -v
```

To run a specific test file or test:

```bash
cd server && uv run pytest server/tests/test_example.py -v
cd server && uv run pytest server/tests/test_example.py::test_specific_case -v
```

## Architecture Overview

Engram is a persistent memory layer for AI agents built around temporal knowledge graphs and cognitive-science-inspired retrieval.

Key architectural points:

- **Dual mode:** SQLite for local/lite usage, FalkorDB + Redis for production deployments. The storage layer auto-detects which backend is available.
- **CQRS pattern:** Episode ingestion splits into a fast path (`store_episode`, no LLM) and a slow path (`project_episode`, runs extraction and embedding).
- **12-phase consolidation:** triage, merge, infer, replay, prune, compact, mature, semanticize, schema, reindex, graph_embed, dream. Inspired by biological memory consolidation.
- **MCP server:** stdio transport with 15 tools, 3 resources, and 2 prompts.
- **ACT-R activation:** Cognitive-architecture-based retrieval with lazy activation computation.

For full architectural details, see `CLAUDE.md` in the project root.

## Reporting Issues

Please use [GitHub Issues](https://github.com/anthropics/Engram/issues) to report bugs or request features. When reporting a bug, include:

- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (OS, Python version, lite vs. full mode)
- Relevant log output or error messages

## License

By contributing to Engram, you agree that your contributions will be licensed under the same license as the project.
