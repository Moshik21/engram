# CI continuity job (manual apply)

GitHub rejected pushing `.github/workflows/ci.yml` with a token that lacks the
`workflow` scope. Apply this job yourself (or push with a PAT that includes
`workflow`):

```yaml
  continuity:
    name: Continuity golden path
    runs-on: ubuntu-latest
    needs: lint
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
          python-version: "3.12"
      - name: Install dependencies
        run: uv sync --frozen --dev
      - name: Continuity suite (product metric, not LongMemEval)
        run: |
          uv run pytest tests/test_continuity_golden_path.py tests/test_durable_entity_rescue.py tests/test_durable_context_pack.py tests/test_promotion_policy.py tests/test_remember_committed_facts.py tests/test_session_promote_nudge_hook.py -v --tb=short
          uv run engram continuity --smoke
        env:
          ENGRAM_MODE: lite
```

The same `continuity:` job block is already staged in the **local** working
tree at `.github/workflows/ci.yml`. Push it with a PAT that includes the
`workflow` scope (fine-grained: "Workflows" write, or classic `workflow`).

Local gate (works now without workflow push):

```bash
cd server
uv run engram continuity --smoke
uv run engram doctor --skip-server --no-lifecycle --no-smoke --require-golden-loop
# Product surface (agents):
export ENGRAM_MCP_SURFACE=public
# Operator/debug:
export ENGRAM_MCP_SURFACE=full
```

Native Helix CI job (`native-continuity`) **probes** `import helix_native`:

- If native is **unavailable** → job records SKIP (exit 0 with notice), not a
  green-wash of a failed native path.
- If native is **available** → continuity smoke + `doctor --require-golden-loop`
  are a **hard gate** (no `continue-on-error`).

Helix integration tests prefer native data-dir via
`engram.storage.helix.availability` (HTTP :6969 is secondary).
