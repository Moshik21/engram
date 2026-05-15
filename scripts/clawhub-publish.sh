#!/usr/bin/env bash
set -euo pipefail

# Publish engram-brain skill to ClawHub
#
# Prerequisites:
#   npm install -g clawhub   (or: npx clawhub ...)
#   clawhub login            (browser-based auth)
#
# Usage:
#   ./scripts/clawhub-publish.sh              # publish current version
#   ./scripts/clawhub-publish.sh --dry-run    # preview without publishing

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SKILL_DIR="$REPO_ROOT/skills/engram-brain"
SKILL_FILE="$SKILL_DIR/SKILL.md"

# Extract version from SKILL.md frontmatter
VERSION=$(grep '^version:' "$SKILL_FILE" | head -1 | awk '{print $2}')

if [ -z "$VERSION" ]; then
    echo "Error: Could not extract version from $SKILL_FILE"
    exit 1
fi

echo "Publishing engram-brain v${VERSION} to ClawHub..."
echo "Skill directory: $SKILL_DIR"
echo ""

# Check clawhub CLI is available
if ! command -v clawhub &>/dev/null; then
    echo "Error: clawhub CLI not found."
    echo "Install with: npm install -g clawhub"
    echo "Then run: clawhub login"
    exit 1
fi

# Check auth
if ! clawhub whoami &>/dev/null; then
    echo "Error: Not authenticated. Run: clawhub login"
    exit 1
fi

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "[DRY RUN] Scanning skill for upload..."
    clawhub sync --root "$REPO_ROOT/skills" --dry-run
else
    clawhub publish "$SKILL_DIR" \
        --slug engram-brain \
        --name "Engram Memory" \
        --version "$VERSION" \
        --tags "memory,knowledge-graph,mcp,recall,long-term-memory,cognitive-architecture"
    echo ""
    echo "Done. Skill available at: https://clawhub.ai/skills/engram-brain"
fi
