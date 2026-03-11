#!/usr/bin/env bash
# Engram public installer — lightweight bootstrap that installs engramctl,
# then runs `engramctl setup` for interactive mode selection (lite or full).
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "  ${GREEN}*${RESET} $1"; }
warn()  { echo -e "  ${RED}!${RESET} $1"; }
note()  { echo -e "  ${CYAN}>${RESET} $1"; }
dim()   { echo -e "  ${DIM}$1${RESET}"; }
die()   { warn "$1"; exit 1; }

MODE="${1:-}"
BIN_DIR="${ENGRAM_INSTALL_BIN_DIR:-$HOME/.local/bin}"
ENGRAM_HOME="${ENGRAM_HOME:-$HOME/.engram}"
RELEASE_REPOSITORY="${ENGRAM_RELEASE_REPOSITORY:-engram-labs/engram}"

ensure_supported_os() {
  case "$(uname -s)" in
    Darwin|Linux) ;;
    *) die "Engram currently supports macOS and Linux only." ;;
  esac
}

ensure_tools() {
  for cmd in curl tar; do
    command -v "$cmd" >/dev/null 2>&1 || die "Missing required command: $cmd"
  done
}

ensure_python() {
  if ! command -v python3 &>/dev/null; then
    warn "Python 3.10+ is required but not found."
    case "$(uname -s)" in
      Darwin)
        note "Install with Homebrew: brew install python@3.12"
        note "Or download from: https://www.python.org/downloads/"
        ;;
      Linux)
        note "Install with your package manager:"
        dim "  Ubuntu/Debian: sudo apt install python3"
        dim "  Fedora:        sudo dnf install python3"
        dim "  Arch:          sudo pacman -S python"
        ;;
    esac
    exit 1
  fi

  local py_version py_major py_minor
  py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  py_major=$(echo "$py_version" | cut -d. -f1)
  py_minor=$(echo "$py_version" | cut -d. -f2)
  if [ "$py_major" -lt 3 ] || ([ "$py_major" -eq 3 ] && [ "$py_minor" -lt 10 ]); then
    die "Python 3.10+ required, found $py_version"
  fi
  info "Python $py_version"
}

ensure_uv() {
  if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv &>/dev/null || die "uv installation failed."
  fi
  info "uv $(uv --version 2>/dev/null | head -1)"
}

install_engram_package() {
  info "Installing Engram..."

  # Allow override for testing (install from local checkout)
  if [ -n "${ENGRAM_PACKAGE_SOURCE:-}" ]; then
    uv tool install "$ENGRAM_PACKAGE_SOURCE" \
      || die "Failed to install engram from $ENGRAM_PACKAGE_SOURCE"
    info "Installed engram from $ENGRAM_PACKAGE_SOURCE"
    return
  fi

  # Try PyPI first, fall back to git
  if uv tool install "engram[local]" 2>/dev/null; then
    info "Installed engram from PyPI"
  elif uv tool install "git+https://github.com/${RELEASE_REPOSITORY}.git#subdirectory=server[local]" 2>/dev/null; then
    info "Installed engram from GitHub"
  else
    die "Failed to install engram. Check your network connection and try again."
  fi
}

install_engramctl() {
  mkdir -p "$BIN_DIR"

  local engramctl_path="$BIN_DIR/engramctl"

  # Try sources in order: release bundle asset → local repo copy → raw main (last resort)
  local bundle_url="https://github.com/${RELEASE_REPOSITORY}/releases/latest/download/engram-install-bundle.tar.gz"
  local temp_dir
  temp_dir="$(mktemp -d)"
  local fetched=0

  # 1. Extract engramctl from release bundle (pinned to the same release as the package)
  if curl -fsSL "$bundle_url" -o "$temp_dir/bundle.tar.gz" 2>/dev/null; then
    if tar -xzf "$temp_dir/bundle.tar.gz" -C "$temp_dir" engramctl 2>/dev/null \
       || tar -xzf "$temp_dir/bundle.tar.gz" -C "$temp_dir" ./engramctl 2>/dev/null; then
      cp "$temp_dir/engramctl" "$engramctl_path"
      chmod +x "$engramctl_path"
      info "Installed engramctl to $engramctl_path (from release bundle)"
      fetched=1
    fi
  fi

  # 2. Fallback: check if engramctl is in the same directory as this script (local repo)
  if [ "$fetched" = "0" ]; then
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$script_dir/../installer/engramctl" ]; then
      cp "$script_dir/../installer/engramctl" "$engramctl_path"
      chmod +x "$engramctl_path"
      info "Installed engramctl to $engramctl_path (from local repo)"
      fetched=1
    fi
  fi

  # 3. Last resort: fetch from raw main (unpinned, but better than failing)
  if [ "$fetched" = "0" ]; then
    local raw_url="https://raw.githubusercontent.com/${RELEASE_REPOSITORY}/main/installer/engramctl"
    if curl -fsSL "$raw_url" -o "$engramctl_path" 2>/dev/null; then
      chmod +x "$engramctl_path"
      warn "Installed engramctl from main branch (could not fetch release bundle)"
    else
      die "Could not download engramctl."
    fi
  fi

  rm -rf "$temp_dir"

  if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    note "Add $BIN_DIR to your PATH:"
    dim "  export PATH=\"$BIN_DIR:\$PATH\""
    export PATH="$BIN_DIR:$PATH"
  fi
}

main() {
  echo ""
  echo -e "  ${BOLD}Engram Installer${RESET}"
  echo -e "  ${DIM}Persistent memory layer for AI agents${RESET}"
  echo ""

  ensure_supported_os
  ensure_tools

  # For explicit full/openclaw mode, take the Docker fast path
  if [ "$MODE" = "full" ] || [ "$MODE" = "openclaw" ]; then
    _install_full_mode
    return
  fi

  # Lite-capable bootstrap: Python + uv + engram package + engramctl
  ensure_python
  ensure_uv
  install_engram_package
  install_engramctl

  # Create home directory
  mkdir -p "$ENGRAM_HOME"

  # Run setup (interactive mode picker or pre-selected)
  if [ "$MODE" = "lite" ]; then
    "$BIN_DIR/engramctl" setup --mode lite
  else
    "$BIN_DIR/engramctl" setup
  fi

  echo ""
  echo -e "  ${BOLD}${GREEN}Install complete!${RESET}"
  dim "  Command: $BIN_DIR/engramctl"
  echo ""
}

_install_full_mode() {
  # Legacy Docker-first path for explicit full/openclaw mode
  local bundle_url="${ENGRAM_BUNDLE_URL:-https://github.com/${RELEASE_REPOSITORY}/releases/latest/download/engram-install-bundle.tar.gz}"
  local sha_url="${ENGRAM_BUNDLE_SHA256_URL:-https://github.com/${RELEASE_REPOSITORY}/releases/latest/download/engram-install-bundle.sha256}"

  # Docker checks
  if ! command -v docker >/dev/null 2>&1; then
    warn "Docker is required for full mode."
    case "$(uname -s)" in
      Darwin)
        note "Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
        ;;
      Linux)
        note "Install Docker Engine: https://docs.docker.com/engine/install/"
        ;;
    esac
    exit 1
  fi
  docker info >/dev/null 2>&1 || die "Docker is installed but the daemon is not running."

  local temp_dir bundle_path sha_path extract_dir
  temp_dir="$(mktemp -d)"
  trap "rm -rf '$temp_dir'" EXIT
  bundle_path="$temp_dir/engram-install-bundle.tar.gz"
  sha_path="$temp_dir/engram-install-bundle.sha256"
  extract_dir="$temp_dir/bundle"

  info "Downloading install bundle..."
  curl -fsSL "$bundle_url" -o "$bundle_path"

  # Verify checksum
  if curl -fsSL "$sha_url" -o "$sha_path" 2>/dev/null; then
    local expected actual
    expected="$(awk '{print $1}' "$sha_path")"
    if command -v sha256sum >/dev/null 2>&1; then
      actual="$(sha256sum "$bundle_path" | awk '{print $1}')"
    else
      actual="$(shasum -a 256 "$bundle_path" | awk '{print $1}')"
    fi
    [ "$expected" = "$actual" ] || die "Checksum mismatch."
  else
    warn "Checksum asset not found. Continuing without verification."
  fi

  mkdir -p "$extract_dir"
  tar -xzf "$bundle_path" -C "$extract_dir"

  info "Installing Engram (${MODE})..."
  ENGRAM_INSTALL_ROOT="${ENGRAM_INSTALL_ROOT:-$HOME/.engram/full}" \
  ENGRAM_INSTALL_BIN_DIR="$BIN_DIR" \
  ENGRAM_INSTALL_SKIP_OPEN="${ENGRAM_INSTALL_SKIP_OPEN:-0}" \
  ENGRAM_INSTALL_NONINTERACTIVE="${ENGRAM_INSTALL_NONINTERACTIVE:-0}" \
  ENGRAM_INSTALL_SKIP_RUNTIME="${ENGRAM_INSTALL_SKIP_RUNTIME:-0}" \
    bash "$extract_dir/engramctl" install --mode "$MODE"

  echo ""
  echo -e "  ${BOLD}${GREEN}Install complete!${RESET}"
  dim "  Command: $BIN_DIR/engramctl"
  echo ""
}

main "$@"
