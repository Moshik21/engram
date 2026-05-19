#!/usr/bin/env bash
# Engram public installer — lightweight bootstrap that installs engramctl,
# then runs `engramctl quickstart` for native-first setup and readiness checks.
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
RELEASE_REPOSITORY="${ENGRAM_RELEASE_REPOSITORY:-Moshik21/engram}"
HELIX_NATIVE_VERSION="${ENGRAM_HELIX_NATIVE_VERSION:-0.1.0}"
HELIX_NATIVE_SUBDIR="native/helix-repo/helix-python"

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

is_native_install_mode() {
  [ "${MODE:-helix}" = "helix" ] || [ "${MODE:-}" = "auto" ] || [ "${MODE:-}" = "openclaw" ]
}

discover_helix_native_release_wheel() {
  [ "${ENGRAM_HELIX_NATIVE_SKIP_RELEASE_WHEEL:-0}" = "1" ] && return 1
  command -v python3 >/dev/null 2>&1 || return 1

  local metadata_file api_url
  metadata_file="$(mktemp)"
  api_url="https://api.github.com/repos/${RELEASE_REPOSITORY}/releases/latest"
  if ! curl -fsSL "$api_url" -o "$metadata_file" 2>/dev/null; then
    rm -f "$metadata_file"
    return 1
  fi

  local wheel_url
  wheel_url="$(
    python3 - "$metadata_file" "$HELIX_NATIVE_VERSION" <<'PY'
import json
import platform
import sys

metadata_path, expected_version = sys.argv[1], sys.argv[2]
try:
    assets = json.load(open(metadata_path, encoding="utf-8")).get("assets", [])
except Exception:
    sys.exit(1)

system = platform.system()
machine = platform.machine().lower()
py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

if machine in {"arm64", "aarch64"}:
    arch = "arm64" if system == "Darwin" else "aarch64"
elif machine in {"x86_64", "amd64"}:
    arch = "x86_64"
else:
    sys.exit(1)

def score(asset):
    name = asset.get("name", "").lower()
    if not name.endswith(".whl") or not name.startswith("helix_native-"):
        return -1
    if expected_version and f"-{expected_version}-" not in name:
        return -1
    if "cp310-abi3" not in name and py_tag not in name:
        return -1

    score_value = 0
    if system == "Darwin":
        if "macosx" not in name:
            return -1
        if "universal2" in name:
            score_value += 30
        elif arch not in name:
            return -1
    elif system == "Linux":
        if "manylinux" not in name and "linux" not in name:
            return -1
        if arch not in name:
            return -1
        if "manylinux" in name:
            score_value += 20
    else:
        return -1

    if "cp310-abi3" in name:
        score_value += 10
    return score_value

matches = sorted(
    ((score(asset), asset.get("browser_download_url", "")) for asset in assets),
    reverse=True,
)
for score_value, url in matches:
    if score_value >= 0 and url:
        print(url)
        sys.exit(0)
sys.exit(1)
PY
  )" || true
  rm -f "$metadata_file"

  [ -n "$wheel_url" ] || return 1
  printf '%s' "$wheel_url"
}

helix_native_git_requirement() {
  local repo_url="https://github.com/${RELEASE_REPOSITORY}.git"
  if [ -n "${ENGRAM_RELEASE_REF:-}" ]; then
    repo_url="${repo_url}@${ENGRAM_RELEASE_REF}"
  fi
  printf 'helix-native @ git+%s#subdirectory=%s' "$repo_url" "$HELIX_NATIVE_SUBDIR"
}

local_helix_native_requirement() {
  local script_dir repo_root candidate
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "$script_dir/.." >/dev/null 2>&1 && pwd || true)"
  [ -n "$repo_root" ] || return 1

  candidate="$repo_root/$HELIX_NATIVE_SUBDIR"
  if [ -f "$candidate/pyproject.toml" ] && [ -d "$candidate/../helix-db" ]; then
    printf 'helix-native @ file://%s' "$candidate"
    return
  fi

  return 1
}

resolve_helix_native_requirement() {
  if [ -n "${ENGRAM_HELIX_NATIVE_SOURCE:-}" ]; then
    printf '%s' "$ENGRAM_HELIX_NATIVE_SOURCE"
    return
  fi

  local wheel_url
  if wheel_url="$(discover_helix_native_release_wheel)"; then
    printf '%s' "$wheel_url"
    return
  fi

  if local_helix_native_requirement; then
    return
  fi

  helix_native_git_requirement
}

uv_tool_install_engram() {
  local package_spec="$1"
  local source_label="$2"

  if is_native_install_mode; then
    local helix_native_req
    helix_native_req="$(resolve_helix_native_requirement)"
    if [[ "$helix_native_req" == helix-native\ @\ file://* ]]; then
      note "Using local Helix native source; build can take several minutes and requires Rust/Cargo."
    elif [[ "$helix_native_req" == helix-native\ @\ git+* ]]; then
      note "No compatible Helix native release wheel was found; building from Engram's bundled source."
      note "This can take several minutes and requires Rust/Cargo."
    fi
    if [[ "$helix_native_req" == helix-native\ @\ file://* || "$helix_native_req" == helix-native\ @\ git+* ]]; then
      if ! command -v cargo >/dev/null 2>&1 || ! cargo --version >/dev/null 2>&1; then
        warn "Helix native source build requires a working Rust toolchain."
        note "Install Rust with rustup and run: rustup default stable"
        return 1
      fi
    fi
    info "Using Helix native runtime: $helix_native_req"

    uv tool install --with "$helix_native_req" "$package_spec"
    return
  fi

  uv tool install "$package_spec"
}

install_engram_package() {
  if [ "${ENGRAM_INSTALL_SKIP_PACKAGE:-0}" = "1" ]; then
    warn "Skipping Engram package install because ENGRAM_INSTALL_SKIP_PACKAGE=1."
    return
  fi

  info "Installing Engram..."

  local package_spec="engram[local]"
  local github_spec="git+https://github.com/${RELEASE_REPOSITORY}.git#subdirectory=server[local]"
  if is_native_install_mode; then
    package_spec="engram[local,native]"
    github_spec="git+https://github.com/${RELEASE_REPOSITORY}.git#subdirectory=server[local,native]"
  fi

  # Allow override for testing (install from local checkout)
  if [ -n "${ENGRAM_PACKAGE_SOURCE:-}" ]; then
    uv_tool_install_engram "$ENGRAM_PACKAGE_SOURCE" "$ENGRAM_PACKAGE_SOURCE" \
      || die "Failed to install engram from $ENGRAM_PACKAGE_SOURCE"
    info "Installed engram from $ENGRAM_PACKAGE_SOURCE"
    return
  fi

  # Prefer the GitHub source until the PyPI package is promoted from placeholder.
  if uv_tool_install_engram "$github_spec" "GitHub" 2>/dev/null; then
    info "Installed engram from GitHub"
  elif uv_tool_install_engram "$package_spec" "PyPI" 2>/dev/null; then
    info "Installed engram from PyPI"
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

  case "$MODE" in
    ""|helix|lite|auto|full|openclaw) ;;
    *) die "Unknown install mode: $MODE. Use helix, lite, auto, full, or openclaw." ;;
  esac

  # Docker is explicit. OpenClaw mode is native Helix plus skill/MCP setup.
  if [ "$MODE" = "full" ]; then
    _install_full_mode
    return
  fi

  # Local bootstrap: Python + uv + engram package + engramctl
  ensure_python
  ensure_uv
  install_engram_package
  install_engramctl

  # Create home directory
  mkdir -p "$ENGRAM_HOME"

  # Run quickstart (native-first default, pre-selected local backend, or OpenClaw setup)
  if [ "$MODE" = "openclaw" ]; then
    if [ "${ENGRAM_INSTALL_SKIP_RUNTIME:-0}" = "1" ]; then
      "$BIN_DIR/engramctl" quickstart --mode helix --install-openclaw --connect openclaw --no-start --no-doctor
    else
      "$BIN_DIR/engramctl" quickstart --mode helix --install-openclaw --connect openclaw
    fi
  elif [ -n "$MODE" ]; then
    "$BIN_DIR/engramctl" quickstart --mode "$MODE"
  else
    "$BIN_DIR/engramctl" quickstart
  fi

  echo ""
  echo -e "  ${BOLD}${GREEN}Install complete!${RESET}"
  dim "  Command: $BIN_DIR/engramctl"
  echo ""
}

_install_full_mode() {
  # Legacy Docker-backed path for explicit full mode only.
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
