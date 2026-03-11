"""Interactive setup wizard for Engram. Run: python -m engram setup"""

import getpass
import json
import secrets
import textwrap
from datetime import datetime
from pathlib import Path

# ANSI helpers (no deps)
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"


def _bold(t: str) -> str:
    return f"{_BOLD}{t}{_RESET}"


def _dim(t: str) -> str:
    return f"{_DIM}{t}{_RESET}"


def _green(t: str) -> str:
    return f"{_GREEN}{t}{_RESET}"


def _cyan(t: str) -> str:
    return f"{_CYAN}{t}{_RESET}"


def _yellow(t: str) -> str:
    return f"{_YELLOW}{t}{_RESET}"


def _red(t: str) -> str:
    return f"{_RED}{t}{_RESET}"


_M, _C, _R = _MAGENTA, _CYAN, _RESET
_BRAIN = f"""
    {_M}⣀⣤⣶⣿⣿⣿⣶⣤⣀{_R}
  {_M}⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣴{_R}
 {_M}⣾⣿⣿⣿⣿{_R}⠿⠿⠿⠿{_M}⣿⣿⣿⣿⣾{_R}
{_M}⣿⣿⣿{_R}⠋⠁  {_C}memory{_R}  ⠁⠋{_M}⣿⣿⣿{_R}
{_M}⣿⣿⣿⣆{_R}⡀  {_C}layer{_R}   ⡀{_M}⣆⣿⣿⣿{_R}
 {_M}⠻⣿⣿⣿⣿⣶⣶⣶⣶⣿⣿⣿⣿⠟{_R}
  ⠈{_M}⠻⣿⣿⣿⣿⣿⣿⣿⣿⠟{_R}⠁
     ⠉⠛⠿⠿⠿⠛⠉
"""


def _section(title: str) -> None:
    print(f"\n{_BOLD}┌─ {title} ─┐{_RESET}")


def _check(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET} {msg}")


def _ask(
    prompt: str,
    *,
    default: str | None = None,
    secret: bool = False,
    choices: list[str] | None = None,
) -> str:
    """Prompt user with optional default, masking, and choice validation."""
    suffix = ""
    if choices:
        suffix = f" [{'/'.join(choices)}]"
    if default is not None:
        suffix += f" ({default})"
    suffix += ": "
    full_prompt = f"  {prompt}{suffix}"

    while True:
        if secret:
            value = getpass.getpass(full_prompt)
        else:
            value = input(full_prompt)
        value = value.strip()
        if not value and default is not None:
            return default
        if not value:
            print(f"    {_RED}This field is required.{_RESET}")
            continue
        if choices and value not in choices:
            print(f"    {_RED}Choose one of: {', '.join(choices)}{_RESET}")
            continue
        return value


def _welcome() -> None:
    print(_BRAIN)
    print(f"  {_BOLD}Engram Setup Wizard{_RESET}")
    print(f"  {_DIM}Persistent memory layer for AI agents{_RESET}")
    print()


def _collect_config(preset_mode: str | None = None) -> dict:
    """Walk user through all config questions. Returns config dict.

    If *preset_mode* is given (e.g. ``"lite"`` or ``"full"``), the mode
    prompt is skipped and the value is used directly.
    """
    cfg: dict[str, str | None] = {}

    # --- API Keys ---
    _section("API Keys")
    cfg["ANTHROPIC_API_KEY"] = _ask("Anthropic API key", secret=True)
    _check("Anthropic API key configured")

    voyage = _ask(
        "Voyage AI key (enables vector search, blank to skip)",
        default="", secret=True,
    )
    if voyage:
        cfg["VOYAGE_API_KEY"] = voyage
        _check("Voyage AI key configured")
    else:
        cfg["VOYAGE_API_KEY"] = None
        _warn("Voyage AI skipped — vector search disabled")

    # --- Mode ---
    if preset_mode is not None:
        mode = preset_mode
        _section("Engine Mode")
        _check(f"Mode: {mode} (pre-selected)")
    else:
        _section("Engine Mode")
        print(f"  {_DIM}lite  = SQLite only (no Docker needed){_RESET}")
        print(f"  {_DIM}full  = FalkorDB + Redis (requires Docker){_RESET}")
        print(f"  {_DIM}auto  = try full, fall back to lite{_RESET}")
        mode = _ask("Mode", default="auto", choices=["lite", "full", "auto"])
        _check(f"Mode: {mode}")
    cfg["ENGRAM_MODE"] = mode

    # --- Full-mode passwords ---
    if mode in ("full", "auto"):
        _section("Full Mode (FalkorDB / Redis)")
        falkor_pw = _ask("FalkorDB password", default="engram_dev")
        redis_pw = _ask("Redis password", default="engram_dev")
        cfg["ENGRAM_FALKORDB__PASSWORD"] = falkor_pw
        cfg["ENGRAM_REDIS__URL"] = f"redis://:{redis_pw}@localhost:6381/0"
        _check("Database passwords configured")
    else:
        cfg["ENGRAM_FALKORDB__PASSWORD"] = None
        cfg["ENGRAM_REDIS__URL"] = None

    # --- Consolidation ---
    _section("Consolidation Profile")
    print(f"  {_DIM}off          = no background consolidation{_RESET}")
    print(f"  {_DIM}observe      = dry-run (logs only, no mutations){_RESET}")
    print(f"  {_DIM}conservative = merge + prune, no LLM inference{_RESET}")
    print(f"  {_DIM}standard     = all phases enabled{_RESET}")
    profile = _ask(
        "Profile",
        default="standard",
        choices=["off", "observe", "conservative", "standard"],
    )
    cfg["ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE"] = profile
    _check(f"Consolidation profile: {profile}")

    # --- Recall Features ---
    _section("Recall Features")
    print(f"  {_DIM}off    = no smart recall{_RESET}")
    print(
        f"  {_DIM}wave1  = auto-recall + natural analyzer + structural signals{_RESET}"
    )
    print(
        f"  {_DIM}wave2  = + graph grounding + conversation awareness + planner{_RESET}"
    )
    print(
        f"  {_DIM}wave3  = + shift/impoverishment live + proactive intelligence{_RESET}"
    )
    print(f"  {_DIM}wave4  = + prospective memory (trigger-based intentions){_RESET}")
    print(f"  {_DIM}all    = all recall features enabled{_RESET}")
    recall_profile = _ask(
        "Recall profile", default="all",
        choices=["off", "wave1", "wave2", "wave3", "wave4", "all"],
    )
    cfg["ENGRAM_ACTIVATION__RECALL_PROFILE"] = recall_profile
    _check(f"Recall profile: {recall_profile}")

    # --- Integration Profile ---
    _section("Integration Profile")
    print(
        f"  {_DIM}off     = keep consolidation, recall, and cue/projection rollout "
        f"separate{_RESET}"
    )
    print(
        f"  {_DIM}rework  = recommended recall-ready preset with cue/projection "
        f"and live natural recall{_RESET}"
    )
    integration_profile = _ask(
        "Integration profile",
        default="rework",
        choices=["off", "rework"],
    )
    cfg["ENGRAM_ACTIVATION__INTEGRATION_PROFILE"] = integration_profile
    _check(f"Integration profile: {integration_profile}")

    # --- Security ---
    _section("Security (optional)")
    enable_auth = _ask("Enable bearer-token auth?", default="n", choices=["y", "n"])
    if enable_auth == "y":
        auto = _ask("Auto-generate token?", default="y", choices=["y", "n"])
        if auto == "y":
            token = secrets.token_hex(32)
            print(f"    {_DIM}Generated token (save this!): {token[:8]}...{_RESET}")
        else:
            token = _ask("Bearer token", secret=True)
        cfg["ENGRAM_AUTH__ENABLED"] = "true"
        cfg["ENGRAM_AUTH__BEARER_TOKEN"] = token
        _check("Auth enabled")
    else:
        cfg["ENGRAM_AUTH__ENABLED"] = None
        cfg["ENGRAM_AUTH__BEARER_TOKEN"] = None

    enable_enc = _ask("Enable encryption?", default="n", choices=["y", "n"])
    if enable_enc == "y":
        auto = _ask("Auto-generate master key?", default="y", choices=["y", "n"])
        if auto == "y":
            key = secrets.token_hex(32)
            print(f"    {_DIM}Generated key (save this!): {key[:8]}...{_RESET}")
        else:
            key = _ask("Master key", secret=True)
        cfg["ENGRAM_ENCRYPTION__ENABLED"] = "true"
        cfg["ENGRAM_ENCRYPTION__MASTER_KEY"] = key
        _check("Encryption enabled")
    else:
        cfg["ENGRAM_ENCRYPTION__ENABLED"] = None
        cfg["ENGRAM_ENCRYPTION__MASTER_KEY"] = None

    return cfg


def _generate_env(config: dict, env_path: Path) -> None:
    """Write .env from config dict. Backs up existing file."""
    if env_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = env_path.with_name(f".env.backup.{ts}")
        env_path.rename(backup)
        _warn(f"Existing .env backed up to {backup.name}")

    lines = [
        "# Engram configuration — generated by `python -m engram setup`",
        f"# {datetime.now().isoformat()}",
        "",
    ]

    # Define sections for organized output
    sections: list[tuple[str, list[tuple[str, str | None]]]] = [
        ("API Keys", [
            ("ANTHROPIC_API_KEY", config.get("ANTHROPIC_API_KEY")),
            ("VOYAGE_API_KEY", config.get("VOYAGE_API_KEY")),
        ]),
        ("Engine", [
            ("ENGRAM_MODE", config.get("ENGRAM_MODE")),
        ]),
        ("Full Mode", [
            ("ENGRAM_FALKORDB__PASSWORD", config.get("ENGRAM_FALKORDB__PASSWORD")),
            ("ENGRAM_REDIS__URL", config.get("ENGRAM_REDIS__URL")),
        ]),
        ("Consolidation", [
            (
                "ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE",
                config.get("ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE"),
            ),
        ]),
        ("Recall", [
            (
                "ENGRAM_ACTIVATION__RECALL_PROFILE",
                config.get("ENGRAM_ACTIVATION__RECALL_PROFILE"),
            ),
            (
                "ENGRAM_ACTIVATION__INTEGRATION_PROFILE",
                config.get("ENGRAM_ACTIVATION__INTEGRATION_PROFILE"),
            ),
        ]),
        ("Security", [
            ("ENGRAM_AUTH__ENABLED", config.get("ENGRAM_AUTH__ENABLED")),
            ("ENGRAM_AUTH__BEARER_TOKEN", config.get("ENGRAM_AUTH__BEARER_TOKEN")),
            ("ENGRAM_ENCRYPTION__ENABLED", config.get("ENGRAM_ENCRYPTION__ENABLED")),
            ("ENGRAM_ENCRYPTION__MASTER_KEY", config.get("ENGRAM_ENCRYPTION__MASTER_KEY")),
        ]),
    ]

    for section_name, keys in sections:
        lines.append(f"# {section_name}")
        for key, value in keys:
            if value is not None:
                lines.append(f"{key}={value}")
            else:
                lines.append(f"# {key}=")
        lines.append("")

    env_path.write_text("\n".join(lines))
    _check(f".env written to {env_path}")


def _print_mcp_config(config: dict) -> None:
    """Print ready-to-paste MCP config snippets."""
    _section("MCP Client Configuration")

    server_dir = str(Path(__file__).resolve().parent.parent)
    api_key = config.get("ANTHROPIC_API_KEY", "")

    env_block: dict[str, str] = {
        "ANTHROPIC_API_KEY": api_key,
        "ENGRAM_MODE": str(config.get("ENGRAM_MODE", "auto")),
        "ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE": str(
            config.get("ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE", "standard")
        ),
        "ENGRAM_ACTIVATION__RECALL_PROFILE": str(
            config.get("ENGRAM_ACTIVATION__RECALL_PROFILE", "all")
        ),
        "ENGRAM_ACTIVATION__INTEGRATION_PROFILE": str(
            config.get("ENGRAM_ACTIVATION__INTEGRATION_PROFILE", "rework")
        ),
    }
    voyage_key = config.get("VOYAGE_API_KEY")
    if voyage_key:
        env_block["VOYAGE_API_KEY"] = voyage_key
    if config.get("ENGRAM_FALKORDB__PASSWORD"):
        env_block["ENGRAM_FALKORDB__PASSWORD"] = str(config["ENGRAM_FALKORDB__PASSWORD"])
    if config.get("ENGRAM_REDIS__URL"):
        env_block["ENGRAM_REDIS__URL"] = str(config["ENGRAM_REDIS__URL"])

    snippet = {
        "mcpServers": {
            "engram": {
                "command": "uv",
                "args": ["run", "--directory", server_dir, "python", "-m", "engram.mcp.server"],
                "env": env_block,
            }
        }
    }

    formatted = json.dumps(snippet, indent=2)

    desktop_path = "~/Library/Application Support/Claude/claude_desktop_config.json"
    print(f"\n  {_bold('Claude Desktop')} {_dim(f'({desktop_path})')}")
    print(textwrap.indent(formatted, "  "))

    print(f"\n  {_bold('Claude Code')} {_dim('(.mcp.json or .claude/settings.json)')}")
    print(textwrap.indent(formatted, "  "))
    print()


def _print_recall_ready_summary(config: dict) -> None:
    """Print the practical recall posture configured by setup."""
    _section("Recall Ready")
    print("  This install is configured for end-to-end recall testing:")
    print("    - analyzer + structural signals live")
    print("    - graph grounding live")
    print("    - shift + impoverishment live")
    print("    - cue/projection rework enabled")
    print("    - surfaced/used recall telemetry enabled")
    print("    - adaptive thresholds, graph override, and chat retry stay off by default")
    print()
    print(f"  {_DIM}Mode: {config.get('ENGRAM_MODE', 'auto')}{_RESET}")
    print(
        f"  {_DIM}Profiles: consolidation="
        f"{config.get('ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE', 'standard')}, "
        f"recall={config.get('ENGRAM_ACTIVATION__RECALL_PROFILE', 'all')}, "
        f"integration={config.get('ENGRAM_ACTIVATION__INTEGRATION_PROFILE', 'rework')}"
        f"{_RESET}"
    )
    print()


def _smoke_test(config: dict) -> None:
    """Quick validation: import check + optional API key ping."""
    _section("Smoke Test")
    try:
        import engram  # noqa: F401
        _check("engram package importable")
    except ImportError as e:
        _warn(f"Import failed: {e}")
        return

    api_key = config.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        _warn("No API key to test")
        return

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=16,
            messages=[{"role": "user", "content": "Say 'ok'"}],
        )
        _check("Anthropic API reachable (model: claude-haiku-4-5-20251001)")
    except Exception as e:
        _warn(f"API check failed: {e}")


def _default_env_path() -> Path:
    """Global config location: ~/.engram/.env"""
    return Path.home() / ".engram" / ".env"


# --- Config editor (interactive menu) ---

# Setting definitions: (key, label, secret, choices_or_None)
_SETTINGS: list[tuple[str, str, str, bool, list[str] | None]] = [
    # (key, section, label, secret, choices)
    ("ANTHROPIC_API_KEY", "API Keys", "Anthropic API key", True, None),
    ("VOYAGE_API_KEY", "API Keys", "Voyage AI key", True, None),
    ("ENGRAM_MODE", "Engine", "Engine mode", False, ["lite", "full", "auto"]),
    (
        "ENGRAM_FALKORDB__PASSWORD",
        "Full Mode",
        "FalkorDB password",
        True,
        None,
    ),
    ("ENGRAM_REDIS__URL", "Full Mode", "Redis URL", True, None),
    (
        "ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE",
        "Consolidation",
        "Profile",
        False,
        ["off", "observe", "conservative", "standard"],
    ),
    (
        "ENGRAM_ACTIVATION__RECALL_PROFILE",
        "Recall",
        "Recall profile",
        False,
        ["off", "wave1", "wave2", "wave3", "wave4", "all"],
    ),
    (
        "ENGRAM_ACTIVATION__INTEGRATION_PROFILE",
        "Recall",
        "Integration profile",
        False,
        ["off", "rework"],
    ),
    ("ENGRAM_AUTH__ENABLED", "Security", "Auth enabled", False, ["true", "false"]),
    ("ENGRAM_AUTH__BEARER_TOKEN", "Security", "Bearer token", True, None),
    (
        "ENGRAM_ENCRYPTION__ENABLED",
        "Security",
        "Encryption enabled",
        False,
        ["true", "false"],
    ),
    ("ENGRAM_ENCRYPTION__MASTER_KEY", "Security", "Master key", True, None),
]


def _load_env(env_path: Path) -> dict[str, str]:
    """Parse a .env file into a dict. Skips comments and blank lines."""
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip()
    return values


def _mask_value(value: str, secret: bool) -> str:
    """Mask secret values for display, showing first 4 + last 4 chars."""
    if not value:
        return f"{_DIM}(not set){_RESET}"
    if not secret:
        return value
    if len(value) <= 10:
        return "\u2022" * len(value)
    return f"{value[:4]}{'•' * 8}{value[-4:]}"


def _render_menu(
    config: dict[str, str], env_path: Path, dirty: bool,
) -> list[str]:
    """Render the settings menu. Returns list of keys in display order."""
    print("\033[2J\033[H", end="")  # clear screen
    print(f"  {_BOLD}Engram Configuration{_RESET}", end="")
    if dirty:
        print(f"  {_YELLOW}(unsaved changes){_RESET}", end="")
    print(f"\n  {_DIM}{env_path}{_RESET}\n")

    keys_in_order: list[str] = []
    current_section = ""
    for i, (key, section, label, secret, _choices) in enumerate(_SETTINGS):
        if section != current_section:
            current_section = section
            print(f"  {_BOLD}{_CYAN}{section}{_RESET}")
        num = i + 1
        val = config.get(key, "")
        display = _mask_value(val, secret)
        print(f"    {_DIM}{num:>2}.{_RESET} {label:<28} {display}")
        keys_in_order.append(key)

    print()
    print(f"  {_DIM}[1-{len(_SETTINGS)}]{_RESET} Edit setting")
    print(f"  {_DIM}[s]{_RESET}    Save    {_DIM}[q]{_RESET} Quit")
    if dirty:
        print(f"  {_DIM}[d]{_RESET}    Discard changes")
    print()
    return keys_in_order


def _edit_setting(
    config: dict[str, str], idx: int,
) -> bool:
    """Edit a single setting. Returns True if changed."""
    key, section, label, secret, choices = _SETTINGS[idx]
    current = config.get(key, "")

    print(f"\n  {_BOLD}Editing: {label}{_RESET}")
    if current:
        masked = _mask_value(current, secret)
        print(f"  Current: {masked}")

    if choices:
        print(f"  Options: {', '.join(choices)}")

    # Special: offer auto-generate for tokens/keys
    if key in ("ENGRAM_AUTH__BEARER_TOKEN", "ENGRAM_ENCRYPTION__MASTER_KEY"):
        gen = _ask("Auto-generate?", default="y", choices=["y", "n"])
        if gen == "y":
            val = secrets.token_hex(32)
            print(f"  {_DIM}Generated: {val[:8]}...{_RESET}")
            config[key] = val
            _check(f"{label} updated")
            return True

    default = current if current else None
    new_val = _ask(
        f"New value (blank to {'keep' if current else 'skip'})",
        default=default or "",
        secret=secret,
        choices=choices,
    )

    if new_val != current:
        if new_val:
            config[key] = new_val
        elif key in config:
            del config[key]
        _check(f"{label} updated")
        return True
    return False


def _resolve_env_path(env_path: Path | None = None) -> Path | None:
    """Find existing .env: explicit path > ~/.engram/.env > ./.env > ../.env"""
    if env_path is not None:
        return env_path if env_path.exists() else None

    candidates = [
        _default_env_path(),                          # ~/.engram/.env
        Path.cwd() / ".env",                          # ./server/.env
        Path.cwd().parent / ".env",                   # ./Engram/.env (project root)
        Path(__file__).resolve().parent.parent.parent / ".env",  # relative to package
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def config_editor(env_path: Path | None = None) -> None:
    """Interactive config menu. Edit settings and save."""
    resolved = _resolve_env_path(env_path)

    if resolved is None:
        searched = env_path or _default_env_path()
        print(f"\n  {_YELLOW}No config found at {searched}{_RESET}")
        print(f"  Run {_bold('python -m engram setup')} first.\n")
        return

    env_path = resolved

    config = _load_env(env_path)
    dirty = False

    while True:
        _render_menu(config, env_path, dirty)
        choice = input(f"  {_BOLD}>{_RESET} ").strip().lower()

        if choice == "q":
            if dirty:
                confirm = _ask(
                    "Unsaved changes. Quit without saving?",
                    default="n", choices=["y", "n"],
                )
                if confirm != "y":
                    continue
            break
        elif choice == "s":
            _generate_env(config, env_path)
            dirty = False
            input(f"  {_DIM}Press Enter to continue...{_RESET}")
        elif choice == "d" and dirty:
            config = _load_env(env_path)
            dirty = False
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(_SETTINGS):
                changed = _edit_setting(config, idx)
                if changed:
                    dirty = True
                input(f"  {_DIM}Press Enter to continue...{_RESET}")
            else:
                print(f"  {_RED}Invalid number.{_RESET}")
                input(f"  {_DIM}Press Enter to continue...{_RESET}")


# ─── Hook Installation ──────────────────────────────────────────────

_HOOKS_DIR = Path.home() / ".engram" / "hooks"
_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

_HOOK_SCRIPTS = {
    "capture-prompt.sh": "capture-prompt.sh",
    "capture-response.sh": "capture-response.sh",
    "session-start.sh": "session-start.sh",
    "session-end.sh": "session-end.sh",
}

_HOOKS_CONFIG = {
    "UserPromptSubmit": [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": str(_HOOKS_DIR / "capture-prompt.sh"),
                    "async": True,
                    "timeout": 5000,
                }
            ],
        }
    ],
    "Stop": [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": str(_HOOKS_DIR / "capture-response.sh"),
                    "async": True,
                    "timeout": 5000,
                }
            ],
        }
    ],
    "SessionStart": [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": str(_HOOKS_DIR / "session-start.sh"),
                    "async": True,
                    "timeout": 5000,
                }
            ],
        }
    ],
    "SessionEnd": [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": str(_HOOKS_DIR / "session-end.sh"),
                    "async": True,
                    "timeout": 10000,
                }
            ],
        }
    ],
}


def _get_hook_source_dir() -> Path:
    """Return the directory containing the hook script templates."""
    return Path.home() / ".engram" / "hooks"


def install_hooks(
    hooks_dir: Path | None = None,
    settings_path: Path | None = None,
) -> dict:
    """Install Engram AutoCapture hooks into Claude Code settings.

    Creates hook scripts in ~/.engram/hooks/ and merges hook config
    into ~/.claude/settings.json (preserving existing hooks).

    Returns dict with 'scripts' and 'settings_updated' keys.
    """
    hooks_dir = hooks_dir or _HOOKS_DIR
    settings_path = settings_path or _SETTINGS_PATH

    result: dict = {"scripts": [], "settings_updated": False}

    # Ensure hooks dir exists
    hooks_dir.mkdir(parents=True, exist_ok=True)

    # Copy hook scripts from the installed location
    source_dir = _get_hook_source_dir()
    for script_name in _HOOK_SCRIPTS:
        src = source_dir / script_name
        dst = hooks_dir / script_name
        if src.exists() and src != dst:
            import shutil
            shutil.copy2(src, dst)
            dst.chmod(0o755)
            result["scripts"].append(str(dst))
        elif dst.exists():
            dst.chmod(0o755)
            result["scripts"].append(str(dst))

    # Merge hooks config into settings.json
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
    else:
        settings = {}

    existing_hooks = settings.get("hooks", {})

    # Build hooks config with resolved paths
    hooks_config = {}
    for event_name, event_hooks in _HOOKS_CONFIG.items():
        resolved = []
        for hook in event_hooks:
            hook_copy = dict(hook)
            command = hook_copy.get("command")
            if isinstance(command, str):
                hook_copy["command"] = str(hooks_dir / Path(command).name)
            nested_hooks = hook_copy.get("hooks")
            if isinstance(nested_hooks, list):
                inner = []
                for h in nested_hooks:
                    hc = dict(h)
                    inner_command = hc.get("command")
                    if isinstance(inner_command, str):
                        hc["command"] = str(hooks_dir / Path(inner_command).name)
                    inner.append(hc)
                hook_copy["hooks"] = inner
            resolved.append(hook_copy)
        hooks_config[event_name] = resolved

    # Merge: add Engram hooks without removing existing hooks
    for event_name, new_hooks in hooks_config.items():
        if event_name not in existing_hooks:
            existing_hooks[event_name] = new_hooks
        else:
            # Collect all existing command paths (inside matcher wrappers)
            existing_cmds = set()
            for entry in existing_hooks[event_name]:
                nested_hooks = entry.get("hooks", [])
                if not isinstance(nested_hooks, list):
                    continue
                for inner in nested_hooks:
                    if isinstance(inner, dict):
                        existing_cmds.add(str(inner.get("command", "")))

            for new_entry in new_hooks:
                nested_hooks = new_entry.get("hooks", [])
                inner_cmds = [
                    str(h.get("command", ""))
                    for h in nested_hooks
                    if isinstance(h, dict)
                ]
                if not any(cmd in existing_cmds for cmd in inner_cmds):
                    existing_hooks[event_name].append(new_entry)

    settings["hooks"] = existing_hooks
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    result["settings_updated"] = True

    return result


def install_hooks_interactive(
    hooks_dir: Path | None = None,
    settings_path: Path | None = None,
) -> None:
    """Interactive hook installation with terminal output."""
    _section("Engram AutoCapture Hooks")
    print(f"  {_DIM}Installs Claude Code hooks that automatically capture{_RESET}")
    print(f"  {_DIM}user prompts and assistant responses into Engram.{_RESET}")
    print()

    result = install_hooks(hooks_dir=hooks_dir, settings_path=settings_path)

    for script in result["scripts"]:
        _check(f"Script: {script}")

    if result["settings_updated"]:
        path = settings_path or _SETTINGS_PATH
        _check(f"Settings updated: {path}")

    print()
    print(f"  {_BOLD}Hooks configured:{_RESET}")
    print("    UserPromptSubmit  → capture user prompts")
    print("    Stop              → capture assistant responses")
    print("    SessionStart      → replay queued + session marker")
    print("    SessionEnd        → session marker + trigger consolidation")
    print()
    print(f"  {_DIM}Hooks are async and never block Claude.{_RESET}")
    print(f"  {_DIM}Server: $ENGRAM_URL (default http://localhost:8100){_RESET}")
    print()
    print(f"  {_GREEN}{_BOLD}AutoCapture installed!{_RESET}")
    print()


def setup(env_path: Path | None = None, mode: str | None = None) -> None:
    """Interactive setup wizard entry point.

    Parameters
    ----------
    env_path : Path | None
        Where to write the ``.env`` file.  Defaults to ``~/.engram/.env``.
    mode : str | None
        Pre-select engine mode (``"lite"``, ``"full"``, or ``"auto"``).
        When set, the mode prompt in the wizard is skipped.
    """
    _welcome()

    config = _collect_config(preset_mode=mode)

    if env_path is None:
        env_path = _default_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)

    _section("Generate Files")
    _generate_env(config, env_path)
    _print_mcp_config(config)
    _print_recall_ready_summary(config)

    # Optional smoke test
    run_test = _ask("Run smoke test?", default="n", choices=["y", "n"])
    if run_test == "y":
        _smoke_test(config)

    print(f"\n  {_GREEN}{_BOLD}Setup complete!{_RESET}")
    print(f"  {_DIM}Config: {env_path}{_RESET}")
    mcp_cmd = "cd server && uv run python -m engram.mcp.server"
    print(f"  {_DIM}Start MCP server:  {mcp_cmd}{_RESET}")
    http_cmd = "cd server && uv run python -m engram --transport http"
    print(f"  {_DIM}Start HTTP server: {http_cmd}{_RESET}")
    print()
