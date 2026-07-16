"""CLI entrypoint for Engram.

Usage:
    engram setup          Interactive setup wizard
    engram serve          Start REST API server (port 8100)
    engram mcp            Start MCP server (stdio or HTTP)
    engram config         Edit configuration
    engram hooks          Install AutoCapture hooks
    engram lifecycle      Print local Capture -> Cue -> Project -> Recall -> Consolidate state
    engram evaluate       Print local brain-loop evaluation report
    engram axi            Print compact agent-facing AXI context
    engram authority      Generate claim_authority payload for adoption validation
    engram adoption       Validate MCP client adoption transcript
    engram doctor         Run diagnostics and brain-loop readiness smoke
    engram health         Check if server is running
    engram update         Update Engram to latest version
    engram version        Show installed version
"""

import argparse
import logging
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Engram — persistent memory for AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  engram setup              Interactive setup wizard\n"
        "  engram serve              Start REST API at localhost:8100\n"
        "  engram serve --port 9000  Custom port\n"
        "  engram mcp                Start MCP server (stdio)\n"
        "  engram mcp --transport streamable-http\n"
        "                            Start MCP server (HTTP on :8200)\n"
        "  engram lifecycle         Print local brain-loop lifecycle snapshot\n"
        "  engram evaluate          Print brain-loop evaluation report\n"
        "  engram evaluate --require-evaluation-signals\n"
        "                            Fail unless all evaluation signals are measured\n"
        "  engram evaluate --smoke --mode helix\n"
        "  engram continuity --smoke   Continuity golden path (product metric)\n"
        "                            Verify the native Helix brain loop end to end\n"
        "  engram axi --project $PWD\n"
        "                            Print compact agent-facing runtime/context packet\n"
        '  engram axi recall "query" --limit 5\n'
        "                            Search memory from a shell-capable agent\n"
        "  engram authority --out claim-authority.json\n"
        "                            Generate memory authority payload for clients\n"
        "  engram adoption --authority claim-authority.json --calls mcp-calls.jsonl\n"
        "                            Validate an MCP client followed Engram authority\n"
        "  engram doctor            Run diagnostics and readiness smoke\n"
        "  engram showcase run       Replay bundled Liam continuity demo\n"
        "  engram health             Check if server is running\n",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- setup ---
    setup_parser = subparsers.add_parser("setup", help="Interactive setup wizard")
    setup_parser.add_argument(
        "--env-path",
        default=None,
        help="Override .env output path (default: ~/.engram/.env)",
    )
    setup_parser.add_argument(
        "--mode",
        choices=["lite", "full", "helix", "auto"],
        default=None,
        help="Pre-select engine mode (skips mode prompt)",
    )

    # --- serve (REST API) ---
    serve_parser = subparsers.add_parser("serve", help="Start REST API server")
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address (default: 127.0.0.1 — the REST API is unauthenticated "
            "by default; pass 0.0.0.0 only if you understand the exposure)"
        ),
    )
    serve_parser.add_argument("--port", type=int, default=8100, help="Port (default: 8100)")
    serve_parser.add_argument(
        "--mode",
        choices=["lite", "full", "helix", "auto"],
        default=None,
        help="Override engine mode (default: config/auto-detect)",
    )
    serve_parser.add_argument(
        "--helix-data-dir",
        type=Path,
        default=None,
        help="Native Helix data directory for --mode helix.",
    )
    serve_parser.add_argument(
        "--role",
        choices=["shell", "monolith", "brain"],
        default=None,
        help="Runtime role (default: ENGRAM_RUNTIME_ROLE / monolith). shell = no in-process brain.",
    )

    # --- mcp ---
    mcp_parser = subparsers.add_parser("mcp", help="Start MCP server")
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    mcp_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address for HTTP transport (default: 127.0.0.1)",
    )
    mcp_parser.add_argument(
        "--port",
        type=int,
        default=8200,
        help="Port for HTTP transport (default: 8200)",
    )
    mcp_parser.add_argument(
        "--mode",
        choices=["lite", "full", "helix", "auto"],
        default=None,
        help="Override engine mode (default: config/auto-detect)",
    )
    mcp_parser.add_argument(
        "--helix-data-dir",
        type=Path,
        default=None,
        help="Native Helix data directory for --mode helix.",
    )

    # --- config ---
    config_parser = subparsers.add_parser("config", help="Edit configuration")
    config_parser.add_argument(
        "--env-path",
        default=None,
        help="Override .env path (default: ~/.engram/.env)",
    )

    # --- hooks ---
    hooks_parser = subparsers.add_parser("hooks", help="Install AutoCapture hooks")
    hooks_parser.add_argument(
        "--hooks-dir",
        default=None,
        help="Override hooks directory (default: ~/.engram/hooks/)",
    )
    hooks_parser.add_argument(
        "--settings-path",
        default=None,
        help="Override Claude settings path (default: ~/.claude/settings.json)",
    )

    # --- lifecycle ---
    lifecycle_parser = subparsers.add_parser(
        "lifecycle",
        help="Print local brain-loop lifecycle snapshot",
    )
    from engram.lifecycle_cli import configure_lifecycle_parser

    configure_lifecycle_parser(lifecycle_parser)

    # --- evaluate ---
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Print local brain-loop evaluation report",
    )
    from engram.evaluation.cli import configure_evaluate_parser

    configure_evaluate_parser(evaluate_parser)

    # --- dogfood ---
    dogfood_parser = subparsers.add_parser(
        "dogfood",
        help="Replay local transcripts through Engram memory modes",
    )
    from engram.evaluation.dogfood import configure_dogfood_parser

    configure_dogfood_parser(dogfood_parser)

    # --- continuity ---
    continuity_parser = subparsers.add_parser(
        "continuity",
        help="Run continuity golden-path smoke (promote → cold get_context/recall)",
    )
    continuity_parser.add_argument(
        "--smoke",
        action="store_true",
        default=True,
        help="Run the lite continuity golden-path smoke (default)",
    )
    continuity_parser.add_argument(
        "--against-live",
        action="store_true",
        help=(
            "Product gate against the running server: cold get_context/recall must "
            "surface a Decision when the graph has entities (not disposable smoke)."
        ),
    )
    continuity_parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8100",
        help="Server URL for --against-live (default http://127.0.0.1:8100)",
    )
    continuity_parser.add_argument(
        "--max-recall-ms",
        type=float,
        default=2000.0,
        help="Max cold recall latency for --against-live (default 2000)",
    )
    continuity_parser.add_argument(
        "--organic",
        action="store_true",
        help=(
            "Metric v2 for --against-live: target must be a real Decision "
            "aged >= --min-organic-age-days (no self-promotion) with 0 "
            "Decision scrap in top-5"
        ),
    )
    continuity_parser.add_argument(
        "--min-organic-age-days",
        type=float,
        default=7.0,
        help="Minimum age for the organic Decision target (default 7)",
    )
    continuity_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of markdown",
    )

    # --- axi ---
    axi_parser = subparsers.add_parser(
        "axi",
        help="Print compact agent-facing AXI packets",
    )
    from engram.axi.cli import configure_axi_parser

    configure_axi_parser(axi_parser)

    # --- authority ---
    authority_parser = subparsers.add_parser(
        "authority",
        help="Generate memory authority payload for adoption validation",
    )
    from engram.mcp.authority_cli import configure_authority_parser

    configure_authority_parser(authority_parser)

    # --- adoption ---
    adoption_parser = subparsers.add_parser(
        "adoption",
        help="Validate MCP client adoption transcript",
    )
    from engram.mcp.adoption_cli import configure_adoption_parser

    configure_adoption_parser(adoption_parser)

    # --- harness ---
    harness_parser = subparsers.add_parser(
        "harness",
        help="Harness adoption helpers for installer connect flows",
    )
    from engram.harness_cli import configure_harness_parser

    configure_harness_parser(harness_parser)

    # --- captain ---
    captain_parser = subparsers.add_parser(
        "captain",
        help="Export/import human-editable captain preferences",
    )
    from engram.captain_cli import configure_captain_parser

    configure_captain_parser(captain_parser)

    # --- showcase ---
    showcase_parser = subparsers.add_parser(
        "showcase",
        help="Run the bundled Liam continuity demo against lite demo.db",
    )
    from engram.showcase.cli import configure_showcase_parser

    configure_showcase_parser(showcase_parser)

    # --- doctor ---
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run diagnostics and brain-loop readiness smoke",
    )
    from engram.doctor import configure_doctor_parser

    configure_doctor_parser(doctor_parser)

    # --- index (hybrid entity-vector completeness) ---
    index_parser = subparsers.add_parser(
        "index",
        help="Measure/backfill hybrid entity-vector index completeness",
    )
    from engram.index_cli import configure_index_parser

    configure_index_parser(index_parser)

    # --- hygiene (operator mop protocol; not public MCP) ---
    hygiene_parser = subparsers.add_parser(
        "hygiene",
        help="Debt scoreboard + bounded mop drains (operator; not public MCP)",
    )
    from engram.hygiene_cli import configure_hygiene_parser

    configure_hygiene_parser(hygiene_parser)

    # --- loop (Loop Steward control plane; operator / harness only) ---
    loop_parser = subparsers.add_parser(
        "loop",
        help="Loop Steward adjustments: status|apply|clear (operator; not public MCP)",
    )
    from engram.loop_cli import configure_loop_parser

    configure_loop_parser(loop_parser)

    # --- backup (operator; native data dir + state snapshot) ---
    backup_parser = subparsers.add_parser(
        "backup",
        help="Snapshot/verify/restore the native brain (operator; shell must be down)",
    )
    from engram.backup_cli import configure_backup_parser

    configure_backup_parser(backup_parser)

    # --- brain (cold process; not public MCP) ---
    brain_parser = subparsers.add_parser(
        "brain",
        help="Cold-brain process: run consolidation outside the hot shell",
    )
    from engram.brain_cli import configure_brain_parser

    configure_brain_parser(brain_parser)

    # --- health ---
    subparsers.add_parser("health", help="Check if Engram server is running")

    # --- update ---
    subparsers.add_parser("update", help="Update Engram to latest version")

    # --- version ---
    subparsers.add_parser("version", help="Show installed version")

    # Legacy flags (for backward compat with --transport http)
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--host", default="0.0.0.0", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8100, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode",
        choices=["lite", "full", "helix", "auto"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--helix-data-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    # --- setup ---
    if args.command == "setup":
        from engram.setup import setup

        env_path = Path(args.env_path) if args.env_path else None
        setup(env_path=env_path, mode=args.mode)
        return

    # --- serve ---
    if args.command == "serve":
        import uvicorn

        _apply_mode_override(args)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )
        print(f"Starting Engram REST API at http://{args.host}:{args.port}")
        print(f"Interactive docs: http://localhost:{args.port}/docs")
        print("Press Ctrl+C to stop.\n")
        uvicorn.run(
            "engram.main:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
        return

    # --- mcp ---
    if args.command == "mcp":
        _apply_mode_override(args)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )
        from engram.mcp.server import main as mcp_main

        mcp_main(
            transport=args.transport,
            host=args.host,
            port=args.port,
        )
        return

    # --- config ---
    if args.command == "config":
        from engram.setup import config_editor

        env_path = Path(args.env_path) if args.env_path else None
        config_editor(env_path=env_path)
        return

    # --- hooks ---
    if args.command == "hooks":
        from engram.setup import install_hooks_interactive

        hooks_dir = Path(args.hooks_dir) if args.hooks_dir else None
        settings_path = Path(args.settings_path) if args.settings_path else None
        install_hooks_interactive(hooks_dir=hooks_dir, settings_path=settings_path)
        return

    # --- lifecycle ---
    if args.command == "lifecycle":
        import asyncio

        from engram.lifecycle_cli import run_lifecycle_command

        asyncio.run(run_lifecycle_command(args))
        return

    # --- evaluate ---
    if args.command == "evaluate":
        import asyncio

        from engram.evaluation.cli import run_evaluate_command

        asyncio.run(run_evaluate_command(args))
        return

    # --- dogfood ---
    if args.command == "dogfood":
        import asyncio

        from engram.evaluation.dogfood import run_dogfood_command

        sys.exit(asyncio.run(run_dogfood_command(args)))

    # --- continuity ---
    if args.command == "continuity":
        import asyncio
        import json as _json

        from engram.evaluation.continuity import (
            format_continuity_report,
            run_continuity_against_live,
            run_continuity_golden_path_smoke,
        )

        if getattr(args, "against_live", False):
            result = asyncio.run(
                run_continuity_against_live(
                    server_url=getattr(args, "server_url", "http://127.0.0.1:8100"),
                    max_recall_ms=float(getattr(args, "max_recall_ms", 2000.0)),
                    require_organic=bool(getattr(args, "organic", False)),
                    min_organic_age_days=float(getattr(args, "min_organic_age_days", 7.0)),
                )
            )
        else:
            result = asyncio.run(run_continuity_golden_path_smoke())
        if args.json:
            print(_json.dumps(result, indent=2, default=str))
        else:
            print(format_continuity_report(result))
        sys.exit(0 if result.get("passed") else 1)

    # --- axi ---
    if args.command == "axi":
        from engram.axi.cli import run_axi_command

        sys.exit(run_axi_command(args))

    # --- authority ---
    if args.command == "authority":
        import asyncio

        from engram.mcp.authority_cli import run_authority_command

        asyncio.run(run_authority_command(args))
        return

    # --- adoption ---
    if args.command == "adoption":
        from engram.mcp.adoption_cli import run_adoption_command

        sys.exit(run_adoption_command(args))

    # --- harness ---
    if args.command == "harness":
        from engram.harness_cli import run_harness_command

        sys.exit(run_harness_command(args))

    # --- captain ---
    if args.command == "captain":
        from engram.captain_cli import run_captain_command

        sys.exit(run_captain_command(args))

    # --- showcase ---
    if args.command == "showcase":
        from engram.showcase.cli import run_showcase_command

        sys.exit(run_showcase_command(args))

    # --- doctor ---
    if args.command == "doctor":
        import asyncio

        from engram.doctor import run_doctor_command

        asyncio.run(run_doctor_command(args))
        return

    # --- index (hybrid entity-vector completeness) ---
    if args.command == "index":
        import asyncio

        from engram.index_cli import run_index_command

        sys.exit(asyncio.run(run_index_command(args)))

    # --- hygiene (operator mop) ---
    if args.command == "hygiene":
        import asyncio

        from engram.hygiene_cli import run_hygiene_command

        sys.exit(asyncio.run(run_hygiene_command(args)))

    # --- loop (Loop Steward) ---
    if args.command == "loop":
        from engram.loop_cli import run_loop_command

        sys.exit(run_loop_command(args))

    # --- backup ---
    if args.command == "backup":
        import asyncio

        from engram.backup_cli import run_backup_command

        sys.exit(asyncio.run(run_backup_command(args)))

    # --- brain (cold process) ---
    if args.command == "brain":
        from engram.brain_cli import run_brain_command

        sys.exit(run_brain_command(args))

    # --- health ---
    if args.command == "health":
        import urllib.error
        import urllib.request

        port = args.port
        url = f"http://localhost:{port}/health"
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    print(f"Engram is running at localhost:{port}")
                    sys.exit(0)
        except (urllib.error.URLError, OSError):
            pass
        print(f"Engram is not running at localhost:{port}")
        print(f"Start it with: engram serve --port {port}")
        sys.exit(1)

    # --- update ---
    if args.command == "update":
        import subprocess

        # Detect install method: git clone vs pip package
        package_dir = Path(__file__).resolve().parent.parent
        # Walk up to find .git — handles server/engram/__main__.py layout
        repo_root = None
        for parent in [package_dir, package_dir.parent, package_dir.parent.parent]:
            if (parent / ".git").is_dir():
                repo_root = parent
                break
        if repo_root is not None:
            # Git-based install
            print(f"Updating from git ({repo_root})...")
            try:
                subprocess.run(
                    ["git", "pull", "--quiet"],
                    cwd=repo_root,
                    check=True,
                )
                subprocess.run(
                    ["uv", "sync", "--quiet"],
                    cwd=repo_root / "server",
                    check=True,
                )
                # Show new version
                from importlib.metadata import version as get_version

                try:
                    v = get_version("engram")
                except Exception:
                    v = "unknown"
                print(f"Updated to engram {v}")
            except subprocess.CalledProcessError as e:
                print(f"Update failed: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # PyPI-based install
            print("Updating from PyPI...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "engram"],
                    check=True,
                )
                print("Updated successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Update failed: {e}", file=sys.stderr)
                sys.exit(1)
        return

    # --- version ---
    if args.command == "version":
        from importlib.metadata import version as get_version

        try:
            v = get_version("engram")
        except Exception:
            v = "unknown"
        print(f"engram {v}")
        return

    # --- legacy --transport flag ---
    if args.transport:
        _apply_mode_override(args)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )
        if args.transport == "stdio":
            from engram.mcp.server import main as mcp_main

            mcp_main()
        else:
            import uvicorn

            uvicorn.run(
                "engram.main:app",
                host=args.host,
                port=args.port,
                log_level="info",
            )
        return

    # No command given — show help
    parser.print_help()


def _apply_mode_override(args: argparse.Namespace) -> None:
    """Let CLI runtime flags override config files for commands that boot runtime."""
    mode = getattr(args, "mode", None)
    if mode:
        os.environ["ENGRAM_MODE"] = mode
    role = getattr(args, "role", None)
    if role:
        os.environ["ENGRAM_RUNTIME_ROLE"] = role
    helix_data_dir = getattr(args, "helix_data_dir", None)
    if helix_data_dir is not None:
        os.environ["ENGRAM_HELIX__TRANSPORT"] = "native"
        os.environ["ENGRAM_HELIX__DATA_DIR"] = str(helix_data_dir.expanduser())


if __name__ == "__main__":
    main()
