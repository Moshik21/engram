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
        "                            Verify the native Helix brain loop end to end\n"
        "  engram axi --project $PWD\n"
        "                            Print compact agent-facing runtime/context packet\n"
        "  engram axi recall \"query\" --limit 5\n"
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
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
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
    helix_data_dir = getattr(args, "helix_data_dir", None)
    if helix_data_dir is not None:
        os.environ["ENGRAM_HELIX__TRANSPORT"] = "native"
        os.environ["ENGRAM_HELIX__DATA_DIR"] = str(helix_data_dir.expanduser())


if __name__ == "__main__":
    main()
