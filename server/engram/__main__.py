"""CLI entrypoint for Engram.

Usage:
    engram setup          Interactive setup wizard
    engram serve          Start REST API server (port 8100)
    engram mcp            Start MCP server (stdio or HTTP)
    engram config         Edit configuration
    engram hooks          Install AutoCapture hooks
    engram health         Check if server is running
    engram update         Update Engram to latest version
    engram version        Show installed version
"""

import argparse
import logging
import sys


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
        choices=["lite", "full", "auto"],
        default=None,
        help="Pre-select engine mode (skips mode prompt)",
    )

    # --- serve (REST API) ---
    serve_parser = subparsers.add_parser("serve", help="Start REST API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8100, help="Port (default: 8100)")
    serve_parser.add_argument(
        "--mode",
        choices=["lite", "full", "auto"],
        default="auto",
        help="Engine mode (default: auto-detect)",
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
        choices=["lite", "full", "auto"],
        default="auto",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    # --- setup ---
    if args.command == "setup":
        from pathlib import Path

        from engram.setup import setup

        env_path = Path(args.env_path) if args.env_path else None
        setup(env_path=env_path, mode=args.mode)
        return

    # --- serve ---
    if args.command == "serve":
        import uvicorn

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
        from pathlib import Path

        from engram.setup import config_editor

        env_path = Path(args.env_path) if args.env_path else None
        config_editor(env_path=env_path)
        return

    # --- hooks ---
    if args.command == "hooks":
        from pathlib import Path

        from engram.setup import install_hooks_interactive

        hooks_dir = Path(args.hooks_dir) if args.hooks_dir else None
        settings_path = Path(args.settings_path) if args.settings_path else None
        install_hooks_interactive(hooks_dir=hooks_dir, settings_path=settings_path)
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
        from pathlib import Path

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


if __name__ == "__main__":
    main()
