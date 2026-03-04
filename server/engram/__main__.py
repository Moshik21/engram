"""CLI entrypoint: python -m engram.mcp.server (stdio) or python -m engram (HTTP)."""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(description="Engram Memory Server")
    subparsers = parser.add_subparsers(dest="command")
    setup_parser = subparsers.add_parser("setup", help="Interactive setup wizard")
    setup_parser.add_argument(
        "--env-path",
        default=None,
        help="Override .env output path (default: ~/.engram/.env)",
    )
    config_parser = subparsers.add_parser("config", help="Edit configuration")
    config_parser.add_argument(
        "--env-path",
        default=None,
        help="Override .env path (default: ~/.engram/.env)",
    )
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

    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport mode (default: stdio for MCP)",
    )
    parser.add_argument(
        "--mode",
        choices=["lite", "full", "auto"],
        default="auto",
        help="Engine mode (default: auto-detect)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host (HTTP mode)")
    parser.add_argument("--port", type=int, default=8100, help="Server port (HTTP mode)")
    args = parser.parse_args()

    if args.command == "setup":
        from pathlib import Path

        from engram.setup import setup

        env_path = Path(args.env_path) if args.env_path else None
        setup(env_path=env_path)
        return

    if args.command == "config":
        from pathlib import Path

        from engram.setup import config_editor

        env_path = Path(args.env_path) if args.env_path else None
        config_editor(env_path=env_path)
        return

    if args.command == "hooks":
        from pathlib import Path

        from engram.setup import install_hooks_interactive

        hooks_dir = Path(args.hooks_dir) if args.hooks_dir else None
        settings_path = Path(args.settings_path) if args.settings_path else None
        install_hooks_interactive(hooks_dir=hooks_dir, settings_path=settings_path)
        return

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


if __name__ == "__main__":
    main()
