"""CLI entrypoint: python -m engram.mcp.server (stdio) or python -m engram (HTTP)."""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(description="Engram Memory Server")
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
