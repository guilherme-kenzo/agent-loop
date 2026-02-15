#!/usr/bin/env python3
"""Minimal MCP server exposing a single 'hello' tool.

This server runs over stdio transport and is used by mcp_agent.py.

Run standalone (for testing):
    python examples/mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

server = FastMCP("hello-server")


@server.tool()
def hello(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}! Welcome to the world of MCP."


if __name__ == "__main__":
    server.run(transport="stdio")
