from __future__ import annotations

from fastmcp import FastMCP


def create_server() -> FastMCP:
    return FastMCP(name="paper-context")


def create_http_app():
    mcp = create_server()
    return mcp.http_app(path="/", transport="streamable-http")
