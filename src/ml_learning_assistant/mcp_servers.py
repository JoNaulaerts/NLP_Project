"""
MCP Server configuration (Docker MCP Gateway via STDIO).

We keep this file minimal and deterministic.
"""

def docker_mcp_stdio_params():
    """
    Returns a plain dict of stdio params compatible with MCPServerAdapter.
    (Avoids CrewAI native MCP caching edge cases.)
    """
    return {
        "command": "docker",
        "args": ["mcp", "gateway", "run"],
        # env is intentionally omitted here; Docker MCP will inherit your process env.
        # If you need to force env vars, pass them in the adapter call site.
    }
