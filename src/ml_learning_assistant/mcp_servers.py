import os

def get_mcp_server_params():
    """
    Returns MCP server params in the format MCPServerAdapter expects.
    - For container_gateway: dict with 'url' and 'transport' (streamable-http)
    - For local: dict with 'command' and 'args' (stdio)
    """
    mode = (os.getenv("MCP_MODE", "local") or "local").strip().lower()

    if mode == "container_gateway":
        # Streamable HTTP: return url + transport
        gateway_url = os.getenv("MCP_GATEWAY_URL", "http://mcp_gateway:3000/mcp")
        return {
            "url": gateway_url,
            "transport": "streamable-http",
        }

    # Local stdio (requires docker mcp CLI on host)
    return {
        "command": "docker",
        "args": ["mcp", "gateway", "run"],
    }
