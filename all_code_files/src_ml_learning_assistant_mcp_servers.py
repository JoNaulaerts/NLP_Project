import os

def docker_mcp_stdio_params():
    mode = (os.getenv("MCP_MODE", "local") or "local").strip().lower()

    if mode == "container_gateway":
        gateway_url = os.getenv("MCP_GATEWAY_URL", "http://mcp_gateway:3000/mcp")
        return {
            "command": "npx",
            "args": [
                "-y",
                "supergateway",
                "--streamableHttp", gateway_url,
                "--outputTransport", "stdio",
            ],
        }

    # old way (works on your localhost where you said it works)
    return {"command": "docker", "args": ["mcp", "gateway", "run"]}
