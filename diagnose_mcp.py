#!/usr/bin/env python3
import os
import sys
import json
import httpx

print("=" * 70)
print("MCP GATEWAY (STREAMABLE HTTP) DIAGNOSTIC")
print("=" * 70)

url = os.getenv("MCP_GATEWAY_URL", "http://mcp_gateway:3000/mcp").strip()
print(f"\nGateway URL: {url}")

# Docker MCP Gateway streaming transport requires /mcp
if not url.rstrip("/").endswith("/mcp"):
    print("✗ URL must end with /mcp (Docker MCP Gateway streaming endpoint).")
    sys.exit(2)

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}

init_payload = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "diagnose_mcp_gateway.py", "version": "1.0"},
    },
}

with httpx.Client(timeout=20.0) as client:
    print("\n1) POST initialize ...")
    r = client.post(url, headers=headers, json=init_payload)

    # Some gateways respond with SSE; still, the session id is in headers.
    sid = r.headers.get("Mcp-Session-Id") or r.headers.get("mcp-session-id")
    print(f"   HTTP {r.status_code}")
    print(f"   Mcp-Session-Id: {sid}")

    if not sid:
        print("✗ No Mcp-Session-Id returned; gateway not behaving like streamable-http session mode.")
        print("Body (first 800 chars):")
        print(r.text[:800])
        sys.exit(3)

    print("\n2) POST tools/list with Mcp-Session-Id ...")
    tools_payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    r2 = client.post(
        url,
        headers={**headers, "Mcp-Session-Id": sid},
        json=tools_payload,
    )

    print(f"   HTTP {r2.status_code}")
    text = r2.text

    # Try to parse JSON directly; if SSE, fallback to extracting "data:" lines.
    tools_json = None
    try:
        tools_json = r2.json()
    except Exception:
        lines = [ln for ln in text.splitlines() if ln.startswith("data:")]
        if lines:
            try:
                tools_json = json.loads(lines[-1].removeprefix("data:").strip())
            except Exception:
                tools_json = None

    if not tools_json:
        print("✗ Could not parse tools/list response as JSON.")
        print("Raw (first 1200 chars):")
        print(text[:1200])
        sys.exit(4)

    if "error" in tools_json:
        print("✗ MCP error from gateway:")
        print(json.dumps(tools_json["error"], indent=2))
        sys.exit(5)

    tools = (tools_json.get("result") or {}).get("tools") or []
    print(f"✓ tools/list returned {len(tools)} tools")
    print("First 30 tool names:")
    for t in tools[:30]:
        print(" -", t.get("name"))

print("\n" + "=" * 70)
print("✓ MCP GATEWAY DIAGNOSTIC PASSED")
print("=" * 70)
