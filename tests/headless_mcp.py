"""Exercise the Linux headless MCP server without a Telegram login or model."""

import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, port, socket_path):
        super().__init__("127.0.0.1", port, timeout=3)
        self.socket_path = socket_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(str(self.socket_path))
        except OSError:
            sock.close()
            raise
        self.sock = sock


def request(port, method, path, body=None, token=None, origin=None, socket_path=None):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if origin is not None:
        headers["Origin"] = origin
    conn = (http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            if socket_path is None else UnixHTTPConnection(port, socket_path))
    try:
        conn.request(method, path, None if body is None else json.dumps(body), headers)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def rpc(port, method, status=200, params=None, **headers):
    body = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        body["params"] = params
    actual, response = request(port, "POST", "/mcp", body, **headers)
    expect(actual == status, f"{method}: expected HTTP {status}, got {actual}")
    if status == 200:
        message = json.loads(response)
        expect(message.get("jsonrpc") == "2.0" and message.get("id") == 1,
               f"{method}: invalid JSON-RPC response")
        return message["result"]


def wait_for_server(port, service, socket_path):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        expect(service.poll() is None, f"service exited early ({service.returncode})")
        try:
            status, _ = request(port, "GET", "/api/status", socket_path=socket_path)
        except OSError:
            time.sleep(0.05)
        else:
            expect(status == 401, f"expected status endpoint to require auth, got {status}")
            return
    raise AssertionError("service did not start within 15 seconds")


def run(binary, enabled):
    with tempfile.TemporaryDirectory(prefix="seekgram-mcp-") as directory:
        data = Path(directory)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        settings = {"port": port, "check_updates": False}
        if enabled:
            settings["mcp"] = True
        (data / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        (data / "settings.json").chmod(0o600)
        with subprocess.Popen(
            [str(binary), "service"],
            env={**os.environ, "SEEKGRAM_DATA_DIR": directory},
        ) as service:
            try:
                socket_path = data / "control.sock"
                wait_for_server(port, service, socket_path)
                # This is the service-generated token in our new empty temp directory.
                secrets = json.loads((data / "secrets.bin").read_text(encoding="utf-8"))
                token = secrets["token"]
                expect(isinstance(token, str) and len(token) >= 32, "missing test bearer token")
                status, _ = request(port, "GET", "/api/status", token=token,
                                    socket_path=socket_path)
                expect(status == 200, f"owner socket rejected status request: HTTP {status}")
                status, _ = request(port, "GET", "/api/status", token=token)
                expect(status == 404, f"owner API exposed on TCP: HTTP {status}")
                if not enabled:
                    rpc(port, "initialize", status=404, token=token)
                else:
                    verify_mcp(port, token)
                status, _ = request(port, "POST", "/api/shutdown", token=token,
                                    socket_path=socket_path)
                expect(status == 200, f"shutdown failed: HTTP {status}")
                expect(service.wait(timeout=10) == 0, "service did not exit cleanly")
                for name in ("telegram.sqlite3", "telegram.session", "model"):
                    expect(not (data / name).exists(), f"headless service unexpectedly created {name}")
            finally:
                if service.poll() is None:
                    service.kill()
                    service.wait()


def verify_mcp(port, token):
    rpc(port, "initialize", status=401)
    hello = rpc(port, "initialize", token=token, params={
        "protocolVersion": "2025-11-25", "capabilities": {},
        "clientInfo": {"name": "seekgram-smoke-test", "version": "1"},
    })
    expect(hello["protocolVersion"] == "2025-11-25" and "tools" in hello["capabilities"]
           and isinstance(hello["serverInfo"]["name"], str), "invalid MCP handshake")

    tools = rpc(port, "tools/list", token=token)["tools"]
    by_name = {tool["name"]: tool["inputSchema"] for tool in tools}
    expect(len(by_name) == len(tools), "duplicate MCP tool names")
    expect(all(schema["type"] == "object" for schema in by_name.values()),
           "invalid MCP input schema")
    expect({"dialogs", "history", "search", "media_search_vector"} <= by_name.keys(), "missing read tools")
    expect(by_name["history"]["required"] == ["chat"], "unexpected history schema")

    history = {"name": "history", "arguments": {"chat": "12345"}}
    rpc(port, "tools/call", status=403, token=token, origin="http://example.invalid",
        params=history)
    rpc(port, "tools/call", status=401, params=history)
    rpc(port, "tools/call", status=401, token="wrong-token", params=history)
    result = rpc(port, "tools/call", token=token, params=history)
    expect(result["isError"] is True and result["content"][0]["type"] == "text"
           and "not logged in" in result["content"][0]["text"], "missing logged-out history error")


if __name__ == "__main__":
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "target/debug/seekgram").resolve()
    run(binary, enabled=False)
    run(binary, enabled=True)
    print("Headless MCP smoke test passed.")
