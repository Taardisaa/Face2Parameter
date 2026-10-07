"""Serve the source-method forward viewer through a loopback-only bridge proxy.

No model fitting, probe sweeps, source formula approximations or game writes.
The browser requests explicit candidate calculations on bridge-owned copies.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from forward_guide import guide
from forward_template import TEMPLATE

BRIDGE = "http://127.0.0.1:43127"
ROUTES = {
    "/api/context": "/maker/face/forward/context",
    "/api/evaluate": "/maker/face/forward/evaluate",
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def source_request(path, body):
    request = Request(
        BRIDGE + path, json.dumps(body, allow_nan=False).encode("utf-8"),
        {"Content-Type": "application/json"}, method="POST",
    )
    with urlopen(request, timeout=65) as response:
        return response.status, response.read()


class Handler(BaseHTTPRequestHandler):
    def send(self, status, content, kind="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def local_request(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in hosts and (
            origin is None or origin in {f"http://{host}" for host in hosts}
        )

    def do_GET(self):
        if not self.local_request():
            return self.send(403, b'{"error":"Loopback host/origin required"}')
        if self.path != "/":
            return self.send(404, b'{"error":"Unknown viewer route"}')
        self.send(200, TEMPLATE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        if not self.local_request():
            return self.send(403, b'{"error":"Loopback host/origin required"}')
        if self.path not in ROUTES and self.path != "/api/guide":
            return self.send(404, b'{"error":"Unknown viewer route"}')
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("Request size must be 1..65536 bytes")
            body = self.rfile.read(size)
            parsed = json.loads(body.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=invalid_constant)
            if not isinstance(parsed, dict):
                raise TypeError("JSON object required")
            if self.path == "/api/guide":
                def evaluate(candidate):
                    _, raw = source_request(ROUTES["/api/evaluate"], candidate)
                    return json.loads(raw)
                result = guide(getattr(self.server, "forward_context", None), parsed, evaluate)
                return self.send(200, json.dumps(result, allow_nan=False).encode("utf-8"))
            status, raw = source_request(ROUTES[self.path], parsed)
            result = json.loads(raw)
            if self.path == "/api/context":
                self.server.forward_context = result
            self.send(status, raw)
        except HTTPError as error:
            self.send(error.code, error.read())
        except (ValueError, TypeError, UnicodeError) as error:
            self.send(422, json.dumps({"error": str(error)}).encode())
        except (URLError, TimeoutError, OSError) as error:
            self.send(503, json.dumps({"error": str(error)}).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=43128)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be within 1..65535")
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Forward viewer: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
