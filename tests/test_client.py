from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError

import pytest
from hermes_plugins.nemotron_embed_test.client import EmbeddingError, NemotronClient


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_v2_payload_and_dimension_normalization():
    captured = {}

    def opener(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response({"embeddings": {"float": [[3.0, 4.0, 100.0]]}})

    client = NemotronClient(
        endpoint="http://127.0.0.1:8000/v2/embed",
        model="example/model",
        timeout=7.5,
        dimensions=2,
        opener=opener,
    )
    vectors = client.embed("query", ["find this"])

    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert captured["payload"] == {
        "model": "example/model",
        "input_type": "query",
        "texts": ["find this"],
        "embedding_types": ["float"],
        "truncate": "END",
    }
    assert captured["timeout"] == 7.5


def test_optional_bearer_token_is_sent_without_entering_payload():
    captured = {}

    def opener(request, timeout):
        del timeout
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data)
        return _Response({"embeddings": {"float": [[1.0, 0.0]]}})

    client = NemotronClient(
        endpoint="http://127.0.0.1:8000/v2/embed",
        model="example/model",
        timeout=1,
        dimensions=2,
        api_key="secret-token",
        opener=opener,
    )

    client.embed("document", ["text"])

    assert captured["authorization"] == "Bearer secret-token"
    assert "secret-token" not in json.dumps(captured["payload"])


def test_http_errors_are_bounded_and_actionable():
    def opener(_request, timeout):
        del timeout
        raise HTTPError(
            url="http://localhost/v2/embed",
            code=503,
            msg="unavailable",
            hdrs=None,
            fp=io.BytesIO(b"server warming"),
        )

    client = NemotronClient(
        endpoint="http://127.0.0.1:8000/v2/embed",
        model="example/model",
        timeout=1,
        dimensions=2,
        opener=opener,
    )
    with pytest.raises(EmbeddingError, match="HTTP 503: server warming"):
        client.embed("document", ["text"])


def _start_server(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_server(server, thread) -> None:
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


def test_embed_ignores_proxy_env_and_uses_the_configured_endpoint(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            body = json.dumps({"embeddings": {"float": [[3.0, 4.0]]}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:9")
    server, thread = _start_server(Handler)
    try:
        port = server.server_address[1]
        client = NemotronClient(
            endpoint=f"http://127.0.0.1:{port}/v2/embed",
            model="example/model",
            timeout=2,
            dimensions=2,
        )
        assert client.embed("query", ["find this"])[0] == pytest.approx([0.6, 0.8])
    finally:
        _stop_server(server, thread)


def test_embed_refuses_to_follow_redirects():
    hits = {"target": 0}

    class Target(BaseHTTPRequestHandler):
        def do_POST(self):
            hits["target"] += 1
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            body = json.dumps({"embeddings": {"float": [[1.0, 0.0]]}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    class Redirect(BaseHTTPRequestHandler):
        location = ""

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self.send_response(307)
            self.send_header("Location", Redirect.location)
            self.end_headers()

        def log_message(self, fmt, *args):
            return

    target, target_thread = _start_server(Target)
    redirect, redirect_thread = _start_server(Redirect)
    Redirect.location = f"http://127.0.0.1:{target.server_address[1]}/stolen"
    try:
        client = NemotronClient(
            endpoint=f"http://127.0.0.1:{redirect.server_address[1]}/v2/embed",
            model="example/model",
            timeout=2,
            dimensions=2,
        )
        with pytest.raises(EmbeddingError, match="redirect"):
            client.embed("document", ["conversation text"])
        assert hits["target"] == 0
    finally:
        _stop_server(redirect, redirect_thread)
        _stop_server(target, target_thread)


def test_rejects_non_finite_vectors():
    client = NemotronClient(
        endpoint="http://127.0.0.1:8000/v2/embed",
        model="example/model",
        timeout=1,
        dimensions=2,
        opener=lambda *_args, **_kwargs: _Response(
            {"embeddings": {"float": [[1.0, float("nan")]]}}
        ),
    )
    with pytest.raises(EmbeddingError, match="invalid"):
        client.embed("query", ["text"])


@pytest.mark.parametrize("payload", [{}, {"embeddings": {}}, {"embeddings": {"float": []}}])
def test_rejects_malformed_responses(payload):
    client = NemotronClient(
        endpoint="http://127.0.0.1:8000/v2/embed",
        model="example/model",
        timeout=1,
        dimensions=2,
        opener=lambda *_args, **_kwargs: _Response(payload),
    )
    with pytest.raises(EmbeddingError):
        client.embed("query", ["text"])
