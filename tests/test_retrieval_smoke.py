from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "retrieval_smoke",
    Path(__file__).resolve().parents[1] / "evals" / "retrieval_smoke.py",
)
assert _SPEC is not None and _SPEC.loader is not None
retrieval_smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(retrieval_smoke)


def test_score_retrieval_reports_rank_and_mrr():
    scored = retrieval_smoke.score_retrieval(
        query_vectors=[[0.0, 1.0]],
        document_vectors=[[0.0, 1.0], [1.0, 0.0]],
        relevant_indices=[1],
        case_ids=["second"],
    )

    assert scored["results"][0]["rank"] == 2
    assert scored["metrics"]["recall_at_1"] == 0
    assert scored["metrics"]["recall_at_3"] == 1
    assert scored["metrics"]["mrr"] == 0.5


def test_score_retrieval_ranks_the_matching_vector_first():
    scored = retrieval_smoke.score_retrieval(
        query_vectors=[[1.0, 0.0], [0.0, 1.0]],
        document_vectors=[[1.0, 0.0], [0.0, 1.0]],
        relevant_indices=[0, 1],
        case_ids=["a", "b"],
    )

    assert scored["metrics"]["recall_at_1"] == 1
    assert scored["metrics"]["mrr"] == 1
    assert [item["rank"] for item in scored["results"]] == [1, 1]


def test_eval_embed_refuses_redirects():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self.send_response(307)
            self.send_header("Location", "http://127.0.0.1:9/stolen")
            self.end_headers()

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="redirect"):
            retrieval_smoke._embed(
                f"http://127.0.0.1:{server.server_address[1]}/v2/embed",
                "example/model",
                "query",
                ["text"],
                2,
                2,
                None,
            )
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_eval_embed_reads_float_vectors():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            assert payload["input_type"] == "document"
            body = json.dumps({"embeddings": {"float": [[3.0, 4.0, 0.0]]}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        vectors, native, _elapsed = retrieval_smoke._embed(
            f"http://127.0.0.1:{server.server_address[1]}/v2/embed",
            "example/model",
            "document",
            ["passage"],
            2,
            2,
            None,
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert native == 3
    assert vectors[0] == pytest.approx([0.6, 0.8])
