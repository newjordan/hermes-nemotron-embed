from __future__ import annotations

import io
import json
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
