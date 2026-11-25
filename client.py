"""Small stdlib client for vLLM's recommended Nemotron retrieval endpoint."""

from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request


class EmbeddingError(RuntimeError):
    """The local embedding service returned an unusable response."""


class NemotronClient:
    """Call a vLLM ``/v2/embed`` endpoint without adding dependencies."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout: float,
        dimensions: int,
        api_key: str | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.dimensions = dimensions
        self.api_key = api_key
        # Default transport ignores HTTP(S)_PROXY and refuses redirects so a
        # local embedding call cannot be replayed at another origin.
        self._opener = opener or _open_refusing_redirects

    def embed(self, input_type: str, texts: Sequence[str]) -> list[list[float]]:
        """Embed query or document text using NVIDIA's saved prompt metadata."""
        if input_type not in {"query", "document"}:
            raise ValueError("input_type must be 'query' or 'document'")
        if not texts:
            return []

        payload = {
            "model": self.model,
            "input_type": input_type,
            "texts": [str(text) for text in texts],
            "embedding_types": ["float"],
            "truncate": "END",
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = _http_error_detail(exc)
            raise EmbeddingError(f"embedding server returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise EmbeddingError(f"embedding server is unavailable: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EmbeddingError("embedding server returned invalid JSON") from exc

        try:
            vectors = data["embeddings"]["float"]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError("embedding response is missing embeddings.float") from exc
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError("embedding response count does not match request")
        return [self._slice_and_normalize(vector) for vector in vectors]

    def healthcheck(self) -> bool:
        """Probe the real retrieval path, not merely the server process."""
        try:
            return bool(self.embed("query", ["health check"]))
        except EmbeddingError:
            return False

    def _slice_and_normalize(self, raw: Any) -> list[float]:
        if not isinstance(raw, list) or not raw:
            raise EmbeddingError("embedding response contains an empty vector")
        try:
            vector = [float(value) for value in raw[: self.dimensions]]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("embedding response contains non-numeric values") from exc
        if not vector or not all(math.isfinite(value) for value in vector):
            raise EmbeddingError("embedding response contains invalid values")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            raise EmbeddingError("embedding response contains a zero vector")
        return [value / norm for value in vector]


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise EmbeddingError(
            f"embedding server returned HTTP {code} redirect; refusing to follow it"
        )


def _open_refusing_redirects(request: Request, timeout: float = 0):
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RejectRedirects,
    )
    return opener.open(request, timeout=timeout)


def _http_error_detail(exc: HTTPError) -> str:
    try:
        raw = exc.read()
    except Exception:
        raw = b""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")[:500]
    return str(raw)[:500]


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for normalized vectors."""
    if len(left) != len(right):
        return float("-inf")
    return sum(a * b for a, b in zip(left, right, strict=True))
