#!/usr/bin/env python3
"""Run a small, deterministic retrieval smoke eval against ``/v2/embed``.

This is an integration check for serving and prompt-prefix behavior, not a
claim about public benchmark performance. It intentionally uses only the
Python standard library so it can be piped over SSH to a model host.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"
ENDPOINT = "http://127.0.0.1:8089/v2/embed"

# Each case models a detail that an assistant may need to recover from an
# older turn. Distractors deliberately reuse some vocabulary without answering
# the query.
CASES = [
    {
        "id": "deploy-window",
        "query": "When is the production deployment window?",
        "relevant": "Production deploys are approved after 4 PM Central on Thursdays.",
        "distractor": "The staging deployment finished before lunch on Tuesday.",
    },
    {
        "id": "database-port",
        "query": "Which port did we choose for the local Postgres instance?",
        "relevant": "The local PostgreSQL container maps its database to port 55432.",
        "distractor": "The Redis development service listens on port 6380.",
    },
    {
        "id": "python-version",
        "query": "What Python version must the release workflow use?",
        "relevant": "The release workflow is pinned to Python 3.12 for reproducible builds.",
        "distractor": "The browser test matrix includes the latest Firefox release.",
    },
    {
        "id": "design-color",
        "query": "What accent color did the user request for warning badges?",
        "relevant": "Warning badges should use amber, while destructive actions remain red.",
        "distractor": "The navigation background uses a neutral slate color.",
    },
    {
        "id": "api-retry",
        "query": "How many retries should the payment client make?",
        "relevant": "The payment client retries transient failures twice with exponential backoff.",
        "distractor": "The analytics client batches up to two hundred events.",
    },
    {
        "id": "test-command",
        "query": "Which command verifies the Rust workspace before release?",
        "relevant": "Run cargo test --workspace --all-features before cutting the release.",
        "distractor": "Use npm run test:e2e for the web checkout flow.",
    },
    {
        "id": "customer-region",
        "query": "Where must the Acme customer data remain?",
        "relevant": "Acme requires all customer records and backups to remain in the EU region.",
        "distractor": "Northwind serves static images from a global CDN.",
    },
    {
        "id": "feature-flag",
        "query": "What is the feature flag for the new context engine?",
        "relevant": "The semantic context engine is guarded by the flag semantic_context_v2.",
        "distractor": "The legacy search screen uses the flag search_classic.",
    },
    {
        "id": "meeting-owner",
        "query": "Who owns the Friday incident review?",
        "relevant": "Maya owns the Friday incident review and will circulate the agenda.",
        "distractor": "Jon will take notes during Monday's product planning session.",
    },
    {
        "id": "cache-policy",
        "query": "How long should successful profile responses be cached?",
        "relevant": "Successful profile responses have a cache TTL of fifteen minutes.",
        "distractor": "Authentication failures are never cached.",
    },
    {
        "id": "backup-bucket",
        "query": "What bucket stores the nightly encrypted backups?",
        "relevant": "Nightly encrypted backups are written to the bucket ops-archive-primary.",
        "distractor": "Public screenshots are uploaded to the bucket marketing-assets.",
    },
    {
        "id": "accessibility-key",
        "query": "Which key should close the command palette?",
        "relevant": "Pressing Escape closes the command palette and returns focus to its trigger.",
        "distractor": "Pressing Enter runs the currently highlighted command.",
    },
]


def _embed(
    endpoint: str,
    model: str,
    input_type: str,
    texts: list[str],
    timeout: float,
    dimensions: int,
    api_key: str | None,
) -> tuple[list[list[float]], int, float]:
    payload = {
        "model": model,
        "input_type": input_type,
        "texts": texts,
        "embedding_types": ["float"],
        "truncate": "END",
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode())
    elapsed_ms = (time.perf_counter() - started) * 1000

    raw_vectors = data["embeddings"]["float"]
    if len(raw_vectors) != len(texts):
        raise RuntimeError("embedding response count does not match request")
    native_dimensions = len(raw_vectors[0]) if raw_vectors else 0
    vectors = [_normalize(vector[:dimensions]) for vector in raw_vectors]
    return vectors, native_dimensions, elapsed_ms


def _normalize(values: list[Any]) -> list[float]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if not vector or not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("embedding response contains an invalid vector")
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    documents: list[str] = []
    relevant_indices: list[int] = []
    for case in CASES:
        relevant_indices.append(len(documents))
        documents.extend((case["relevant"], case["distractor"]))

    document_vectors, native_dimensions, document_ms = _embed(
        args.endpoint,
        args.model,
        "document",
        documents,
        args.timeout,
        args.dimensions,
        api_key,
    )
    query_vectors, query_native_dimensions, query_ms = _embed(
        args.endpoint,
        args.model,
        "query",
        [case["query"] for case in CASES],
        args.timeout,
        args.dimensions,
        api_key,
    )
    if native_dimensions != query_native_dimensions:
        raise RuntimeError("query and document vector dimensions differ")

    ranks: list[int] = []
    margins: list[float] = []
    case_results: list[dict[str, Any]] = []
    for case, query_vector, relevant_index in zip(
        CASES, query_vectors, relevant_indices, strict=True
    ):
        scores = [_dot(query_vector, vector) for vector in document_vectors]
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        rank = order.index(relevant_index) + 1
        best_other = max(score for index, score in enumerate(scores) if index != relevant_index)
        margin = scores[relevant_index] - best_other
        ranks.append(rank)
        margins.append(margin)
        case_results.append(
            {
                "id": case["id"],
                "rank": rank,
                "relevant_score": round(scores[relevant_index], 6),
                "top_margin": round(margin, 6),
            }
        )

    count = len(ranks)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "label": args.label,
        "model": args.model,
        "endpoint_path": "/v2/embed",
        "cases": count,
        "corpus_documents": len(documents),
        "native_dimensions": native_dimensions,
        "evaluated_dimensions": len(document_vectors[0]),
        "metrics": {
            "recall_at_1": sum(rank <= 1 for rank in ranks) / count,
            "recall_at_3": sum(rank <= 3 for rank in ranks) / count,
            "mrr": statistics.fmean(1 / rank for rank in ranks),
            "mean_top_margin": statistics.fmean(margins),
        },
        "latency_ms": {
            "documents_batch": round(document_ms, 2),
            "queries_batch": round(query_ms, 2),
        },
        "results": case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", default="local")
    parser.add_argument(
        "--api-key-env",
        default="",
        help="Environment variable containing a bearer token (the value is never printed)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.dimensions <= 0:
        parser.error("--dimensions must be positive")

    result = evaluate(args)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
