"""Deterministic end-to-end compaction smoke benchmark."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .archive import ContextArchive
from .client import NemotronClient
from .engine import NemotronEmbedEngine, _estimate_messages_tokens
from .settings import NemotronSettings

_RELEVANT_FACT = "The production database must remain in us-central1."


def run_compaction_benchmark(
    settings: NemotronSettings,
    *,
    client: NemotronClient | None = None,
) -> dict[str, Any]:
    """Exercise selection, reduction, archival, and recovery as one flow."""
    messages = _benchmark_messages()
    message_tokens = _estimate_messages_tokens(messages)
    fixed_request_tokens = 300
    before_tokens = message_tokens + fixed_request_tokens

    with tempfile.TemporaryDirectory(prefix="nemotron-embed-benchmark-") as temp_dir:
        archive = ContextArchive(Path(temp_dir) / "archive.db")
        engine = NemotronEmbedEngine(settings, client=client, archive=archive)
        engine.update_model("benchmark-chat-model", context_length=8_000)
        engine.on_session_start("benchmark-session")
        compacted = engine.compress(messages, current_tokens=before_tokens)

        rendered = json.dumps(compacted, ensure_ascii=False)
        archived = archive.count("benchmark-session")
        omitted_labels = [label for label in _candidate_labels() if label not in rendered]
        recovery_results: list[dict[str, Any]] = []
        if omitted_labels:
            recovery = json.loads(
                engine.handle_tool_call(
                    "nemotron_context_search",
                    {"query": omitted_labels[0], "limit": 3},
                )
            )
            recovery_results = recovery.get("results", [])

        after_tokens = engine.get_status()["last_after_tokens"]
        atomic_turns = _has_atomic_user_assistant_turns(compacted)
        relevant_retained = _RELEVANT_FACT in rendered
        archive_recovery = bool(recovery_results)
        made_progress = 0 < after_tokens < before_tokens
        passed = all(
            (
                made_progress,
                relevant_retained,
                archived > 0,
                archive_recovery,
                atomic_turns,
            )
        )
        return {
            "schema_version": 1,
            "model": settings.model,
            "endpoint": settings.endpoint,
            "input_request_tokens_estimated": before_tokens,
            "output_request_tokens_estimated": after_tokens,
            "token_reduction_ratio": round(1 - after_tokens / before_tokens, 4),
            "input_turns": sum(message.get("role") == "user" for message in messages),
            "output_turns": sum(message.get("role") == "user" for message in compacted),
            "archived_chunks": archived,
            "relevant_turn_retained": relevant_retained,
            "archive_recovery_succeeded": archive_recovery,
            "atomic_turns_valid": atomic_turns,
            "passed": passed,
        }


def _benchmark_messages() -> list[dict[str, Any]]:
    turns = [
        ("opening", "We are preparing a database migration plan."),
        ("region", _RELEVANT_FACT),
        ("cache", "Profile responses use a fifteen minute cache TTL."),
        ("port", "The local Redis service listens on port 6380."),
        ("owner", "Maya owns the Friday incident review."),
        ("color", "Warning badges use amber rather than red."),
        ("tests", "Release verification runs cargo test --workspace."),
        ("backup", "Encrypted backups use the ops-archive-primary bucket."),
        ("deploy", "Production deploys happen after 4 PM Central on Thursday."),
        ("question", "Which region must host the production database?"),
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Keep operational decisions accurate."}
    ]
    for label, fact in turns:
        messages.extend(
            (
                {
                    "role": "user",
                    "content": f"[{label}] {fact} " + "context " * 120,
                },
                {
                    "role": "assistant",
                    "content": f"Recorded {label}. " + "acknowledged " * 35,
                },
            )
        )
    return messages


def _candidate_labels() -> list[str]:
    return ["[cache]", "[port]", "[owner]", "[color]", "[tests]"]


def _has_atomic_user_assistant_turns(messages: list[dict[str, Any]]) -> bool:
    non_system = [message for message in messages if message.get("role") != "system"]
    return (
        bool(non_system)
        and len(non_system) % 2 == 0
        and all(
            non_system[index].get("role") == "user"
            and non_system[index + 1].get("role") == "assistant"
            for index in range(0, len(non_system), 2)
        )
    )
