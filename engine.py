"""Semantic context compaction backed by a local Nemotron Embed server."""

from __future__ import annotations

import copy
import json
import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.context_engine import ContextEngine

from .archive import ContextArchive
from .client import EmbeddingError, NemotronClient, dot_product
from .settings import NemotronSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnGroup:
    """An atomic user turn, including its assistant/tool chain."""

    sequence: int
    messages: list[dict[str, Any]]
    text: str
    estimated_tokens: int


class NemotronEmbedEngine(ContextEngine):
    """Keep query-relevant turns and archive evicted context for recall."""

    def __init__(
        self,
        settings: NemotronSettings,
        *,
        client: NemotronClient | None = None,
        archive: ContextArchive | None = None,
    ) -> None:
        self.settings = settings
        self.threshold_percent = settings.threshold_percent
        self.protect_first_n = settings.protect_first_turns
        self.protect_last_n = settings.protect_last_turns
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 0
        self.context_length = 0
        self.compression_count = 0
        self._session_id = ""
        self._last_before_tokens = 0
        self._last_after_tokens = 0
        self._last_backend = "not-run"
        self._client = client or NemotronClient(
            endpoint=settings.endpoint,
            model=settings.model,
            timeout=settings.request_timeout,
            dimensions=settings.embedding_dimensions,
            api_key=settings.api_key,
        )
        archive_path = settings.archive_path or Path("archive.db")
        self._archive = archive or ContextArchive(archive_path)

    @property
    def name(self) -> str:
        return "nemotron-embed"

    def __deepcopy__(self, memo: dict[int, Any]) -> NemotronEmbedEngine:
        """Create an agent-local engine without copying live client state."""
        clone = type(self)(self.settings)
        memo[id(self)] = clone
        for field in (
            "last_prompt_tokens",
            "last_completion_tokens",
            "last_total_tokens",
            "threshold_tokens",
            "context_length",
            "compression_count",
        ):
            setattr(clone, field, getattr(self, field))
        return clone

    def update_from_response(self, usage: dict[str, Any]) -> None:
        self.last_prompt_tokens = _usage_int(usage, "input_tokens", fallback="prompt_tokens")
        self.last_completion_tokens = _usage_int(
            usage, "output_tokens", fallback="completion_tokens"
        )
        self.last_total_tokens = _usage_int(usage, "total_tokens") or (
            self.last_prompt_tokens + self.last_completion_tokens
        )

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        current = self.last_prompt_tokens if prompt_tokens is None else prompt_tokens
        return bool(self.threshold_tokens and current >= self.threshold_tokens)

    def should_compress_preflight(self, messages: list[dict[str, Any]]) -> bool:
        return self.should_compress(_estimate_messages_tokens(messages))

    def has_content_to_compress(self, messages: list[dict[str, Any]]) -> bool:
        _preamble, groups = _group_turns(messages, self.settings.max_chunk_chars)
        protected = self.settings.protect_first_turns + self.settings.protect_last_turns
        return len(groups) > protected

    def compress(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
        force: bool = False,
        memory_context: str = "",
    ) -> list[dict[str, Any]]:
        """Select relevant atomic turns, archiving every evicted turn locally.

        ``memory_context`` is accepted so newer Hermes hosts can pass provider
        memory without a TypeError. It is not added to the retrieval query:
        that text is not one of the conversation turns being kept or archived.
        """
        del force, memory_context
        if not messages:
            return messages

        preamble, groups = _group_turns(messages, self.settings.max_chunk_chars)
        protected = self._protected_indices(len(groups))
        candidates = [index for index in range(len(groups)) if index not in protected]
        if not candidates:
            return messages

        message_tokens = _estimate_messages_tokens(messages)
        before_tokens = message_tokens if current_tokens is None else current_tokens
        # Hermes passes a full request estimate here, including the system
        # prompt and tool schemas. Those fixed tokens survive compaction and
        # must consume part of the target budget even though they are not in
        # ``messages``.
        fixed_overhead = max(0, before_tokens - message_tokens)
        target_tokens = self._target_tokens(before_tokens)
        protected_tokens = (
            fixed_overhead
            + _estimate_messages_tokens(preamble)
            + sum(groups[index].estimated_tokens for index in protected)
        )
        available = max(0, target_tokens - protected_tokens)

        query = (focus_topic or "").strip() or _latest_user_text(messages)
        document_vectors: list[list[float]] | None = None
        scores: dict[int, float] = {}
        try:
            query_vector = self._client.embed("query", [query])[0]
            candidate_texts = [groups[index].text for index in candidates]
            document_vectors = self._client.embed("document", candidate_texts)
            denominator = max(1, len(groups) - 1)
            for offset, index in enumerate(candidates):
                recency = index / denominator
                scores[index] = (
                    dot_product(query_vector, document_vectors[offset])
                    + self.settings.recency_weight * recency
                )
            self._last_backend = "nemotron"
        except (EmbeddingError, IndexError, ValueError) as exc:
            logger.warning(
                "Nemotron Embed unavailable during compaction; using recency fallback: %s",
                exc,
            )
            for index in candidates:
                scores[index] = float(index)
            self._last_backend = "recency-fallback"

        selected = set(protected)
        for index in sorted(candidates, key=lambda item: scores[item], reverse=True):
            cost = groups[index].estimated_tokens
            if cost <= available:
                selected.add(index)
                available -= cost

        # Returning a copy with every turn still present would make Hermes
        # treat this as a successful compaction boundary, rotating the session
        # and invalidating its prompt cache without saving a token.
        if len(selected) == len(groups):
            return messages

        kept_messages = list(preamble)
        for index in sorted(selected):
            kept_messages.extend(groups[index].messages)
        after_tokens = fixed_overhead + _estimate_messages_tokens(kept_messages)
        if after_tokens >= before_tokens:
            return messages

        vector_by_index = (
            {index: document_vectors[offset] for offset, index in enumerate(candidates)}
            if document_vectors is not None
            else {}
        )
        for index in candidates:
            if index in selected:
                continue
            group = groups[index]
            try:
                self._archive.store(
                    session_id=self._session_id or "unknown-session",
                    sequence=group.sequence,
                    text=group.text,
                    messages=group.messages,
                    embedding=vector_by_index.get(index),
                )
            except (OSError, sqlite3.Error) as exc:
                logger.warning("Could not archive compacted context: %s", exc)

        self.compression_count += 1
        self._last_before_tokens = before_tokens
        self._last_after_tokens = after_tokens
        return copy.deepcopy(kept_messages)

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        if kwargs.get("boundary_reason") == "compression":
            old_session_id = str(kwargs.get("old_session_id") or "")
            try:
                self._archive.link_sessions(
                    parent_session_id=old_session_id,
                    child_session_id=session_id,
                )
            except (OSError, sqlite3.Error) as exc:
                logger.warning("Could not record context archive lineage: %s", exc)
        self._session_id = session_id or "unknown-session"

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._session_id = ""
        self._last_before_tokens = 0
        self._last_after_tokens = 0
        self._last_backend = "not-run"

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "nemotron_context_search",
                "description": (
                    "Search older conversation turns archived by the active "
                    "Nemotron context engine. Use when relevant details may have "
                    "been removed by context compaction."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Meaning or detail to recover.",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 8,
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

    def handle_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        del kwargs
        if name != "nemotron_context_search":
            return super().handle_tool_call(name, args)
        query = str(args.get("query") or "").strip()
        if not query:
            return json.dumps({"success": False, "error": "query is required"})
        raw_limit = args.get("limit", 3)
        if isinstance(raw_limit, bool):
            limit = 3
        else:
            try:
                limit = max(1, min(int(raw_limit), 8))
            except (TypeError, ValueError):
                limit = 3

        query_vector: Sequence[float] | None = None
        backend = "lexical"
        try:
            query_vector = self._client.embed("query", [query])[0]
            backend = "nemotron"
        except (EmbeddingError, IndexError, ValueError):
            pass
        try:
            results = self._archive.search(
                session_id=self._session_id or "unknown-session",
                query=query,
                embedding=query_vector,
                limit=limit,
            )
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Could not search archived context: %s", exc)
            return json.dumps({"success": False, "error": "context archive is unavailable"})
        return json.dumps(
            {
                "success": True,
                "backend": backend,
                "results": [
                    {
                        "chunk_id": item.chunk_id,
                        "sequence": item.sequence,
                        "score": round(item.score, 4),
                        "content": item.text[:4000],
                        "truncated": len(item.text) > 4000,
                    }
                    for item in results
                ],
            },
            ensure_ascii=False,
        )

    def get_status(self) -> dict[str, Any]:
        status = super().get_status()
        try:
            archived_chunks = self._archive.count(self._session_id or None)
        except (OSError, sqlite3.Error):
            archived_chunks = None
        status.update(
            {
                "engine": self.name,
                "embedding_model": self.settings.model,
                "last_compaction_backend": self._last_backend,
                "last_before_tokens": self._last_before_tokens,
                "last_after_tokens": self._last_after_tokens,
                "archived_chunks": archived_chunks,
            }
        )
        return status

    def _protected_indices(self, group_count: int) -> set[int]:
        first = range(min(self.settings.protect_first_turns, group_count))
        last_start = max(0, group_count - self.settings.protect_last_turns)
        return set(first) | set(range(last_start, group_count))

    def _target_tokens(self, before_tokens: int) -> int:
        basis = self.context_length or before_tokens
        return max(1024, int(basis * self.settings.target_ratio))


def _group_turns(
    messages: list[dict[str, Any]],
    max_chunk_chars: int,
) -> tuple[list[dict[str, Any]], list[TurnGroup]]:
    preamble: list[dict[str, Any]] = []
    raw_groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "user":
            if current:
                raw_groups.append(current)
            current = [message]
        elif current:
            current.append(message)
        else:
            preamble.append(message)
    if current:
        raw_groups.append(current)

    groups = []
    for sequence, group_messages in enumerate(raw_groups):
        groups.append(
            TurnGroup(
                sequence=sequence,
                messages=copy.deepcopy(group_messages),
                text=_render_group(group_messages)[:max_chunk_chars],
                estimated_tokens=_estimate_messages_tokens(group_messages),
            )
        )
    return copy.deepcopy(preamble), groups


def _render_group(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role") or "unknown")
        content = _content_text(message.get("content"))
        tool_calls = message.get("tool_calls")
        if tool_calls:
            content = f"{content}\ntool_calls: {_json_text(tool_calls)}"
        parts.append(f"{role}: {content}".strip())
    return "\n".join(parts)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            text = _content_text(message.get("content")).strip()
            if text:
                return text
    return "continue the current task"


def _estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    if not messages:
        return 0
    try:
        serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        serialized = str(messages)
    return max(1, len(serialized) // 4 + len(messages) * 4)


def _usage_int(
    usage: dict[str, Any],
    key: str,
    *,
    fallback: str | None = None,
) -> int:
    raw = usage.get(key)
    if raw is None and fallback:
        raw = usage.get(fallback)
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0
