"""Profile-local archive for context evicted by semantic compaction."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import dot_product


@dataclass(frozen=True)
class ArchivedChunk:
    chunk_id: str
    session_id: str
    sequence: int
    text: str
    messages: list[dict[str, Any]]
    score: float = 0.0


class ContextArchive:
    """SQLite archive with vector and lexical retrieval."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def store(
        self,
        *,
        session_id: str,
        sequence: int,
        text: str,
        messages: list[dict[str, Any]],
        embedding: Sequence[float] | None,
    ) -> str:
        canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        chunk_id = hashlib.sha256(f"{session_id}\0{sequence}\0{canonical}".encode()).hexdigest()[
            :20
        ]
        blob = _vector_to_blob(embedding) if embedding is not None else None
        self._ensure_parent()
        with self._connect() as conn:
            self._create_schema(conn)
            conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, session_id, sequence, created_at, text,
                    messages_json, embedding, dimensions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    text = excluded.text,
                    messages_json = excluded.messages_json,
                    embedding = COALESCE(excluded.embedding, chunks.embedding),
                    dimensions = MAX(excluded.dimensions, chunks.dimensions)
                """,
                (
                    chunk_id,
                    session_id,
                    sequence,
                    time.time(),
                    text,
                    canonical,
                    blob,
                    len(embedding or ()),
                ),
            )
        return chunk_id

    def link_sessions(self, *, parent_session_id: str, child_session_id: str) -> None:
        """Record a Hermes compression continuation for lineage-aware recall."""
        if not parent_session_id or not child_session_id or parent_session_id == child_session_id:
            return
        self._ensure_parent()
        with self._connect() as conn:
            self._create_schema(conn)
            conn.execute(
                """
                INSERT INTO session_lineage (child_session_id, parent_session_id)
                VALUES (?, ?)
                ON CONFLICT(child_session_id) DO UPDATE SET
                    parent_session_id = excluded.parent_session_id
                """,
                (child_session_id, parent_session_id),
            )

    def search(
        self,
        *,
        session_id: str,
        query: str,
        embedding: Sequence[float] | None,
        limit: int,
    ) -> list[ArchivedChunk]:
        if not self.path.exists():
            return []
        with self._connect() as conn:
            self._create_schema(conn)
            rows = conn.execute(
                """
                WITH RECURSIVE lineage(session_id) AS (
                    VALUES (?)
                    UNION
                    SELECT links.parent_session_id
                      FROM session_lineage AS links
                      JOIN lineage ON links.child_session_id = lineage.session_id
                )
                SELECT chunk_id, session_id, sequence, text, messages_json,
                       embedding, dimensions
                  FROM chunks
                 WHERE session_id IN (SELECT session_id FROM lineage)
                 ORDER BY sequence ASC
                """,
                (session_id,),
            ).fetchall()

        query_terms = _terms(query)
        ranked: list[ArchivedChunk] = []
        for row in rows:
            vector_score: float | None = None
            if embedding is not None and row[5] is not None:
                stored = _blob_to_vector(row[5], row[6])
                if len(stored) == len(embedding):
                    vector_score = dot_product(embedding, stored)
            lexical_score = _lexical_score(query_terms, _terms(row[3]))
            score = vector_score if vector_score is not None else lexical_score
            # A negative cosine is still a valid nearest-neighbor score. Only
            # suppress rows when retrieval has fallen back to lexical matching
            # and there is no term overlap at all.
            if vector_score is None and lexical_score <= 0 and query_terms:
                continue
            try:
                messages = json.loads(row[4])
            except (TypeError, json.JSONDecodeError):
                messages = []
            ranked.append(
                ArchivedChunk(
                    chunk_id=row[0],
                    session_id=row[1],
                    sequence=row[2],
                    text=row[3],
                    messages=messages,
                    score=score,
                )
            )
        ranked.sort(key=lambda item: (item.score, item.sequence), reverse=True)
        return ranked[: max(1, min(limit, 20))]

    def count(self, session_id: str | None = None) -> int:
        if not self.path.exists():
            return 0
        with self._connect() as conn:
            self._create_schema(conn)
            if session_id:
                row = conn.execute(
                    """
                    WITH RECURSIVE lineage(session_id) AS (
                        VALUES (?)
                        UNION
                        SELECT links.parent_session_id
                          FROM session_lineage AS links
                          JOIN lineage ON links.child_session_id = lineage.session_id
                    )
                    SELECT COUNT(*)
                      FROM chunks
                     WHERE session_id IN (SELECT session_id FROM lineage)
                    """,
                    (session_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            self.path.chmod(0o600)
        except OSError:
            conn.close()
            raise
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                created_at REAL NOT NULL,
                text TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                embedding BLOB,
                dimensions INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_session_sequence
            ON chunks(session_id, sequence)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_lineage (
                child_session_id TEXT PRIMARY KEY,
                parent_session_id TEXT NOT NULL
            )
            """
        )


def _vector_to_blob(vector: Sequence[float]) -> bytes:
    values = array("f", (float(value) for value in vector))
    return values.tobytes()


def _blob_to_vector(blob: bytes, dimensions: int) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    if dimensions > 0 and len(values) != dimensions:
        return []
    return values.tolist()


def _terms(text: str) -> set[str]:
    return {token.casefold() for token in _word_tokens(text) if len(token) >= 2}


def _word_tokens(text: str) -> Iterable[str]:
    current: list[str] = []
    for char in text:
        if char.isalnum() or char == "_":
            current.append(char)
        elif current:
            yield "".join(current)
            current = []
    if current:
        yield "".join(current)


def _lexical_score(query: set[str], document: set[str]) -> float:
    if not query:
        return 0.0
    return len(query & document) / len(query)
