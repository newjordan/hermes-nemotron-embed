"""Profile-local archive for context evicted by semantic compaction."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import dot_product

logger = logging.getLogger(__name__)


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
        canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
        chunk_id = hashlib.sha256(f"{session_id}\0{sequence}\0{canonical}".encode()).hexdigest()[
            :20
        ]
        # An empty vector must not overwrite a previously stored embedding.
        blob = _vector_to_blob(embedding) if embedding else None
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
                    dimensions = CASE
                        WHEN excluded.embedding IS NOT NULL THEN excluded.dimensions
                        ELSE chunks.dimensions
                    END
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
            # Walk the parent's ancestors. If the child is already above the
            # parent, recording the reverse edge would make both sessions
            # searchable from each other.
            if _ancestors_include(conn, start=parent_session_id, target=child_session_id):
                logger.warning(
                    "Refusing session lineage link that would cycle: %s -> %s",
                    child_session_id,
                    parent_session_id,
                )
                return
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
        parent = self.path.parent
        created = not parent.exists()
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Tighten the directory we just created, and the default plugin
        # state directory even when an older install left it world-readable.
        # Leave an operator-chosen parent that already existed alone.
        if created or parent.name == "nemotron-embed":
            try:
                parent.chmod(0o700)
            except OSError:
                logger.warning("Could not restrict permissions on %s", parent)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA secure_delete=ON")
            _chmod_private(self.path)
            for suffix in ("-wal", "-shm"):
                _chmod_private(self.path.with_name(self.path.name + suffix), required=False)
        except Exception:
            conn.close()
            raise
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


def _ancestors_include(conn: sqlite3.Connection, *, start: str, target: str) -> bool:
    row = conn.execute(
        """
        WITH RECURSIVE ancestors(session_id) AS (
            VALUES (?)
            UNION
            SELECT links.parent_session_id
              FROM session_lineage AS links
              JOIN ancestors ON links.child_session_id = ancestors.session_id
        )
        SELECT 1 FROM ancestors WHERE session_id = ? LIMIT 1
        """,
        (start, target),
    ).fetchone()
    return row is not None


def _chmod_private(path: Path, *, required: bool = True) -> None:
    try:
        if path.exists():
            path.chmod(0o600)
        elif required:
            raise OSError(f"missing private file: {path}")
    except OSError:
        if required:
            raise


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
    """Index ASCII words and overlapping bigrams for unspaced scripts.

    Lexical search is the fallback when the embedding server is down.
    Chinese and Japanese turns have no spaces, so treating a whole sentence
    as one token makes a short query miss the archived text.
    """
    terms: set[str] = set()
    for token in _word_tokens(text):
        folded = token.casefold()
        if folded.isascii():
            if len(folded) >= 2:
                terms.add(folded)
            continue
        if len(folded) == 1:
            terms.add(folded)
            continue
        terms.update(folded[index : index + 2] for index in range(len(folded) - 1))
    return terms


def _word_tokens(text: str) -> Iterable[str]:
    current: list[str] = []
    current_ascii: bool | None = None
    for char in text:
        if char.isalnum() or char == "_":
            is_ascii = char.isascii()
            if current and current_ascii != is_ascii:
                yield "".join(current)
                current = []
            current.append(char)
            current_ascii = is_ascii
        elif current:
            yield "".join(current)
            current = []
            current_ascii = None
    if current:
        yield "".join(current)


def _lexical_score(query: set[str], document: set[str]) -> float:
    if not query:
        return 0.0
    return len(query & document) / len(query)
