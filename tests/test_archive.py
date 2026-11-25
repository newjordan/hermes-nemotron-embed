from __future__ import annotations

import stat

from hermes_plugins.nemotron_embed_test.archive import ContextArchive


def test_archive_is_idempotent_and_vector_searches_current_session(tmp_path):
    archive = ContextArchive(tmp_path / "archive.db")
    messages = [{"role": "user", "content": "database migration details"}]

    first = archive.store(
        session_id="s1",
        sequence=2,
        text="database migration details",
        messages=messages,
        embedding=[1.0, 0.0],
    )
    second = archive.store(
        session_id="s1",
        sequence=2,
        text="database migration details",
        messages=messages,
        embedding=[1.0, 0.0],
    )
    archive.store(
        session_id="s2",
        sequence=1,
        text="unrelated private session",
        messages=[{"role": "user", "content": "other"}],
        embedding=[1.0, 0.0],
    )

    assert first == second
    assert archive.count("s1") == 1
    results = archive.search(
        session_id="s1",
        query="migration",
        embedding=[1.0, 0.0],
        limit=3,
    )
    assert [item.chunk_id for item in results] == [first]
    assert results[0].messages == messages


def test_archive_falls_back_to_lexical_search(tmp_path):
    archive = ContextArchive(tmp_path / "archive.db")
    archive.store(
        session_id="s1",
        sequence=0,
        text="fixed a websocket reconnect race",
        messages=[{"role": "assistant", "content": "fixed it"}],
        embedding=None,
    )

    results = archive.search(
        session_id="s1",
        query="websocket race",
        embedding=None,
        limit=3,
    )
    assert len(results) == 1
    assert results[0].score == 1.0


def test_vector_search_keeps_valid_negative_cosine_matches(tmp_path):
    archive = ContextArchive(tmp_path / "archive.db")
    chunk_id = archive.store(
        session_id="s1",
        sequence=0,
        text="semantically nearest without lexical overlap",
        messages=[{"role": "assistant", "content": "nearest context"}],
        embedding=[-1.0, 0.0],
    )

    results = archive.search(
        session_id="s1",
        query="completely different terms",
        embedding=[1.0, 0.0],
        limit=3,
    )

    assert [item.chunk_id for item in results] == [chunk_id]
    assert results[0].score == -1.0


def test_archive_database_is_owner_only(tmp_path):
    archive = ContextArchive(tmp_path / "state" / "archive.db")
    archive.store(
        session_id="s1",
        sequence=0,
        text="private conversation detail",
        messages=[{"role": "user", "content": "private conversation detail"}],
        embedding=None,
    )

    assert stat.S_IMODE(archive.path.stat().st_mode) == 0o600


def test_archive_search_follows_compression_session_lineage(tmp_path):
    archive = ContextArchive(tmp_path / "archive.db")
    ancestor_chunk = archive.store(
        session_id="parent",
        sequence=2,
        text="the deployment region is us-central1",
        messages=[{"role": "user", "content": "use us-central1"}],
        embedding=None,
    )
    archive.link_sessions(parent_session_id="parent", child_session_id="child")
    archive.link_sessions(parent_session_id="child", child_session_id="grandchild")

    results = archive.search(
        session_id="grandchild",
        query="deployment region",
        embedding=None,
        limit=3,
    )

    assert [item.chunk_id for item in results] == [ancestor_chunk]
    assert archive.count("grandchild") == 1


def test_archive_replaces_vector_when_dimensions_shrink(tmp_path):
    archive = ContextArchive(tmp_path / "archive.db")
    messages = [{"role": "user", "content": "zzzz unique"}]
    archive.store(
        session_id="s1",
        sequence=1,
        text="zzzz unique",
        messages=messages,
        embedding=[1.0, 0.0, 0.0],
    )
    archive.store(
        session_id="s1",
        sequence=1,
        text="zzzz unique",
        messages=messages,
        embedding=[0.0, 1.0],
    )

    results = archive.search(
        session_id="s1",
        query="qqqq other",
        embedding=[0.0, 1.0],
        limit=3,
    )

    assert len(results) == 1
    assert results[0].score == 1.0


def test_archive_keeps_existing_vector_when_update_has_none(tmp_path):
    archive = ContextArchive(tmp_path / "archive.db")
    messages = [{"role": "user", "content": "alpha"}]
    archive.store(
        session_id="s1",
        sequence=1,
        text="alpha",
        messages=messages,
        embedding=[1.0, 0.0],
    )
    archive.store(
        session_id="s1",
        sequence=1,
        text="alpha",
        messages=messages,
        embedding=None,
    )

    results = archive.search(
        session_id="s1",
        query="zzzz",
        embedding=[1.0, 0.0],
        limit=3,
    )

    assert len(results) == 1
    assert results[0].score == 1.0


def test_lineage_cycle_does_not_expose_the_other_session(tmp_path):
    archive = ContextArchive(tmp_path / "archive.db")
    archive.store(
        session_id="parent",
        sequence=0,
        text="alphaunique",
        messages=[{"role": "user", "content": "alphaunique"}],
        embedding=None,
    )
    archive.store(
        session_id="child",
        sequence=0,
        text="betaunique",
        messages=[{"role": "user", "content": "betaunique"}],
        embedding=None,
    )
    archive.link_sessions(parent_session_id="parent", child_session_id="child")
    archive.link_sessions(parent_session_id="child", child_session_id="parent")

    parent_view = archive.search(
        session_id="parent",
        query="betaunique",
        embedding=None,
        limit=3,
    )
    child_view = archive.search(
        session_id="child",
        query="alphaunique",
        embedding=None,
        limit=3,
    )

    assert parent_view == []
    assert [item.text for item in child_view] == ["alphaunique"]


def test_lexical_fallback_matches_unspaced_cjk_inside_a_longer_turn(tmp_path):
    archive = ContextArchive(tmp_path / "archive.db")
    archive.store(
        session_id="s1",
        sequence=0,
        text="数据库迁移详情已经记录",
        messages=[{"role": "user", "content": "数据库迁移详情已经记录"}],
        embedding=None,
    )

    results = archive.search(
        session_id="s1",
        query="迁移",
        embedding=None,
        limit=3,
    )

    assert len(results) == 1
    assert results[0].score > 0


def test_archive_directory_and_sidecars_are_owner_only(tmp_path):
    archive = ContextArchive(tmp_path / "state" / "archive.db")
    archive.store(
        session_id="s1",
        sequence=0,
        text="private conversation detail",
        messages=[{"role": "user", "content": "private conversation detail"}],
        embedding=[0.0, 1.0],
    )

    assert stat.S_IMODE(archive.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(archive.path.parent.stat().st_mode) == 0o700
    for suffix in ("-wal", "-shm"):
        sidecar = archive.path.with_name(archive.path.name + suffix)
        if sidecar.exists():
            assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600


def test_archive_lineage_does_not_leak_into_unrelated_sessions(tmp_path):
    archive = ContextArchive(tmp_path / "archive.db")
    archive.store(
        session_id="parent",
        sequence=0,
        text="private parent detail",
        messages=[{"role": "user", "content": "private parent detail"}],
        embedding=None,
    )
    archive.link_sessions(parent_session_id="parent", child_session_id="child")

    assert (
        archive.search(
            session_id="unrelated",
            query="private detail",
            embedding=None,
            limit=3,
        )
        == []
    )
