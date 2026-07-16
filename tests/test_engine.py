from __future__ import annotations

import json

from hermes_plugins.nemotron_embed_test.archive import ContextArchive
from hermes_plugins.nemotron_embed_test.client import EmbeddingError
from hermes_plugins.nemotron_embed_test.engine import NemotronEmbedEngine
from hermes_plugins.nemotron_embed_test.settings import NemotronSettings


class _SemanticClient:
    def embed(self, input_type, texts):
        if input_type == "query":
            return [[1.0, 0.0] if "alpha" in texts[0] else [0.0, 1.0]]
        return [[1.0, 0.0] if "alpha-relevant" in text else [0.0, 1.0] for text in texts]


class _UnavailableClient:
    def embed(self, _input_type, _texts):
        raise EmbeddingError("offline")


def _settings(tmp_path, **overrides):
    values = {
        "threshold": 0.75,
        "target_ratio": 0.35,
        "protect_first_turns": 1,
        "protect_last_turns": 1,
        "embedding_dimensions": 128,
        **overrides,
    }
    return NemotronSettings.from_mapping(values, hermes_home=tmp_path)


def _turn(label, fill="x"):
    return [
        {"role": "user", "content": f"{label} {fill * 700}"},
        {"role": "assistant", "content": f"answer for {label} {fill * 300}"},
    ]


def _messages():
    messages = [{"role": "system", "content": "stable system prompt"}]
    for label in (
        "opening",
        "alpha-relevant",
        "beta-archive",
        "gamma-archive",
        "delta-recent",
        "alpha current question",
    ):
        messages.extend(_turn(label))
    return messages


def test_semantic_compaction_keeps_relevant_atomic_turn_and_archives_others(tmp_path):
    settings = _settings(tmp_path)
    archive = ContextArchive(settings.archive_path)
    engine = NemotronEmbedEngine(settings, client=_SemanticClient(), archive=archive)
    engine.update_model("chat-model", context_length=4000)
    engine.on_session_start("session-1")

    result = engine.compress(_messages(), current_tokens=1800)
    rendered = json.dumps(result)

    assert result[0] == {"role": "system", "content": "stable system prompt"}
    assert "opening" in rendered
    assert "alpha-relevant" in rendered
    assert "alpha current question" in rendered
    assert "beta-archive" not in rendered
    assert engine.compression_count == 1
    assert archive.count("session-1") >= 1

    recovered = json.loads(
        engine.handle_tool_call(
            "nemotron_context_search",
            {"query": "beta archive", "limit": 3},
        )
    )
    assert recovered["success"] is True
    assert any("beta-archive" in item["content"] for item in recovered["results"])


def test_offline_server_uses_recency_fallback_without_breaking_turns(tmp_path):
    settings = _settings(tmp_path)
    engine = NemotronEmbedEngine(settings, client=_UnavailableClient())
    engine.update_model("chat-model", context_length=4000)
    engine.on_session_start("session-2")

    result = engine.compress(_messages(), current_tokens=1800)
    rendered = json.dumps(result)

    assert "delta-recent" in rendered
    assert "alpha current question" in rendered
    assert engine.get_status()["last_compaction_backend"] == "recency-fallback"
    assert len(result) % 2 == 1  # one system message plus complete user/assistant pairs


def test_usage_threshold_and_reset_contract(tmp_path):
    engine = NemotronEmbedEngine(_settings(tmp_path), client=_SemanticClient())
    engine.update_model("chat-model", context_length=10_000)
    engine.update_from_response({"input_tokens": 7600, "output_tokens": 200, "total_tokens": 7800})

    assert engine.should_compress() is True
    assert engine.last_completion_tokens == 200
    engine.on_session_start("session")
    engine.on_session_reset()
    assert engine.last_prompt_tokens == 0
    assert engine.compression_count == 0


def test_search_tool_rejects_empty_query(tmp_path):
    engine = NemotronEmbedEngine(_settings(tmp_path), client=_SemanticClient())
    response = json.loads(engine.handle_tool_call("nemotron_context_search", {}))
    assert response == {"success": False, "error": "query is required"}


def test_host_force_keyword_preserves_manual_focus_topic(tmp_path):
    engine = NemotronEmbedEngine(_settings(tmp_path), client=_SemanticClient())
    engine.update_model("chat-model", context_length=4000)
    engine.on_session_start("session")

    result = engine.compress(
        _messages(),
        current_tokens=1800,
        focus_topic="alpha historical decision",
        force=True,
    )

    assert "alpha-relevant" in json.dumps(result)


def test_context_search_survives_host_session_rotation_and_resume(tmp_path):
    settings = _settings(tmp_path)
    archive = ContextArchive(settings.archive_path)
    engine = NemotronEmbedEngine(settings, client=_UnavailableClient(), archive=archive)
    archive.store(
        session_id="parent",
        sequence=3,
        text="the staging codename is bluebird",
        messages=[{"role": "user", "content": "staging codename bluebird"}],
        embedding=None,
    )

    engine.on_session_start(
        "child",
        boundary_reason="compression",
        old_session_id="parent",
    )
    resumed = NemotronEmbedEngine(settings, client=_UnavailableClient(), archive=archive)
    resumed.on_session_start("child")
    response = json.loads(
        resumed.handle_tool_call("nemotron_context_search", {"query": "staging codename"})
    )

    assert response["success"] is True
    assert response["results"][0]["content"] == "the staging codename is bluebird"


def test_compaction_accounts_for_fixed_request_overhead(tmp_path):
    engine = NemotronEmbedEngine(_settings(tmp_path), client=_SemanticClient())
    engine.update_model("chat-model", context_length=4000)
    engine.on_session_start("session")
    messages = _messages()

    result = engine.compress(messages, current_tokens=2200)

    assert engine.get_status()["last_after_tokens"] <= int(4000 * 0.35)
    assert result is not messages


def test_compaction_returns_original_when_every_turn_fits(tmp_path):
    engine = NemotronEmbedEngine(_settings(tmp_path), client=_SemanticClient())
    engine.update_model("chat-model", context_length=20_000)
    engine.on_session_start("session")
    messages = _messages()

    result = engine.compress(messages, current_tokens=3000, force=True)

    assert result is messages
    assert engine.compression_count == 0
