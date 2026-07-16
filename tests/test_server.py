from __future__ import annotations

from hermes_plugins.nemotron_embed_test import server
from hermes_plugins.nemotron_embed_test.server import ServerManager
from hermes_plugins.nemotron_embed_test.settings import NemotronSettings


def _settings(tmp_path):
    return NemotronSettings.from_mapping({}, hermes_home=tmp_path)


def test_status_accepts_healthy_operator_managed_endpoint(tmp_path, monkeypatch):
    manager = ServerManager(_settings(tmp_path), state_dir=tmp_path / "state")
    monkeypatch.setattr(server.NemotronClient, "healthcheck", lambda _self: True)

    status = manager.status()

    assert status.running is False
    assert status.pid is None
    assert status.healthy is True


def test_managed_start_uses_explicit_embedding_runner(tmp_path, monkeypatch):
    captured = {}

    class _Process:
        pid = 4321

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _Process()

    monkeypatch.setattr(server.platform, "system", lambda: "Linux")
    monkeypatch.setattr(server.shutil, "which", lambda _name: "/usr/bin/uvx")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(server, "_pid_exists", lambda pid: pid == 4321)
    monkeypatch.setattr(server, "_process_start_time", lambda pid: 123.0 if pid == 4321 else None)
    monkeypatch.setattr(server.NemotronClient, "healthcheck", lambda _self: False)
    manager = ServerManager(_settings(tmp_path), state_dir=tmp_path / "state")

    status = manager.start(wait=False)

    command = captured["command"]
    assert command[:4] == ["/usr/bin/uvx", "--from", "vllm==0.25.0", "vllm"]
    assert command[command.index("--runner") + 1] == "pooling"
    assert command[command.index("--convert") + 1] == "embed"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["env"]["VLLM_NO_USAGE_STATS"] == "1"
    assert captured["kwargs"]["env"]["DO_NOT_TRACK"] == "1"
    assert status.running is True


def test_start_refuses_to_replace_healthy_operator_managed_endpoint(tmp_path, monkeypatch):
    manager = ServerManager(_settings(tmp_path), state_dir=tmp_path / "state")
    monkeypatch.setattr(server.NemotronClient, "healthcheck", lambda _self: True)

    import pytest

    with pytest.raises(RuntimeError, match="already healthy"):
        manager.start(wait=False)


def test_stale_recycled_pid_is_never_treated_as_managed(tmp_path, monkeypatch):
    manager = ServerManager(_settings(tmp_path), state_dir=tmp_path / "state")
    manager.state_dir.mkdir(parents=True)
    manager.state_path.write_text(
        '{"pid": 4321, "pid_start_time": 100.0}',
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "_pid_exists", lambda pid: pid == 4321)
    monkeypatch.setattr(server, "_process_start_time", lambda _pid: 200.0)
    monkeypatch.setattr(server.NemotronClient, "healthcheck", lambda _self: False)

    assert manager.status().running is False
    assert manager.stop() is False


def test_managed_start_rejects_endpoint_the_spawned_server_cannot_serve(tmp_path, monkeypatch):
    monkeypatch.setattr(server.NemotronClient, "healthcheck", lambda _self: False)
    monkeypatch.setattr(server.platform, "system", lambda: "Linux")
    https_settings = NemotronSettings.from_mapping(
        {"endpoint": "https://localhost:8089/v2/embed"},
        hermes_home=tmp_path,
    )
    wrong_path_settings = NemotronSettings.from_mapping(
        {"endpoint": "http://localhost:8089/not-embed"},
        hermes_home=tmp_path,
    )

    import pytest

    with pytest.raises(RuntimeError, match="http loopback"):
        ServerManager(https_settings, state_dir=tmp_path / "https").start(wait=False)
    with pytest.raises(RuntimeError, match="/v2/embed"):
        ServerManager(wrong_path_settings, state_dir=tmp_path / "path").start(wait=False)
