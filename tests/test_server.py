from __future__ import annotations

import json
import os
import signal
import stat
import subprocess

import pytest
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
    monkeypatch.setenv("NEMOTRON_EMBED_API_KEY", "super-secret-value")
    manager = ServerManager(_settings(tmp_path), state_dir=tmp_path / "state")

    status = manager.start(wait=False)

    command = captured["command"]
    assert command[:4] == ["/usr/bin/uvx", "--from", "vllm==0.25.0", "vllm"]
    assert "--convert" not in command
    assert command[command.index("--runner") + 1] == "pooling"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8089"
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["env"]["VLLM_NO_USAGE_STATS"] == "1"
    assert captured["kwargs"]["env"]["DO_NOT_TRACK"] == "1"
    assert captured["kwargs"]["env"]["VLLM_API_KEY"] == "super-secret-value"
    assert "super-secret-value" not in manager.state_path.read_text(encoding="utf-8")
    assert status.running is True
    assert stat.S_IMODE(manager.state_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(manager.log_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(manager.state_dir.stat().st_mode) == 0o700


def test_start_refuses_to_replace_healthy_operator_managed_endpoint(tmp_path, monkeypatch):
    manager = ServerManager(_settings(tmp_path), state_dir=tmp_path / "state")
    monkeypatch.setattr(server.NemotronClient, "healthcheck", lambda _self: True)

    with pytest.raises(RuntimeError, match="already healthy"):
        manager.start(wait=False)


def test_managed_start_requires_linux(tmp_path, monkeypatch):
    monkeypatch.setattr(server.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(server.NemotronClient, "healthcheck", lambda _self: False)
    manager = ServerManager(_settings(tmp_path), state_dir=tmp_path / "state")

    with pytest.raises(RuntimeError, match="Linux"):
        manager.start(wait=False)


def test_managed_start_rejects_non_loopback_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(server.platform, "system", lambda: "Linux")
    monkeypatch.setattr(server.NemotronClient, "healthcheck", lambda _self: False)
    settings = NemotronSettings.from_mapping(
        {"endpoint": "http://example.com/v2/embed"},
        hermes_home=tmp_path,
    )

    with pytest.raises(RuntimeError, match="loopback"):
        ServerManager(settings, state_dir=tmp_path / "state").start(wait=False)


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

    with pytest.raises(RuntimeError, match="http loopback"):
        ServerManager(https_settings, state_dir=tmp_path / "https").start(wait=False)
    with pytest.raises(RuntimeError, match="/v2/embed"):
        ServerManager(wrong_path_settings, state_dir=tmp_path / "path").start(wait=False)


def test_proc_start_time_matches_psutil_for_a_live_process():
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        via_psutil = server._process_start_time(proc.pid)
        via_proc = server._linux_proc_start_time(proc.pid)
        assert via_psutil is not None
        assert via_proc is not None
        assert abs(via_psutil - via_proc) < 1
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2)


def test_stop_terminates_the_recorded_process(tmp_path, monkeypatch):
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        manager = ServerManager(_settings(tmp_path), state_dir=tmp_path / "state")
        manager.state_dir.mkdir(parents=True)
        manager.state_path.write_text(
            json.dumps(
                {
                    "pid": proc.pid,
                    "pid_start_time": server._process_start_time(proc.pid),
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(server.NemotronClient, "healthcheck", lambda _self: False)

        assert manager.stop(timeout=2) is True
        proc.wait(timeout=2)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2)


def test_stop_falls_back_to_the_process_group_without_psutil(tmp_path, monkeypatch):
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        manager = ServerManager(_settings(tmp_path), state_dir=tmp_path / "state")
        manager.state_dir.mkdir(parents=True)
        manager.state_path.write_text(
            json.dumps(
                {
                    "pid": proc.pid,
                    "pid_start_time": server._process_start_time(proc.pid),
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(server.NemotronClient, "healthcheck", lambda _self: False)

        def missing_psutil(_pid, *, timeout):
            del timeout
            raise ImportError("psutil missing")

        monkeypatch.setattr(server, "_terminate_with_psutil", missing_psutil)
        assert manager.stop(timeout=2) is True
        proc.wait(timeout=2)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2)


def test_stop_does_not_signal_a_recycled_pid(tmp_path, monkeypatch):
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        manager = ServerManager(_settings(tmp_path), state_dir=tmp_path / "state")
        manager.state_dir.mkdir(parents=True)
        started = server._process_start_time(proc.pid)
        assert started is not None
        manager.state_path.write_text(
            json.dumps({"pid": proc.pid, "pid_start_time": started + 50}),
            encoding="utf-8",
        )
        monkeypatch.setattr(server.NemotronClient, "healthcheck", lambda _self: False)

        assert manager.stop(timeout=2) is False
        assert proc.poll() is None
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2)
