from __future__ import annotations

import argparse

import pytest
from hermes_plugins.nemotron_embed_test import cli
from hermes_plugins.nemotron_embed_test.settings import NemotronSettings


def _args(**overrides):
    values = {
        "model": "1b",
        "endpoint": "http://127.0.0.1:8089/v2/embed",
        "api_key_env": "NEMOTRON_EMBED_API_KEY",
        "threshold": 0.75,
        "target_ratio": 0.35,
        "start": False,
        "no_wait": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_setup_validates_and_activates_engine_before_saving(tmp_path, monkeypatch, capsys):
    config = {}
    saved = []
    monkeypatch.setenv("NEMOTRON_EMBED_API_KEY", "super-secret-value")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.save_config", lambda value: saved.append(value))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    cli._setup(_args(endpoint="http://127.0.0.1:8089/v2/embed/"))

    assert saved == [config]
    assert config["context"]["engine"] == "nemotron-embed"
    assert config["nemotron_embed"]["endpoint"] == "http://127.0.0.1:8089/v2/embed"
    assert config["nemotron_embed"]["model"].endswith("1B-BF16")
    output = capsys.readouterr().out
    assert "nemotron-embed configured" in output
    assert "http://127.0.0.1:8089/v2/embed/" not in output
    assert "http://127.0.0.1:8089/v2/embed" in output
    assert "super-secret-value" not in output
    assert "super-secret-value" not in str(config)
    assert "NEMOTRON_EMBED_API_KEY (set)" in output


def test_setup_does_not_save_invalid_configuration(tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda value: saved.append(value))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    with pytest.raises(ValueError, match="lower than threshold"):
        cli._setup(_args(threshold=0.5, target_ratio=0.8))

    assert saved == []


def test_parser_requires_a_subcommand():
    parser = argparse.ArgumentParser()
    cli.setup_argparse(parser)
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_status_hides_api_key_and_reports_archive(tmp_path, monkeypatch, capsys):
    import json

    monkeypatch.setenv("NEMOTRON_EMBED_API_KEY", "super-secret-value")
    settings = NemotronSettings.from_mapping({}, hermes_home=tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    class _Status:
        running = False
        healthy = True
        pid = None
        endpoint = settings.endpoint
        model = settings.model
        log_path = tmp_path / "vllm.log"

    class _Manager:
        def status(self):
            return _Status()

    monkeypatch.setattr(cli, "_manager", lambda settings=None: _Manager())
    monkeypatch.setattr(cli, "_active_engine", lambda: "nemotron-embed")

    cli._status()

    output = capsys.readouterr().out
    assert "super-secret-value" not in output
    payload = json.loads(output)
    assert payload["api_key_configured"] is True
    assert payload["active"] is True
    assert payload["server"]["healthy"] is True
    assert payload["server"]["running"] is False
    assert payload["archive"].endswith("archive.db")


def test_status_explains_invalid_configuration(monkeypatch, capsys):
    def boom():
        raise ValueError("bad endpoint")

    monkeypatch.setattr(cli, "load_settings", boom)
    with pytest.raises(ValueError, match="bad endpoint"):
        cli._status()
    assert "hermes nemotron-embed setup" in capsys.readouterr().err
