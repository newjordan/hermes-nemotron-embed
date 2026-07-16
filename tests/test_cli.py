from __future__ import annotations

import argparse

import pytest
from hermes_plugins.nemotron_embed_test import cli


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


def test_setup_validates_and_activates_engine_before_saving(tmp_path, monkeypatch):
    config = {}
    saved = []
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.save_config", lambda value: saved.append(value))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    cli._setup(_args(endpoint="http://127.0.0.1:8089/v2/embed/"))

    assert saved == [config]
    assert config["context"]["engine"] == "nemotron-embed"
    assert config["nemotron_embed"]["endpoint"] == "http://127.0.0.1:8089/v2/embed"
    assert config["nemotron_embed"]["model"].endswith("1B-BF16")


def test_setup_does_not_save_invalid_configuration(tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda value: saved.append(value))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    with pytest.raises(ValueError, match="lower than threshold"):
        cli._setup(_args(threshold=0.5, target_ratio=0.8))

    assert saved == []
