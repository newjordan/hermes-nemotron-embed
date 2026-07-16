from __future__ import annotations

from hermes_plugins import nemotron_embed_test as plugin
from hermes_plugins.nemotron_embed_test import settings


class _Context:
    def __init__(self):
        self.cli_commands = []
        self.context_engines = []

    def register_cli_command(self, **kwargs):
        self.cli_commands.append(kwargs)

    def register_context_engine(self, engine):
        self.context_engines.append(engine)


def test_invalid_configuration_keeps_repair_cli_but_skips_engine(monkeypatch):
    def invalid_settings():
        raise ValueError("bad endpoint")

    monkeypatch.setattr(settings, "load_settings", invalid_settings)
    context = _Context()

    plugin.register(context)

    assert [command["name"] for command in context.cli_commands] == ["nemotron-embed"]
    assert context.context_engines == []
