from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from agent import context_engine

PLUGIN_FILES = (
    "__init__.py",
    "after-install.md",
    "archive.py",
    "benchmark.py",
    "cli.py",
    "client.py",
    "engine.py",
    "plugin.yaml",
    "server.py",
    "settings.py",
)


def test_real_hermes_discovery_registers_copyable_engine_and_cli(tmp_path):
    plugin_root = Path(__file__).resolve().parents[1]
    install_dir = tmp_path / "plugins" / "nemotron-embed"
    install_dir.mkdir(parents=True)
    for filename in PLUGIN_FILES:
        shutil.copy2(plugin_root / filename, install_dir / filename)

    hermes_root = Path(context_engine.__file__).resolve().parents[1]
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)
    env["PYTHONPATH"] = str(hermes_root)
    script = textwrap.dedent(
        """
        import copy

        from hermes_cli.config import save_config
        from hermes_cli.plugins import (
            discover_plugins,
            get_plugin_context_engine,
            get_plugin_manager,
        )

        save_config({
            "plugins": {"enabled": ["nemotron-embed"]},
            "context": {"engine": "nemotron-embed"},
        })
        discover_plugins()
        engine = get_plugin_context_engine()
        assert engine is not None
        assert engine.name == "nemotron-embed"
        assert engine.settings.archive_path.parent == __import__("pathlib").Path(
            __import__("os").environ["HERMES_HOME"]
        ) / "nemotron-embed"
        assert "nemotron-embed" in get_plugin_manager()._cli_commands

        clone = copy.deepcopy(engine)
        clone.update_model("test-model", context_length=128_000)
        assert clone.context_length == 128_000
        assert engine.context_length == 0
        assert clone.compress([], focus_topic="release decision", force=True) == []
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
