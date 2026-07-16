from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "hermes_plugins.nemotron_embed_test"


if "hermes_plugins" not in sys.modules:
    namespace = types.ModuleType("hermes_plugins")
    namespace.__path__ = []
    sys.modules["hermes_plugins"] = namespace

if PACKAGE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load plugin package for tests")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = PACKAGE_NAME
    module.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
