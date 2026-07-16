from __future__ import annotations

import pytest
from hermes_plugins.nemotron_embed_test.settings import NemotronSettings


def test_defaults_are_profile_scoped(tmp_path):
    settings = NemotronSettings.from_mapping({}, hermes_home=tmp_path)
    assert settings.archive_path == tmp_path / "nemotron-embed" / "archive.db"
    assert settings.endpoint.endswith("/v2/embed")
    assert "1B-BF16" in settings.model


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"endpoint": "file:///tmp/model"}, "http"),
        ({"endpoint": "http://user:secret@localhost:8089/v2/embed"}, "credentials"),
        ({"threshold": 1.1}, "threshold"),
        ({"target_ratio": 0.8, "threshold": 0.75}, "lower than threshold"),
        ({"protect_last_turns": 0}, "protect_last_turns"),
    ],
)
def test_rejects_unsafe_or_incoherent_config(tmp_path, raw, message):
    with pytest.raises(ValueError, match=message):
        NemotronSettings.from_mapping(raw, hermes_home=tmp_path)
