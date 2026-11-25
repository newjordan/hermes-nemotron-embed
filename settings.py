"""Configuration for the Nemotron Embed context engine."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"
DEFAULT_ENDPOINT = "http://127.0.0.1:8089/v2/embed"
DEFAULT_API_KEY_ENV = "NEMOTRON_EMBED_API_KEY"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class NemotronSettings:
    """Validated, profile-scoped plugin settings."""

    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    api_key_env: str = DEFAULT_API_KEY_ENV
    threshold_percent: float = 0.75
    target_ratio: float = 0.35
    protect_first_turns: int = 1
    protect_last_turns: int = 4
    request_timeout: float = 120.0
    embedding_dimensions: int = 1024
    max_chunk_chars: int = 12_000
    recency_weight: float = 0.05
    archive_path: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        hermes_home: Path,
    ) -> NemotronSettings:
        values = dict(raw or {})
        endpoint = str(values.get("endpoint") or DEFAULT_ENDPOINT).rstrip("/")
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("nemotron_embed.endpoint must be an http(s) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(
                "nemotron_embed.endpoint must not contain credentials; use api_key_env"
            )

        threshold = _bounded_float(values, "threshold", 0.75, 0.25, 0.95)
        target_ratio = _bounded_float(values, "target_ratio", 0.35, 0.05, 0.90)
        if target_ratio >= threshold:
            raise ValueError("nemotron_embed.target_ratio must be lower than threshold")

        archive_value = values.get("archive_path")
        archive_path = (
            Path(str(archive_value)).expanduser()
            if archive_value
            else hermes_home / "nemotron-embed" / "archive.db"
        )
        api_key_env = str(values.get("api_key_env") or DEFAULT_API_KEY_ENV)
        if not _ENV_NAME.fullmatch(api_key_env):
            raise ValueError("nemotron_embed.api_key_env must be an environment name")

        return cls(
            endpoint=endpoint,
            model=str(values.get("model") or DEFAULT_MODEL),
            api_key_env=api_key_env,
            threshold_percent=threshold,
            target_ratio=target_ratio,
            protect_first_turns=_bounded_int(values, "protect_first_turns", 1, 0, 20),
            protect_last_turns=_bounded_int(values, "protect_last_turns", 4, 1, 50),
            request_timeout=_bounded_float(values, "request_timeout", 120.0, 1.0, 600.0),
            embedding_dimensions=_bounded_int(values, "embedding_dimensions", 1024, 128, 4096),
            max_chunk_chars=_bounded_int(values, "max_chunk_chars", 12_000, 512, 131_072),
            recency_weight=_bounded_float(values, "recency_weight", 0.05, 0.0, 0.5),
            archive_path=archive_path,
        )

    @property
    def api_key(self) -> str | None:
        """Read an optional bearer token without storing it in config.yaml."""
        return os.environ.get(self.api_key_env) or None


def load_settings() -> NemotronSettings:
    """Load the active Hermes profile's plugin configuration."""
    from hermes_cli.config import load_config
    from hermes_constants import get_hermes_home

    config = load_config()
    raw = config.get("nemotron_embed", {})
    if not isinstance(raw, dict):
        raw = {}
    return NemotronSettings.from_mapping(raw, hermes_home=get_hermes_home())


def _bounded_float(
    values: Mapping[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = values.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"nemotron_embed.{key} must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"nemotron_embed.{key} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"nemotron_embed.{key} must be between {minimum} and {maximum}")
    return value


def _bounded_int(
    values: Mapping[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = values.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"nemotron_embed.{key} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"nemotron_embed.{key} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"nemotron_embed.{key} must be between {minimum} and {maximum}")
    return value
