"""Hermes plugin entry point for local Nemotron semantic context."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    from .cli import dispatch, setup_argparse
    from .engine import NemotronEmbedEngine
    from .settings import load_settings

    # Keep the repair path available even when the current settings are
    # malformed. Hermes will fall back to its built-in compressor until setup
    # writes a valid context-engine configuration.
    ctx.register_cli_command(
        name="nemotron-embed",
        help="Configure and operate local Nemotron context retrieval",
        setup_fn=setup_argparse,
        handler_fn=dispatch,
        description=("Configure semantic context compaction and manage its local vLLM server."),
    )

    try:
        settings = load_settings()
    except ValueError as exc:
        logger.error(
            "Invalid nemotron_embed configuration; context engine not registered: %s. "
            "Run `hermes nemotron-embed setup` to repair it.",
            exc,
        )
        return

    ctx.register_context_engine(NemotronEmbedEngine(settings))
