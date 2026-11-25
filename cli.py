"""Operator CLI for setup and local serving."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .server import ServerManager
from .settings import DEFAULT_ENDPOINT, load_settings

MODEL_PRESETS = {
    "1b": "nvidia/Nemotron-3-Embed-1B-BF16",
    "8b": "nvidia/Nemotron-3-Embed-8B-BF16",
}


def setup_argparse(parser: argparse.ArgumentParser) -> None:
    subcommands = parser.add_subparsers(dest="nemotron_command", required=True)

    setup = subcommands.add_parser("setup", help="Configure and optionally start local serving")
    setup.add_argument(
        "--model",
        default="1b",
        help="1b, 8b, or a Hugging Face model ID (default: 1b)",
    )
    setup.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    setup.add_argument(
        "--api-key-env",
        default="NEMOTRON_EMBED_API_KEY",
        help="Environment variable containing an optional server bearer token",
    )
    setup.add_argument("--threshold", type=float, default=0.75)
    setup.add_argument("--target-ratio", type=float, default=0.35)
    setup.add_argument(
        "--start",
        action="store_true",
        help="Start a managed vLLM server after saving configuration",
    )
    setup.add_argument(
        "--no-wait",
        action="store_true",
        help="Return immediately after starting vLLM",
    )

    start = subcommands.add_parser("start", help="Start the managed vLLM server")
    start.add_argument("--no-wait", action="store_true")
    start.add_argument("--timeout", type=float, default=300.0)

    stop = subcommands.add_parser("stop", help="Stop the managed vLLM server")
    stop.add_argument("--timeout", type=float, default=15.0)

    subcommands.add_parser("status", help="Show engine and server status")
    subcommands.add_parser(
        "benchmark",
        help="Run an end-to-end token reduction and recovery smoke benchmark",
    )
    parser.set_defaults(func=dispatch)


def dispatch(args: argparse.Namespace) -> None:
    command = getattr(args, "nemotron_command", None)
    if command == "setup":
        _setup(args)
    elif command == "start":
        _start(args)
    elif command == "stop":
        _stop(args)
    elif command == "status":
        _status()
    elif command == "benchmark":
        _benchmark()
    else:
        print(
            "Usage: hermes nemotron-embed <setup|start|stop|status|benchmark>",
            file=sys.stderr,
        )


def _setup(args: argparse.Namespace) -> None:
    from hermes_cli.config import load_config, save_config
    from hermes_constants import get_hermes_home

    from .settings import NemotronSettings

    model = MODEL_PRESETS.get(args.model.casefold(), args.model)
    config = load_config()
    section = config.setdefault("nemotron_embed", {})
    if not isinstance(section, dict):
        section = {}
        config["nemotron_embed"] = section
    requested = {
        **section,
        "endpoint": args.endpoint,
        "model": model,
        "api_key_env": args.api_key_env,
        "threshold": args.threshold,
        "target_ratio": args.target_ratio,
    }
    validated = NemotronSettings.from_mapping(requested, hermes_home=get_hermes_home())
    section.update(
        {
            "endpoint": validated.endpoint,
            "model": validated.model,
            "api_key_env": validated.api_key_env,
            "threshold": validated.threshold_percent,
            "target_ratio": validated.target_ratio,
        }
    )
    context = config.setdefault("context", {})
    if not isinstance(context, dict):
        context = {}
        config["context"] = context
    context["engine"] = "nemotron-embed"
    save_config(config)
    key_state = "set" if validated.api_key else "unset"
    print("nemotron-embed configured")
    print("  context.engine  nemotron-embed")
    print(f"  model           {validated.model}")
    print(f"  endpoint        {validated.endpoint}")
    print(f"  threshold       {validated.threshold_percent:g}")
    print(f"  target ratio    {validated.target_ratio:g}")
    print(f"  api key env     {validated.api_key_env} ({key_state})")
    if args.start:
        _start(
            argparse.Namespace(
                no_wait=args.no_wait,
                timeout=300.0,
            )
        )
    else:
        print("Start the local server with: hermes nemotron-embed start")


def _start(args: argparse.Namespace) -> None:
    manager = _manager()
    status = manager.start(
        wait=not bool(getattr(args, "no_wait", False)),
        timeout=float(getattr(args, "timeout", 300.0)),
    )
    print(_format_status(status))


def _stop(args: argparse.Namespace) -> None:
    stopped = _manager().stop(timeout=float(getattr(args, "timeout", 15.0)))
    print("Stopped managed vLLM server." if stopped else "Server is not running.")


def _status() -> None:
    settings = _require_settings()
    status = _manager(settings).status()
    payload: dict[str, Any] = {
        "engine": "nemotron-embed",
        "active": _active_engine() == "nemotron-embed",
        "model": settings.model,
        "endpoint": settings.endpoint,
        "api_key_env": settings.api_key_env,
        "api_key_configured": bool(settings.api_key),
        "archive": str(settings.archive_path),
        "server": {
            "running": status.running,
            "healthy": status.healthy,
            "pid": status.pid,
            "endpoint": status.endpoint,
            "model": status.model,
            "log": str(status.log_path),
        },
    }
    print(json.dumps(payload, indent=2))


def _benchmark() -> None:
    from .benchmark import run_compaction_benchmark

    result = run_compaction_benchmark(_require_settings())
    print(json.dumps(result, indent=2))
    verdict = "passed" if result["passed"] else "failed"
    print(f"benchmark {verdict}", file=sys.stderr)
    if not result["passed"]:
        raise RuntimeError("Nemotron Embed compaction benchmark failed")


def _require_settings():
    try:
        return load_settings()
    except ValueError as exc:
        print(f"Invalid nemotron_embed configuration: {exc}", file=sys.stderr)
        print("Repair it with: hermes nemotron-embed setup", file=sys.stderr)
        raise


def _manager(settings=None) -> ServerManager:
    from hermes_constants import get_hermes_home

    active = _require_settings() if settings is None else settings
    return ServerManager(active, state_dir=get_hermes_home() / "nemotron-embed")


def _active_engine() -> str:
    from hermes_cli.config import load_config

    context = load_config().get("context", {})
    return str(context.get("engine", "compressor")) if isinstance(context, dict) else "compressor"


def _format_status(status) -> str:
    if status.running and status.healthy:
        state = f"managed and healthy (pid {status.pid})"
    elif status.running:
        state = f"managed and starting (pid {status.pid})"
    elif status.healthy:
        state = "external endpoint reachable"
    else:
        state = "stopped"
    return (
        f"Nemotron Embed server: {state}\n"
        f"Endpoint: {status.endpoint}\n"
        f"Model: {status.model}\n"
        f"Log: {status.log_path}"
    )
