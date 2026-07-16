"""Lifecycle management for a loopback-only vLLM embedding server."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from .client import NemotronClient
from .settings import NemotronSettings

VLLM_VERSION = "0.25.0"


@dataclass(frozen=True)
class ServerStatus:
    running: bool
    healthy: bool
    pid: int | None
    endpoint: str
    model: str
    log_path: Path


class ServerManager:
    """Start and stop only the process recorded by this Hermes profile."""

    def __init__(self, settings: NemotronSettings, *, state_dir: Path) -> None:
        self.settings = settings
        self.state_dir = state_dir
        self.state_path = state_dir / "server.json"
        self.log_path = state_dir / "vllm.log"

    def start(self, *, wait: bool = True, timeout: float = 300.0) -> ServerStatus:
        status = self.status()
        if status.running:
            return status
        if status.healthy:
            raise RuntimeError(
                "The configured embedding endpoint is already healthy but is not "
                "managed by this Hermes profile; refusing to start a second server."
            )
        if platform.system() != "Linux":
            raise RuntimeError(
                "Managed vLLM startup requires Linux or WSL2. On other systems, "
                "run a compatible /v2/embed server separately and configure its endpoint."
            )

        parsed = urlparse(self.settings.endpoint)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError(
                "Managed startup only supports loopback endpoints; refusing to bind "
                f"for {self.settings.endpoint}"
            )
        if parsed.scheme != "http":
            raise RuntimeError("Managed startup requires an http loopback endpoint")
        if parsed.path.rstrip("/") != "/v2/embed" or parsed.query or parsed.fragment:
            raise RuntimeError("Managed startup requires an endpoint ending in /v2/embed")
        bind_host = "::1" if parsed.hostname == "::1" else "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        uvx = shutil.which("uvx")
        if not uvx:
            raise RuntimeError(
                "uvx is required for managed startup. Install uv or run vLLM "
                "yourself and point nemotron_embed.endpoint at it."
            )

        command = [
            uvx,
            "--from",
            f"vllm=={VLLM_VERSION}",
            "vllm",
            "serve",
            self.settings.model,
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--host",
            bind_host,
            "--port",
            str(port),
            "--served-model-name",
            self.settings.model,
        ]
        self.state_dir.mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        # Managed serving is local infrastructure, not an analytics surface.
        # Override inherited values so this plugin never enables vLLM usage
        # telemetry implicitly.
        child_env["VLLM_NO_USAGE_STATS"] = "1"
        child_env["DO_NOT_TRACK"] = "1"
        if self.settings.api_key:
            child_env["VLLM_API_KEY"] = self.settings.api_key
        else:
            # Do not accidentally inherit a key intended for another vLLM
            # service. The managed server's auth follows this plugin's config.
            child_env.pop("VLLM_API_KEY", None)
        with self.log_path.open("ab") as log_file:
            self.log_path.chmod(0o600)
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
                env=child_env,
            )
        self.state_path.write_text(
            json.dumps(
                {
                    "pid": process.pid,
                    "pid_start_time": _process_start_time(process.pid),
                    "command": command,
                    "model": self.settings.model,
                    "endpoint": self.settings.endpoint,
                    "started_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.state_path.chmod(0o600)

        if wait:
            deadline = time.monotonic() + max(1.0, timeout)
            while time.monotonic() < deadline:
                status = self.status()
                if status.healthy:
                    return status
                if not status.running:
                    break
                time.sleep(2.0)
            status = self.status()
            if not status.healthy:
                raise RuntimeError(
                    f"vLLM did not become healthy before the timeout. Inspect {self.log_path}"
                )
        return self.status()

    def stop(self, *, timeout: float = 15.0) -> bool:
        state = self._read_state()
        pid = _state_pid(state)
        if pid is None or not _managed_process_matches(state):
            self.state_path.unlink(missing_ok=True)
            return False

        import psutil

        process = psutil.Process(pid)
        try:
            children = process.children(recursive=True)
            for child in reversed(children):
                child.terminate()
            process.terminate()
            _gone, alive = psutil.wait_procs(children + [process], timeout=timeout)
            for remaining in alive:
                remaining.kill()
            return True
        except psutil.NoSuchProcess:
            return False
        finally:
            self.state_path.unlink(missing_ok=True)

    def status(self) -> ServerStatus:
        state = self._read_state()
        pid = _state_pid(state)
        running = pid is not None and _managed_process_matches(state)
        client = NemotronClient(
            endpoint=self.settings.endpoint,
            model=self.settings.model,
            timeout=min(self.settings.request_timeout, 5.0),
            dimensions=self.settings.embedding_dimensions,
            api_key=self.settings.api_key,
        )
        # An operator-managed endpoint has no PID in our state file but is
        # still a healthy, usable backend. Keep process ownership and endpoint
        # readiness as separate signals.
        healthy = client.healthcheck()
        return ServerStatus(
            running=running,
            healthy=healthy,
            pid=pid if running else None,
            endpoint=self.settings.endpoint,
            model=self.settings.model,
            log_path=self.log_path,
        )

    def _read_state(self) -> dict:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}


def _state_pid(state: dict) -> int | None:
    try:
        pid = int(state.get("pid", 0))
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _pid_exists(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        return False


def _process_start_time(pid: int) -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.Error, OSError, ValueError):
        return None


def _managed_process_matches(state: dict) -> bool:
    """Reject stale state whose PID has been recycled by another process."""
    pid = _state_pid(state)
    if pid is None or not _pid_exists(pid):
        return False
    expected = state.get("pid_start_time")
    try:
        expected_start = float(expected)
    except (TypeError, ValueError):
        # Old state without an ownership timestamp is not safe to signal.
        return False
    actual_start = _process_start_time(pid)
    return actual_start is not None and abs(actual_start - expected_start) < 0.01
