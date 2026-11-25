"""Lifecycle management for a loopback-only vLLM embedding server."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import signal
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

        bind_host, port = _loopback_bind(self.settings.endpoint)
        uvx = shutil.which("uvx")
        if not uvx:
            raise RuntimeError(
                "uvx is required for managed startup. Install uv or run vLLM "
                "yourself and point nemotron_embed.endpoint at it."
            )

        command = build_vllm_command(
            self.settings,
            uvx=uvx,
            bind_host=bind_host,
            port=port,
        )
        self._prepare_state_dir()
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
        with _open_private_append(self.log_path) as log_file:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
                env=child_env,
            )
        _write_private_text(
            self.state_path,
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
        )

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

        try:
            _terminate_pid_tree(pid, timeout=timeout)
            return True
        except ProcessLookupError:
            return False
        finally:
            self.state_path.unlink(missing_ok=True)

    def status(self) -> ServerStatus:
        self._tighten_state_dir()
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

    def _prepare_state_dir(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._tighten_state_dir()

    def _tighten_state_dir(self) -> None:
        if not self.state_dir.exists():
            return
        try:
            self.state_dir.chmod(0o700)
        except OSError:
            return

    def _read_state(self) -> dict:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}


def build_vllm_command(
    settings: NemotronSettings,
    *,
    uvx: str,
    bind_host: str,
    port: int,
) -> list[str]:
    """Build the managed ``vllm serve`` command for a Nemotron Embed checkpoint.

    ``--runner pooling`` keeps the process on the embedding runner. ``--convert
    embed`` is intentionally absent: Nemotron 3 Embed is already a pooling
    model, and converting it replaces the checkpoint's average pooler.
    """
    return [
        uvx,
        "--from",
        f"vllm=={VLLM_VERSION}",
        "vllm",
        "serve",
        settings.model,
        "--runner",
        "pooling",
        "--host",
        bind_host,
        "--port",
        str(port),
        "--served-model-name",
        settings.model,
    ]


def _loopback_bind(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(endpoint)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(
            f"Managed startup only supports loopback endpoints; refusing to bind for {endpoint}"
        )
    if parsed.scheme != "http":
        raise RuntimeError("Managed startup requires an http loopback endpoint")
    if parsed.path.rstrip("/") != "/v2/embed" or parsed.query or parsed.fragment:
        raise RuntimeError("Managed startup requires an endpoint ending in /v2/embed")
    bind_host = "::1" if parsed.hostname == "::1" else "127.0.0.1"
    port = parsed.port or 80
    return bind_host, port


def _write_private_text(path: Path, text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(path, flags, 0o600)
    try:
        handle = os.fdopen(fd, "w", encoding="utf-8")
    except Exception:
        os.close(fd)
        raise
    with handle:
        handle.write(text)
    try:
        path.chmod(0o600)
    except OSError:
        return


def _open_private_append(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(path, flags, 0o600)
    try:
        handle = os.fdopen(fd, "ab")
    except Exception:
        os.close(fd)
        raise
    with contextlib.suppress(OSError):
        path.chmod(0o600)
    return handle


def _terminate_pid_tree(pid: int, *, timeout: float) -> None:
    try:
        _terminate_with_psutil(pid, timeout=timeout)
    except ImportError:
        _terminate_process_group(pid, timeout=timeout)


def _terminate_with_psutil(pid: int, *, timeout: float) -> None:
    import psutil

    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess as exc:
        raise ProcessLookupError(pid) from exc
    children = process.children(recursive=True)
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            continue
    try:
        process.terminate()
    except psutil.NoSuchProcess as exc:
        raise ProcessLookupError(pid) from exc
    _gone, alive = psutil.wait_procs(children + [process], timeout=timeout)
    for remaining in alive:
        try:
            remaining.kill()
        except psutil.NoSuchProcess:
            continue


def _terminate_process_group(pid: int, *, timeout: float) -> None:
    _signal_group(pid, signal.SIGTERM)
    if _wait_until_gone(pid, timeout):
        return
    _signal_group(pid, signal.SIGKILL)
    _wait_until_gone(pid, min(timeout, 2.0))


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return


def _wait_until_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.05)
    return not _pid_exists(pid)


def _state_pid(state: dict) -> int | None:
    try:
        pid = int(state.get("pid", 0))
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _pid_exists(pid: int) -> bool:
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def _process_start_time(pid: int) -> float | None:
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            return float(psutil.Process(pid).create_time())
        except (psutil.Error, OSError, ValueError):
            return None
    return _linux_proc_start_time(pid)


def _linux_proc_start_time(pid: int) -> float | None:
    """Match psutil's create_time from ``/proc`` when psutil is not installed."""
    try:
        rest = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        ticks = int(rest[19])
        btime = None
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
            if line.startswith("btime "):
                btime = int(line.split()[1])
                break
        if btime is None:
            return None
        hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return btime + ticks / hz
    except (OSError, IndexError, ValueError, KeyError):
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
