#!/usr/bin/env python3
"""Fail if repository text files contain obvious secret material."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private key",
        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA |PGP )?PRIVATE KEY-----"),
    ),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}")),
    ("github token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}")),
    ("github fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
)

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".yml",
    ".yaml",
    ".toml",
    ".txt",
    ".json",
    ".cfg",
    ".ini",
    ".sh",
}
SKIP_PARTS = {".git", ".hermes-agent", ".venv", "__pycache__"}


def find_secret_hits(paths: Iterable[Path]) -> list[str]:
    hits: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", "Dockerfile"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for label, pattern in PATTERNS:
            if pattern.search(text):
                hits.append(f"{path}: {label}")
    return hits


def tracked_files(root: Path) -> list[Path]:
    """Return tracked files plus untracked files that are not gitignored."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    files: list[Path] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        path = root / raw.decode()
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        files.append(path)
    return files


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = tracked_files(root)
    hits = find_secret_hits(files)
    if hits:
        print("Potential secrets in tracked files:", file=sys.stderr)
        for hit in hits:
            print(f"  {hit}", file=sys.stderr)
        return 1
    print(f"Scanned {len(files)} repository files; no secret patterns matched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
