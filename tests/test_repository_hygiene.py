from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "scan_secrets",
    Path(__file__).resolve().parents[1] / "scripts" / "scan_secrets.py",
)
assert _SPEC is not None and _SPEC.loader is not None
scan_secrets = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scan_secrets)


def test_scanner_flags_a_private_key(tmp_path):
    sample = tmp_path / "notes.md"
    # Split the marker so this source file does not itself match the scanner.
    header = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"
    sample.write_text(header + "\n", encoding="utf-8")

    hits = scan_secrets.find_secret_hits([sample])

    assert hits
    assert "private key" in hits[0]


def test_scanner_ignores_the_placeholder_token_used_in_tests(tmp_path):
    sample = tmp_path / "example.py"
    sample.write_text('api_key = "secret-token"\n', encoding="utf-8")

    assert scan_secrets.find_secret_hits([sample]) == []


def test_tracked_repository_files_have_no_secret_patterns():
    root = Path(__file__).resolve().parents[1]
    assert scan_secrets.find_secret_hits(scan_secrets.tracked_files(root)) == []
