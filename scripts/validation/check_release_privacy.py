#!/usr/bin/env python3
"""Check tracked files and distributions without printing sensitive matches.

This bounded check complements secret scanning and manual screenshot review;
it does not certify that arbitrary personal information is absent.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import zipfile


_URL = re.compile(rb"https?://[^\s\"'<>]+")
_HOME = re.compile(rb"/(?:Users|home)/([A-Za-z0-9_.-]+)")
_PYTEST_HOME = re.compile(rb"/pytest-of-([A-Za-z0-9_.-]+)")
_SAFE_HOME = {b"user", b"example", b"runner", b"redacted"}
_SECRETS = {
    "private-key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    "github-token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"
    ),
    "aws-access-key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
}
_PRIVATE_NAMES = {
    ".aws", ".ssh", ".git", ".agents", ".codex", ".env",
    "field-test-c11.md", "refactor-infra.md", "id_rsa", "id_ed25519",
}


def check_payload(name: str, payload: bytes) -> list[str]:
    """Return rule IDs only; never include the matching bytes in diagnostics."""
    problems = []
    if any(part in _PRIVATE_NAMES or part.startswith(".env.")
           for part in PurePosixPath(name).parts):
        problems.append("private-file")
    if name.endswith(".gz"):
        try:
            payload = gzip.decompress(payload)
        except (OSError, EOFError):
            return problems + ["invalid-gzip"]
    # Public vendor URLs can legitimately contain a home/product route.
    without_urls = _URL.sub(b"", payload)
    if any(match[1].lower() not in _SAFE_HOME for match in _HOME.finditer(without_urls)):
        problems.append("personal-home-path")
    if any(match[1].lower() not in _SAFE_HOME for match in _PYTEST_HOME.finditer(payload)):
        problems.append("personal-test-path")
    for rule, pattern in _SECRETS.items():
        if pattern.search(payload):
            problems.append(rule)
    return problems


def artifact_payloads(path: Path):
    if path.suffix == ".whl" or path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if not entry.is_dir():
                    yield entry.filename, archive.read(entry)
    else:
        with tarfile.open(path, "r:*") as archive:
            for entry in archive:
                if (entry.uid != 0 or entry.gid != 0
                        or entry.uname not in ("", "root")
                        or entry.gname not in ("", "root")):
                    raise ValueError(f"{entry.name}: non-neutral archive ownership")
                if entry.issym() or entry.islnk():
                    # Inspect link targets too, without extracting any member.
                    yield entry.name, entry.linkname.encode()
                elif entry.isfile():
                    with archive.extractfile(entry) as stream:
                        yield entry.name, stream.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.artifact:
        entries = (entry for path in args.artifact for entry in artifact_payloads(path))
    else:
        names = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=root,
        ).decode().split("\0")
        entries = ((name, (root / name).read_bytes()) for name in names if name)
    count = 0
    failures = 0
    try:
        for name, payload in entries:
            count += 1
            for rule in check_payload(name, payload):
                print(f"{name}: {rule}")
                failures += 1
    except ValueError as error:
        print(error)
        failures += 1
    print(f"Privacy guard: {count} files checked, {failures} findings")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
