#!/usr/bin/env python3
"""Fetch exact source snapshots from sources.lock.json; no weights or ML runtime.

An existing checkout is never reset. It must already be at the requested commit,
or (if clean) this script checks out the requested revision after fetching it.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def git(directory: Path | None, *args: str, capture: bool = False) -> str:
    cmd = ["git"] + (["-C", str(directory)] if directory else []) + list(args)
    result = subprocess.run(cmd, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--source-root", type=Path, help="Override source parent directory")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    for record in lock["repositories"]:
        name, url, commit = (record[key] for key in ("name", "url", "commit"))
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"{name}: lock must contain a full 40-character commit")
        if not url.startswith("https://github.com/"):
            raise ValueError(f"{name}: expected an HTTPS github.com source URL")
        directory = (args.source_root / name if args.source_root else
                     args.lock.resolve().parent / record.get("directory", f"../upstream/{name}"))
        directory = directory.resolve()
        if not directory.exists():
            directory.parent.mkdir(parents=True, exist_ok=True)
            git(None, "init", str(directory))
            git(directory, "remote", "add", "origin", url)
        elif not (directory / ".git").exists():
            raise RuntimeError(f"{directory}: exists but is not a Git checkout")
        remote = git(directory, "remote", "get-url", "origin", capture=True)
        if remote.removesuffix(".git").rstrip("/") != url.removesuffix(".git").rstrip("/"):
            raise RuntimeError(f"{name}: origin does not match lock; refusing to modify")
        current = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                 text=True, capture_output=True)
        if current.returncode != 0 or current.stdout.strip() != commit:
            if git(directory, "status", "--porcelain", capture=True):
                raise RuntimeError(f"{name}: working tree is dirty; refusing checkout")
            git(directory, "fetch", "--depth", "1", "origin", commit)
            git(directory, "checkout", "--detach", commit)
        actual = git(directory, "rev-parse", "HEAD", capture=True)
        if actual != commit:
            raise RuntimeError(f"{name}: expected {commit}, got {actual}")
        print(f"VERIFIED {name}: {actual} at {directory}")


if __name__ == "__main__":
    main()
