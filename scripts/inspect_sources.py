#!/usr/bin/env python3
"""Audit code evidence against pinned Git snapshots without importing detectors.

Checks commits, line bounds, source hashes and Python AST symbol definitions.
This is static inspection evidence, not evidence of training or model inference.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def git(directory: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(directory), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def definitions(tree: ast.AST) -> list[dict]:
    found = []
    def walk(node: ast.AST, parents: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                path = parents + [child.name]
                found.append({"symbol": ".".join(path), "start_line": child.lineno,
                              "end_line": getattr(child, "end_lineno", child.lineno)})
                walk(child, path)
            else:
                walk(child, parents)
    walk(tree, [])
    return found


def normal_repo(value: str) -> str:
    return value.removesuffix(".git").rstrip("/").removeprefix("https://github.com/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/evidence.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--source-root", type=Path, help="Override source parent directory")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    raw = json.loads(args.evidence.read_text())
    records = raw if isinstance(raw, list) else raw.get("evidence", raw.get("entries", []))
    if not records:
        raise ValueError("No source evidence entries found")
    repo_map, snapshots = {}, []
    for record in lock["repositories"]:
        directory = (args.source_root / record["name"] if args.source_root else
                     args.lock.resolve().parent / record.get("directory", f"../upstream/{record['name']}"))
        directory = directory.resolve()
        actual = git(directory, "rev-parse", "HEAD")
        state = {"name": record["name"], "url": record["url"],
                 "expected_commit": record["commit"], "actual_commit": actual,
                 "commit_matches": actual == record["commit"],
                 "tracked_files_clean": not bool(git(directory, "status", "--porcelain", "--untracked-files=no"))}
        snapshots.append(state)
        for key in (record["name"], normal_repo(record["url"]), normal_repo(record["url"]).split("/")[-1]):
            repo_map[key] = (record, directory, state)
    results, seen = [], set()
    for entry in records:
        item = dict(entry)
        errors = []
        if entry["id"] in seen:
            errors.append("duplicate evidence ID")
        seen.add(entry["id"])
        key = normal_repo(entry["repo"])
        try:
            repo, directory, state = repo_map[key]
            path = (directory / entry["path"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("evidence file escapes checkout")
            data = path.read_bytes()
            source = data.decode("utf-8")
            lines = source.splitlines()
            start, end = int(entry["start_line"]), int(entry["end_line"])
            item.update(file_sha256=hashlib.sha256(data).hexdigest(), file_lines=len(lines),
                        span_sha256=hashlib.sha256("\n".join(lines[start - 1:end]).encode()).hexdigest())
            if not (1 <= start <= end <= len(lines)):
                errors.append("line span is outside file bounds")
            if entry["commit"] != repo["commit"] or not state["commit_matches"]:
                errors.append("evidence or checkout commit differs from lock")
            if not state["tracked_files_clean"]:
                errors.append("tracked checkout files are modified")
            symbol = entry.get("symbol", "")
            if path.suffix == ".py" and symbol:
                candidates = definitions(ast.parse(source, filename=str(path)))
                matches = [d for d in candidates if d["symbol"] == symbol or
                           d["symbol"].endswith("." + symbol)]
                item["definitions"] = matches
                if not matches:
                    errors.append(f"Python symbol not found: {symbol}")
                elif not any(d["start_line"] <= end and d["end_line"] >= start for d in matches):
                    errors.append("evidence span does not overlap symbol definition")
            else:
                item["definitions"] = []
                item["symbol_check"] = "non-Python/configuration reference: line/hash verification only"
            item["permalink"] = f"{repo['url'].removesuffix('.git')}/blob/{repo['commit']}/{entry['path']}#L{start}-L{end}"
        except (KeyError, OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as exc:
            errors.append(str(exc))
        item["errors"] = errors
        item["passed"] = not errors
        results.append(item)
    output = {"kind": "static_source_audit_not_model_execution", "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "python": sys.version.split()[0], "platform": platform.platform(),
              "repositories": snapshots, "evidence_count": len(results),
              "passed_count": sum(item["passed"] for item in results), "evidence": results}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    md = ["# Static source-code inspection audit", "", "This audit verifies source locations and identities. It does not run detector inference or training.", "",
          f"Generated: {output['generated_at_utc']}", f"Python: {output['python']}",
          f"Evidence checks: {output['passed_count']}/{len(results)} passed.", "", "## Snapshot identity", "",
          "| Repository | Commit | Tracked files clean |", "|---|---|---|"]
    for snapshot in snapshots:
        md.append(f"| {snapshot['name']} | `{snapshot['actual_commit']}` | {snapshot['tracked_files_clean']} |")
    md += ["", "## Code references", "", "| ID | File and symbol | Lines | Status |", "|---|---|---|---|"]
    for item in results:
        status = "PASS" if item["passed"] else "; ".join(item["errors"])
        md.append(f"| {item['id']} | `{item['path']}` / `{item.get('symbol', '')}` | {item['start_line']}-{item['end_line']} | {status} |")
    md += ["", "File and selected-span SHA256 values, claims, reasons, definition locations, and immutable permalinks are in `source_audit.json`.",
           "The script verifies that an evidence span overlaps the named definition; it cannot prove the interpretation of that code is correct.", ""]
    (args.output_dir / "source_audit.md").write_text("\n".join(md))
    print(f"Static audit: {output['passed_count']}/{len(results)} evidence references passed")
    if any(not item["passed"] for item in results) or any(not s["commit_matches"] or not s["tracked_files_clean"] for s in snapshots):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
