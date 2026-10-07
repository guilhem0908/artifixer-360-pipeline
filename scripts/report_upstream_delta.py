#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Measure what this repository adds on top of upstream NVIDIA ArtiFixer.

The script compares two commits with ``git diff --numstat`` and groups the
result by area. By default it compares the upstream base commit of
``nv-tlabs/ArtiFixer`` with the last commit of the 2026 internship, so later
housekeeping commits do not inflate the figures. File paths in the report use the
names the files have in the checked-out tree, so files renamed after the
internship commit are listed under their current name.

It needs a full (non-shallow) clone and nothing but Git and the standard
library:

    python scripts/report_upstream_delta.py --output docs/results/upstream_delta.json
    python scripts/report_upstream_delta.py --markdown
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

UPSTREAM_REPOSITORY = "https://github.com/nv-tlabs/ArtiFixer"
UPSTREAM_BASE = "c3207529847bd951b6e0c71a501887076191bdbd"
INTERNSHIP_HEAD = "950fd746b2f1490ae09d9e64a6b988c7f318b88f"

# Ordered: the first matching rule wins.
AREAS: tuple[tuple[str, str, str], ...] = (
    ("pipeline_scripts", "Pipeline scripts: trajectories, manifests, targets, stitching, QC", r"^scripts/.*\.py$"),
    ("cluster_jobs", "PBS / Singularity job scripts", r"^scripts/.*\.pbs$"),
    ("model_and_data_code", "Inference, model and distillation code", r"^(model_eval|model_training|data_processing)/"),
    ("tests", "Tests", r"^tests/"),
    ("documentation", "Documentation and progress archive", r"^docs/"),
    ("diagram_tooling", "Framework diagram generator and validators", r"^presentations/"),
    ("top_level", "Top-level notices, README and configuration", r""),
)

TEST_FUNCTION = re.compile(r"^\s*def test_\w+\s*\(", re.MULTILINE)


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def commit_exists(revision: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{revision}^{{commit}}"],
            capture_output=True,
        )
    except FileNotFoundError:  # Git is not installed.
        return False
    return completed.returncode == 0


def current_names(head: str) -> dict[str, str]:
    """Map the paths of ``head`` that were renamed since to their current names."""
    renamed: dict[str, str] = {}
    for line in git("diff", "-M", "--name-status", head).splitlines():
        code, *paths = line.split("\t")
        if code.startswith("R"):
            renamed[paths[0]] = paths[1]
    return renamed


def area_of(path: str) -> str:
    for key, _, pattern in AREAS:
        if re.search(pattern, path):
            return key
    raise AssertionError(f"no area matches {path}")


def measure(base: str, head: str) -> dict[str, object]:
    for revision in (base, head):
        if not commit_exists(revision):
            raise SystemExit(
                f"commit {revision} is not available; fetch the full history (git fetch --unshallow) first"
            )

    status: dict[str, str] = {}
    for line in git("diff", "--no-renames", "--name-status", base, head).splitlines():
        code, path = line.split("\t", 1)
        status[path] = code

    areas = {
        key: {
            "label": label,
            "files_added": 0,
            "files_modified": 0,
            "files_deleted": 0,
            "binary_files": 0,
            "lines_added": 0,
            "lines_deleted": 0,
        }
        for key, label, _ in AREAS
    }
    renamed = current_names(head)
    modified_upstream_files = []
    added_files = []
    for line in git("diff", "--no-renames", "--numstat", base, head).splitlines():
        added, deleted, path = line.split("\t", 2)
        entry = areas[area_of(path)]
        code = status[path]
        entry[{"A": "files_added", "M": "files_modified", "D": "files_deleted"}[code]] += 1
        if added == "-":
            entry["binary_files"] += 1
            continue
        entry["lines_added"] += int(added)
        entry["lines_deleted"] += int(deleted)
        if code == "M":
            modified_upstream_files.append(
                {"path": renamed.get(path, path), "lines_added": int(added), "lines_deleted": int(deleted)}
            )
        elif code == "A":
            added_files.append({"path": renamed.get(path, path), "lines": int(added)})
    modified_upstream_files.sort(key=lambda item: (-item["lines_added"], item["path"]))
    added_files.sort(key=lambda item: (-item["lines"], item["path"]))

    new_test_modules = sorted(
        path
        for path, code in status.items()
        if code == "A" and re.match(r"^tests/test_\w+\.py$", path)
    )
    test_functions = sum(
        len(TEST_FUNCTION.findall(git("show", f"{head}:{path}"))) for path in new_test_modules
    )

    def total(field: str) -> int:
        return sum(int(entry[field]) for entry in areas.values())

    return {
        "upstream_repository": UPSTREAM_REPOSITORY,
        "base_commit": git("rev-parse", base).strip(),
        "head_commit": git("rev-parse", head).strip(),
        "head_commit_date": git("show", "-s", "--format=%cs", head).strip(),
        "commits_after_base": int(git("rev-list", "--count", f"{base}..{head}").strip()),
        "totals": {
            "files_changed": len(status),
            "files_added": total("files_added"),
            "files_modified": total("files_modified"),
            "files_deleted": total("files_deleted"),
            "binary_files": total("binary_files"),
            "lines_added": total("lines_added"),
            "lines_deleted": total("lines_deleted"),
        },
        "new_test_modules": len(new_test_modules),
        "test_functions_in_new_modules": test_functions,
        "areas": areas,
        "modified_upstream_files": modified_upstream_files,
        "added_text_files": added_files,
    }


def to_markdown(report: dict[str, object]) -> str:
    rows = [
        "| Area | Files added | Files modified | Lines added | Lines removed |",
        "|---|---:|---:|---:|---:|",
    ]
    for entry in report["areas"].values():
        rows.append(
            f"| {entry['label']} | {entry['files_added']} | {entry['files_modified']} "
            f"| {entry['lines_added']:,} | {entry['lines_deleted']:,} |"
        )
    totals = report["totals"]
    rows.append(
        f"| **Total** | **{totals['files_added']}** | **{totals['files_modified']}** "
        f"| **{totals['lines_added']:,}** | **{totals['lines_deleted']:,}** |"
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=UPSTREAM_BASE, help="upstream base commit")
    parser.add_argument("--head", default=INTERNSHIP_HEAD, help="commit to compare against the base")
    parser.add_argument("--output", type=Path, help="write the JSON report to this file")
    parser.add_argument("--markdown", action="store_true", help="print the per-area table as Markdown")
    args = parser.parse_args()

    report = measure(args.base, args.head)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(to_markdown(report) if args.markdown else json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
