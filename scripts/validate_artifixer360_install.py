#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Validate a public ArtiFixer 360 checkout without allocating a GPU."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path


REQUIRED_FILES = (
    "model_eval/run_inference.py",
    "model_eval/run_synchronized_multiview_inference.py",
    "model_eval/synchronized_multiview.py",
    "data_processing/run_artifixer3d.py",
    "scripts/generate_spherical_snake_trajectory.py",
    "scripts/generate_cubemap_trajectories.py",
    "scripts/generate_nerfstudio14_trajectories.py",
    "scripts/generate_world_locked14_trajectories.py",
    "scripts/prepare_depth_synchronized_multiview_windows.py",
    "scripts/stitch_artifixer_frustums360.py",
    "thirdparty/3DGRUT-ArtiFixer/threedgrut/trainer.py",
    "thirdparty/3DGRUT-ArtiFixer/threedgrut/single_surface.py",
)

REQUIRED_COMMANDS = ("git", "ffmpeg", "ffprobe")
PYTHON_MODULES = ("numpy", "torch", "cv2", "PIL")


def validate(root: Path, checkpoint: Path | None = None) -> dict[str, object]:
    root = root.resolve()
    checks: dict[str, object] = {
        "python_version": {
            "value": ".".join(map(str, sys.version_info[:3])),
            "pass": sys.version_info >= (3, 10),
        },
        "files": {},
        "commands": {},
        "python_modules": {},
    }
    for relative in REQUIRED_FILES:
        checks["files"][relative] = (root / relative).is_file()
    for command in REQUIRED_COMMANDS:
        checks["commands"][command] = shutil.which(command) is not None
    for module in PYTHON_MODULES:
        checks["python_modules"][module] = importlib.util.find_spec(module) is not None
    if checkpoint is not None:
        path = checkpoint.expanduser().resolve()
        checks["checkpoint"] = {
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
        }

    required_groups = ("files", "commands", "python_modules")
    passed = bool(checks["python_version"]["pass"])
    passed = passed and all(
        all(bool(value) for value in checks[group].values()) for group in required_groups
    )
    if checkpoint is not None:
        passed = passed and bool(checks["checkpoint"]["exists"])
    return {
        "schema": "artifixer360.install_validation.v1",
        "root": str(root),
        "verdict": "PASS" if passed else "FAIL",
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    report = validate(args.root, args.checkpoint)
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"ArtiFixer 360 checkout: {report['root']}")
        for group in ("files", "commands", "python_modules"):
            for name, passed in report["checks"][group].items():
                print(f"  {'OK' if passed else 'MISSING':7s} {group}:{name}")
        if "checkpoint" in report["checks"]:
            item = report["checks"]["checkpoint"]
            print(f"  {'OK' if item['exists'] else 'MISSING':7s} checkpoint:{item['path']}")
        print(f"VERDICT={report['verdict']}")
    raise SystemExit(0 if report["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
