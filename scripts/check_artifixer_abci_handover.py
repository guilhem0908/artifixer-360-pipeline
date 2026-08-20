#!/usr/bin/env python3
"""Fail-closed audit of the ABCI files needed by the room-tour handover.

This check is intentionally read-only.  It verifies the pinned containers and
the small set of repository/runtime paths required before a PBS job is
submitted.  It does not replace a GPU smoke test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


EXPECTED_RUNTIME_HASHES = {
    "models/artifixer-cuda12.sif": (
        "226b1e00a26f16442be67157be6acb7f94ace5a698ea9c720fb7dcf89e0af132"
    ),
    "models/colmap-4.1.1-a0d785f.sif": (
        "eeb2d997f811208f5098c961ccf18e41b6cc46c2f195159a67db4a63991b38aa"
    ),
}

REQUIRED_DIRECTORIES = (
    "cache/huggingface",
    "src/ArtiFixer/thirdparty/3DGRUT-ArtiFixer",
)

REQUIRED_EXECUTABLES = (
    "venvs/gauvain-colmap411/bin/python",
)

REQUIRED_REPOSITORY_FILES = (
    "scripts/abci_roomtour_standard_colmap.pbs",
    "scripts/abci_roomtour_forward14_depthpeer77_pilot_4gpu.pbs",
    "scripts/run_roomtour_standard_colmap.py",
    "scripts/generate_forward_facing14_trajectories.py",
    "scripts/prepare_depth_synchronized_multiview_windows.py",
    "scripts/evaluate_depth_synchronized_multiview.py",
    "scripts/evaluate_high_confidence_detail.py",
    "scripts/stitch_artifixer_frustums360.py",
    "scripts/evaluate_temporal_panorama.py",
    "model_eval/run_synchronized_multiview_inference.py",
    "model_eval/synchronized_multiview.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_directory(label: str, path: Path) -> dict[str, object]:
    passed = path.is_dir() and not path.is_symlink()
    return {
        "label": label,
        "kind": "directory",
        "path": str(path),
        "pass": passed,
    }


def _check_file(
    label: str,
    path: Path,
    *,
    expected_sha256: str | None = None,
    executable: bool = False,
) -> dict[str, object]:
    exists = path.is_file() and not path.is_symlink()
    result: dict[str, object] = {
        "label": label,
        "kind": "executable" if executable else "file",
        "path": str(path),
        "exists": exists,
        "pass": exists,
    }
    if exists:
        result["size_bytes"] = path.stat().st_size
    if executable:
        is_executable = exists and os.access(path, os.X_OK)
        result["executable"] = is_executable
        result["pass"] = bool(result["pass"] and is_executable)
    if expected_sha256 is not None:
        result["expected_sha256"] = expected_sha256
        if exists:
            actual = sha256(path)
            result["actual_sha256"] = actual
            result["pass"] = bool(result["pass"] and actual == expected_sha256)
    return result


def _check_run(run_dir: Path) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    manifest_path = run_dir / "contracts" / "source_manifest_v2.json"
    checks.append(_check_file("run source manifest", manifest_path))
    checks.append(_check_file("run manifest", run_dir / "run_manifest_v2.json"))
    if not manifest_path.is_file():
        return checks

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest["records"]
        expected_count = int(manifest["observation_count"])
        valid_header = (
            manifest.get("schema") == "artifixer.roomtour_source_manifest_v2"
            and manifest.get("status") == "PASS"
            and manifest.get("verdict") == "PASS_CONTINUOUS_PERSPECTIVE_OBSERVATIONS"
            and isinstance(records, list)
            and len(records) == expected_count
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        valid_header = False
        records = []
        expected_count = -1
    checks.append(
        {
            "label": "run source contract",
            "kind": "json-contract",
            "path": str(manifest_path),
            "observation_count": expected_count,
            "pass": valid_header,
        }
    )
    if not valid_header:
        return checks

    images_dir = run_dir / "inputs" / "colmap_images"
    image_checks_pass = images_dir.is_dir()
    if image_checks_pass:
        disk_names = sorted(path.name for path in images_dir.glob("frame_*.png"))
        declared_names = [str(record.get("name")) for record in records]
        image_checks_pass = disk_names == declared_names
        for record in records:
            image = images_dir / str(record["name"])
            if (
                not image.is_file()
                or image.stat().st_size != record.get("size_bytes")
                or sha256(image) != record.get("sha256")
            ):
                image_checks_pass = False
                break
    checks.append(
        {
            "label": "run immutable source pixels",
            "kind": "image-contract",
            "path": str(images_dir),
            "observation_count": expected_count,
            "pass": image_checks_pass,
        }
    )
    return checks


def audit(
    afroot: Path,
    *,
    run_dir: Path | None = None,
    expected_runtime_hashes: Mapping[str, str] = EXPECTED_RUNTIME_HASHES,
) -> dict[str, object]:
    root = afroot.expanduser().resolve()
    checks: list[dict[str, object]] = []
    checks.append(_check_directory("ABCI ArtiFixer root", root))
    for relative, expected in expected_runtime_hashes.items():
        checks.append(
            _check_file(
                f"pinned runtime {relative}",
                root / relative,
                expected_sha256=expected,
            )
        )
    for relative in REQUIRED_DIRECTORIES:
        checks.append(_check_directory(f"required directory {relative}", root / relative))
    for relative in REQUIRED_EXECUTABLES:
        checks.append(
            _check_file(
                f"required executable {relative}", root / relative, executable=True
            )
        )
    repository = root / "src" / "ArtiFixer"
    for relative in REQUIRED_REPOSITORY_FILES:
        checks.append(
            _check_file(f"repository file {relative}", repository / relative)
        )
    if run_dir is not None:
        checks.extend(_check_run(run_dir.expanduser().resolve()))
    passed = all(bool(check.get("pass")) for check in checks)
    return {
        "schema": "artifixer.abci_handover_preflight_v1",
        "afroot": str(root),
        "run_dir": str(run_dir.expanduser().resolve()) if run_dir else None,
        "status": "PASS" if passed else "FAIL",
        "submission_authorized": passed,
        "checks": checks,
        "note": "A PASS still requires a GPU import/smoke test before a production run.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--afroot",
        type=Path,
        required=True,
        help="ABCI ArtiFixer root containing src, models, venvs, cache and data",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit(args.afroot, run_dir=args.run_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
