from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.check_artifixer_abci_handover import (
    REQUIRED_DIRECTORIES,
    REQUIRED_EXECUTABLES,
    REQUIRED_REPOSITORY_FILES,
    audit,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_runtime(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative, payload in (
        ("models/artifixer-cuda12.sif", b"validated-artifixer-runtime"),
        ("models/colmap-4.1.1-a0d785f.sif", b"validated-colmap-runtime"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        hashes[relative] = digest(path)
    for relative in REQUIRED_DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)
    for relative in REQUIRED_EXECUTABLES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
    repository = root / "src" / "ArtiFixer"
    for relative in REQUIRED_REPOSITORY_FILES:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test fixture\n", encoding="utf-8")
    return hashes


def materialize_run(root: Path) -> Path:
    run = root / "data" / "room_tour_pipeline_v2" / "runs" / "fixture"
    images = run / "inputs" / "colmap_images"
    images.mkdir(parents=True)
    records = []
    for index, payload in enumerate((b"frame-zero", b"frame-one")):
        path = images / f"frame_{index:06d}.png"
        path.write_bytes(payload)
        records.append(
            {
                "index": index,
                "name": path.name,
                "sha256": digest(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema": "artifixer.roomtour_source_manifest_v2",
        "status": "PASS",
        "verdict": "PASS_CONTINUOUS_PERSPECTIVE_OBSERVATIONS",
        "observation_count": 2,
        "records": records,
    }
    contracts = run / "contracts"
    contracts.mkdir()
    (contracts / "source_manifest_v2.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (run / "run_manifest_v2.json").write_text("{}\n", encoding="utf-8")
    return run


def test_preflight_passes_only_with_complete_pinned_layout(tmp_path: Path) -> None:
    hashes = materialize_runtime(tmp_path)
    run = materialize_run(tmp_path)

    report = audit(tmp_path, run_dir=run, expected_runtime_hashes=hashes)

    assert report["status"] == "PASS"
    assert report["submission_authorized"] is True
    assert all(check["pass"] for check in report["checks"])


def test_preflight_rejects_a_container_substitution(tmp_path: Path) -> None:
    hashes = materialize_runtime(tmp_path)
    artifixer = tmp_path / "models" / "artifixer-cuda12.sif"
    artifixer.write_bytes(b"unvalidated-substitute")

    report = audit(tmp_path, expected_runtime_hashes=hashes)

    assert report["status"] == "FAIL"
    assert report["submission_authorized"] is False
    runtime = next(
        check
        for check in report["checks"]
        if check["path"] == str(artifixer.resolve())
    )
    assert runtime["pass"] is False
    assert runtime["actual_sha256"] != runtime["expected_sha256"]


def test_preflight_rejects_drifted_source_pixels(tmp_path: Path) -> None:
    hashes = materialize_runtime(tmp_path)
    run = materialize_run(tmp_path)
    (run / "inputs" / "colmap_images" / "frame_000001.png").write_bytes(b"drift")

    report = audit(tmp_path, run_dir=run, expected_runtime_hashes=hashes)

    source_check = next(
        check
        for check in report["checks"]
        if check["label"] == "run immutable source pixels"
    )
    assert source_check["pass"] is False
    assert report["status"] == "FAIL"
