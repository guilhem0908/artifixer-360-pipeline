#!/usr/bin/env python3
"""Materialize an immutable room-tour v2 pose run from ordered perspective frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-start-seconds", type=float, required=True)
    parser.add_argument("--source-end-seconds", type=float, required=True)
    parser.add_argument("--sampling-fps", type=float, required=True)
    parser.add_argument("--expected-count", type=int, default=150)
    parser.add_argument(
        "--scene-id",
        default="bearlake_bzedjtnhmh0",
        help="Stable scene identifier recorded in the source contract.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.frames_dir.expanduser().resolve()
    source_video = args.source_video.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists():
        raise SystemExit(f"refusing to replace existing run: {run_dir}")
    if args.expected_count < 2:
        raise SystemExit("room-tour v2 requires at least two ordered observations")
    if args.source_end_seconds <= args.source_start_seconds:
        raise SystemExit("source interval is empty")
    if not source_video.is_file():
        raise SystemExit(f"source video does not exist: {source_video}")

    source_frames = sorted(source_dir.glob("frame_*.png"))
    if len(source_frames) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} PNG frames, found {len(source_frames)}"
        )
    expected_source_names = [
        f"frame_{index:04d}.png" for index in range(1, args.expected_count + 1)
    ]
    if [path.name for path in source_frames] != expected_source_names:
        raise SystemExit("source frame order/naming is not the accepted 0001..0150 sequence")

    contracts = run_dir / "contracts"
    images = run_dir / "inputs" / "colmap_images"
    jobs = run_dir / "jobs"
    contracts.mkdir(parents=True)
    images.mkdir(parents=True)
    jobs.mkdir()

    holdouts = list(range(4, args.expected_count, 10))
    train = [index for index in range(args.expected_count) if index not in set(holdouts)]
    records = []
    for index, source in enumerate(source_frames):
        with Image.open(source) as image:
            image.load()
            if image.size != (1920, 1080) or image.mode not in {"RGB", "RGBA"}:
                raise SystemExit(
                    f"unexpected frame geometry/mode for {source}: {image.size} {image.mode}"
                )
        name = f"frame_{index:06d}.png"
        destination = images / name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        records.append(
            {
                "index": index,
                "name": name,
                "source_name": source.name,
                "source_relative_seconds": index / args.sampling_fps,
                "source_absolute_seconds": args.source_start_seconds
                + index / args.sampling_fps,
                "appearance_holdout": index in holdouts,
                "sha256": sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        )

    source_manifest = {
        "schema": "artifixer.roomtour_source_manifest_v2",
        "schema_version": 2,
        "status": "PASS",
        "verdict": "PASS_CONTINUOUS_PERSPECTIVE_OBSERVATIONS",
        "scene_id": args.scene_id,
        "source": {
            "url": args.source_url,
            "segment_start_seconds": args.source_start_seconds,
            "segment_end_seconds": args.source_end_seconds,
            "sampling_fps": args.sampling_fps,
            "timing_kind": "UNIFORM_SAMPLES_FROM_NATIVE_24_FPS",
            "no_people": True,
            "no_scene_cuts": True,
            "source_video_path": str(source_video),
            "source_video_sha256": sha256(source_video),
            "source_video_size_bytes": source_video.stat().st_size,
        },
        "observation_count": args.expected_count,
        "image_geometry": {
            "projection": "PERSPECTIVE",
            "width": 1920,
            "height": 1080,
            "pixel_aspect_ratio": "1:1",
        },
        "appearance_train_indices": train,
        "appearance_holdout_indices": holdouts,
        "records": records,
    }
    source_manifest_path = contracts / "source_manifest_v2.json"
    write_json(source_manifest_path, source_manifest)

    protocol = {
        "schema": "artifixer.roomtour_geometry_protocol_v2",
        "schema_version": 2,
        "status": "LOCKED",
        "verdict": "PREREGISTERED_ROOMTOUR_GEOMETRY_V2",
        "upstream": {
            "source_manifest_sha256": sha256(source_manifest_path),
            "observation_count": args.expected_count,
        },
        "camera": {
            "model": "SIMPLE_RADIAL",
            "shared": True,
            "width": 1920,
            "height": 1080,
            "initial_params": [960.0, 960.0, 540.0, 0.0],
            "principal_point_frozen": True,
            "focal_refined_during_incremental": True,
            "radial_distortion_refined_during_incremental": True,
            "frozen_for_holdouts_and_global_control": True,
        },
        "solver": {
            "colmap_version": "4.1.1",
            "feature": "ALIKED_N16ROT",
            "maximum_features": 8192,
            "matcher": "ALIKED_LIGHTGLUE_EXHAUSTIVE",
            "two_view_ransac_px": 4.0,
            "random_seed": 0,
        },
        "data_firewall": {
            "mapping_indices": train,
            "holdout_indices": holdouts,
            "final_pose_indices": list(range(args.expected_count)),
            "holdouts_forbidden_from_mapping": True,
            "holdout_pose_refinement_only": True,
        },
        "validation": {
            "consecutive_links": {
                "minimum_true_ransac_inliers": 100,
                "minimum_inlier_ratio": 0.50,
                "minimum_covered_grid_cells_per_image": 8,
                "grid_cell_count": 40,
                "minimum_passing_fraction": 0.95,
            },
            "holdout_localization": {
                "localized_fraction_min": 1.0,
                "median_reprojection_px_max": 1.5,
                "p95_reprojection_px_max": 3.0,
                "minimum_observations_per_image": 100,
            },
            "final_incremental": {
                "registered_images_exact": args.expected_count,
                "median_reprojection_px_max": 1.5,
                "p95_reprojection_px_max": 3.0,
                "shared_camera_exact": True,
            },
            "incremental_global_consensus": {
                "center_median_fraction_scene_diagonal_max": 0.02,
                "center_p95_fraction_scene_diagonal_max": 0.05,
                "rotation_median_deg_max": 2.0,
                "rotation_p95_deg_max": 5.0,
                "reflection_allowed": False,
            },
            "trajectory": {
                "translation_step_p95_over_median_max": 10.0,
                "rotation_step_p95_over_median_max": 10.0,
            },
        },
        "downstream": {
            "before_pose_pass": {
                "standard_3dgs": False,
                "artifixer": False,
                "artifixer3d": False,
                "panorama_render": False,
            },
            "after_pose_pass": [
                "STANDARD_3DGRUT_BASELINE",
                "SPHERICAL_SNAKE_ARTIFIXER",
                "GEOMETRY_LOCKED_ARTIFIXER3D",
                "ERP_RENDER",
            ],
        },
        "failure_policy": {
            "verdict": "BLOCKED_GEOMETRY_ROOMTOUR_V2",
            "threshold_changes_after_results": False,
            "downstream_after_failure": False,
        },
    }
    protocol_path = contracts / "geometry_protocol_v2.json"
    write_json(protocol_path, protocol)

    run_manifest = {
        "schema": "artifixer.roomtour_run_v2",
        "schema_version": 2,
        "status": "READY_FOR_POSE_JOB",
        "run_id": run_dir.name,
        "contracts": {
            "source_manifest_v2.json": sha256(source_manifest_path),
            "geometry_protocol_v2.json": sha256(protocol_path),
        },
        "input_count": args.expected_count,
    }
    write_json(run_dir / "run_manifest_v2.json", run_manifest)
    print(json.dumps(run_manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
