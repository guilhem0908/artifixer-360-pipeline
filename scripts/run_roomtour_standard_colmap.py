#!/usr/bin/env python3
"""Reconstruct an ordered single-camera room-tour with standard COLMAP SfM.

This is deliberately a production reconstruction, not a scientific holdout
benchmark.  Every source frame participates in feature matching, mapping, and
bundle adjustment.  The resulting ``final_sparse`` directory is a regular
COLMAP model that can be published as ``colmap_scene/sparse/0`` and consumed by
``data_processing/prepare_colmap_artifixer_inputs.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Make `python scripts/<name>.py` work on a plain clone: the repository root must
# be importable for the `scripts.*` / `model_eval.*` imports below.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_erp_benchmark_colmap import (  # noqa: E402
    Commands,
    GeometryError,
    _array,
    _grid_coverage,
    digest,
    largest_model,
    pair_id,
    pose_matrix,
    read_json,
    reprojection_metrics,
    rotation_angle,
)


DEFAULT_CAMERA_MODEL = "SIMPLE_RADIAL"
DEFAULT_CAMERA_PARAMS = (960.0, 960.0, 540.0, 0.0)
DEFAULT_SEQUENTIAL_OVERLAP = 15


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_source_frames(images_dir: Path, manifest_path: Path) -> tuple[list[str], dict[str, Any]]:
    """Validate the immutable source pixels while intentionally ignoring holdout labels."""
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != "artifixer.roomtour_source_manifest_v2"
        or manifest.get("status") != "PASS"
        or manifest.get("verdict") != "PASS_CONTINUOUS_PERSPECTIVE_OBSERVATIONS"
    ):
        raise GeometryError("source manifest is not an accepted room-tour source")
    expected_count = manifest.get("observation_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 2:
        raise GeometryError("source manifest observation_count must be an integer >= 2")
    source = manifest.get("source", {})
    if (
        source.get("no_people") is not True
        or source.get("no_scene_cuts") is not True
        or float(source.get("sampling_fps", 0.0)) <= 0.0
    ):
        raise GeometryError("source is not an ordered, continuous, person-free video sample")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != expected_count:
        raise GeometryError("source manifest record count drifted")

    names: list[str] = []
    for index, record in enumerate(records):
        name = f"frame_{index:06d}.png"
        image = images_dir / name
        if (
            record.get("index") != index
            or record.get("name") != name
            or not image.is_file()
            or image.stat().st_size != record.get("size_bytes")
            or digest(image) != record.get("sha256")
        ):
            raise GeometryError(f"source image drifted: {image}")
        names.append(name)
    disk_names = sorted(path.name for path in images_dir.iterdir() if path.is_file())
    if disk_names != names:
        raise GeometryError("image directory contains missing or unexpected files")
    return names, manifest


def feature_extractor_arguments(
    database: Path,
    images_dir: Path,
    *,
    camera_model: str,
    camera_params: Sequence[float],
) -> list[str]:
    if camera_model != DEFAULT_CAMERA_MODEL or len(camera_params) != 4:
        raise GeometryError("this production profile expects one SIMPLE_RADIAL camera")
    return [
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images_dir),
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_params",
        ",".join(format(float(value), ".17g") for value in camera_params),
        "--FeatureExtraction.type",
        "ALIKED_N16ROT",
        "--AlikedExtraction.max_num_features",
        "8192",
        "--FeatureExtraction.gpu_index",
        "0",
        "--default_random_seed",
        "0",
    ]


def sequential_matcher_arguments(database: Path, *, overlap: int) -> list[str]:
    if overlap < 2:
        raise GeometryError("sequential overlap must include more than the adjacent frame")
    return [
        "sequential_matcher",
        "--database_path",
        str(database),
        "--FeatureMatching.type",
        "ALIKED_LIGHTGLUE",
        "--TwoViewGeometry.max_error",
        "4",
        "--SequentialMatching.overlap",
        str(overlap),
        # The five-fps sequence contains enough redundant adjacent images.  The
        # quadratic offsets add direct longer-baseline constraints without
        # pretending that this one-way clip contains a physical loop closure.
        "--SequentialMatching.quadratic_overlap",
        "1",
        "--SequentialMatching.loop_detection",
        "0",
        "--FeatureMatching.gpu_index",
        "0",
        "--default_random_seed",
        "0",
    ]


def exhaustive_matcher_arguments(database: Path) -> list[str]:
    """Use every pair: 150 images are comfortably in COLMAP's small-set regime."""
    return [
        "exhaustive_matcher",
        "--database_path",
        str(database),
        "--FeatureMatching.type",
        "ALIKED_LIGHTGLUE",
        "--TwoViewGeometry.max_error",
        "4",
        "--FeatureMatching.gpu_index",
        "0",
        "--default_random_seed",
        "0",
    ]


def mapper_arguments(database: Path, images_dir: Path, output: Path) -> list[str]:
    return [
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(output),
        "--Mapper.multiple_models",
        "0",
        "--Mapper.random_seed",
        "0",
        "--Mapper.ba_refine_focal_length",
        "1",
        "--Mapper.ba_refine_principal_point",
        "0",
        "--Mapper.ba_refine_extra_params",
        "1",
        "--default_random_seed",
        "0",
    ]


def registered_image_names(reconstruction: Any) -> list[str]:
    return sorted(
        image.name for image in reconstruction.images.values() if image.has_pose
    )


def register_remaining_images(
    commands: Commands,
    *,
    database: Path,
    images_dir: Path,
    model_path: Path,
    output_root: Path,
    expected_count: int,
) -> tuple[Path, Any]:
    """Use COLMAP's normal completion tools if the initial mapper missed frames."""
    import pycolmap

    registered_dir = output_root / "registered_all"
    registered_dir.mkdir()
    commands.run(
        [
            "image_registrator",
            "--database_path",
            str(database),
            "--input_path",
            str(model_path),
            "--output_path",
            str(registered_dir),
            "--Mapper.ba_refine_focal_length",
            "1",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "1",
            "--default_random_seed",
            "0",
        ]
    )
    registered = pycolmap.Reconstruction(registered_dir)
    if registered.num_reg_images() < expected_count:
        return registered_dir, registered

    triangulated_dir = output_root / "triangulated_all"
    triangulated_dir.mkdir()
    commands.run(
        [
            "point_triangulator",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--input_path",
            str(registered_dir),
            "--output_path",
            str(triangulated_dir),
            "--clear_points",
            "0",
            "--refine_intrinsics",
            "1",
            "--Mapper.ba_refine_focal_length",
            "1",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "1",
            "--default_random_seed",
            "0",
        ]
    )
    return triangulated_dir, pycolmap.Reconstruction(triangulated_dir)


def final_bundle_adjustment(
    commands: Commands, *, input_model: Path, output_model: Path
) -> Any:
    import pycolmap

    output_model.mkdir()
    commands.run(
        [
            "bundle_adjuster",
            "--input_path",
            str(input_model),
            "--output_path",
            str(output_model),
            "--BundleAdjustment.refine_focal_length",
            "1",
            "--BundleAdjustment.refine_principal_point",
            "0",
            "--BundleAdjustment.refine_extra_params",
            "1",
            "--BundleAdjustment.refine_points3D",
            "1",
            "--BundleAdjustmentCeres.use_gpu",
            "1",
            "--BundleAdjustmentCeres.gpu_index",
            "0",
            "--BundleAdjustmentCeres.max_num_iterations",
            "200",
            "--default_random_seed",
            "0",
        ]
    )
    return pycolmap.Reconstruction(output_model)


def matching_diagnostics(database: Path, names: Sequence[str]) -> dict[str, Any]:
    """Describe the temporal match graph without turning diagnostics into gates."""
    connection = sqlite3.connect(database)
    try:
        image_ids = {
            str(name): int(image_id)
            for image_id, name in connection.execute("SELECT image_id,name FROM images")
        }
        keypoints = {
            int(image_id): _array(blob, "<f4", int(rows), int(cols))[:, :2]
            for image_id, rows, cols, blob in connection.execute(
                "SELECT image_id,rows,cols,data FROM keypoints"
            )
        }
        raw = {
            int(identifier): int(rows)
            for identifier, rows in connection.execute("SELECT pair_id,rows FROM matches")
        }
        geometries = {
            int(identifier): (int(rows), blob)
            for identifier, rows, blob in connection.execute(
                "SELECT pair_id,rows,data FROM two_view_geometries"
            )
        }
    finally:
        connection.close()

    links: list[dict[str, Any]] = []
    for first_name, second_name in zip(names, names[1:]):
        first_id = image_ids[first_name]
        second_id = image_ids[second_name]
        identifier = pair_id(first_id, second_id)
        count, blob = geometries.get(identifier, (0, None))
        inliers = _array(blob, "<u4", count, 2)
        if first_id < second_id:
            first_indices, second_indices = inliers[:, 0], inliers[:, 1]
        else:
            first_indices, second_indices = inliers[:, 1], inliers[:, 0]
        first_points = keypoints[first_id][first_indices] if count else np.empty((0, 2))
        second_points = keypoints[second_id][second_indices] if count else np.empty((0, 2))
        raw_count = raw.get(identifier, 0)
        links.append(
            {
                "pair": [first_name, second_name],
                "raw_matches": raw_count,
                "geometric_inliers": count,
                "inlier_ratio": count / raw_count if raw_count else 0.0,
                "covered_grid_cells": [
                    _grid_coverage(first_points, 1920, 1080),
                    _grid_coverage(second_points, 1920, 1080),
                ],
            }
        )
    counts = np.asarray([record["geometric_inliers"] for record in links], dtype=np.float64)
    return {
        "method": "ordered_adjacent_pair_diagnostics_only",
        "pair_count": len(links),
        "pairs_with_geometry": int(np.count_nonzero(counts)),
        "geometric_inliers_min": int(counts.min()) if len(counts) else 0,
        "geometric_inliers_median": float(np.median(counts)) if len(counts) else 0.0,
        "geometric_inliers_p95": float(np.percentile(counts, 95)) if len(counts) else 0.0,
        "links": links,
    }


def export_poses(
    reconstruction: Any, names: Sequence[str], *, solver_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_name = {image.name: image for image in reconstruction.images.values() if image.has_pose}
    if set(by_name) != set(names):
        missing = sorted(set(names) - set(by_name))
        raise GeometryError(f"final COLMAP pose set is incomplete: {missing}")
    records: list[dict[str, Any]] = []
    camera_to_world_values: list[np.ndarray] = []
    for name in names:
        world_to_camera = pose_matrix(by_name[name])
        camera_to_world = np.linalg.inv(world_to_camera)
        if np.linalg.det(camera_to_world[:3, :3]) <= 0:
            raise GeometryError(f"COLMAP returned a reflected pose: {name}")
        camera_to_world_values.append(camera_to_world)
        records.append(
            {
                "name": name,
                "world_to_camera": world_to_camera.tolist(),
                "camera_to_world": camera_to_world.tolist(),
            }
        )
    matrices = np.stack(camera_to_world_values)
    centers = matrices[:, :3, 3]
    translations = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    rotations = np.asarray(
        [
            rotation_angle(matrices[index, :3, :3], matrices[index + 1, :3, :3])
            for index in range(len(matrices) - 1)
        ],
        dtype=np.float64,
    )

    def distribution(values: np.ndarray) -> dict[str, float | None]:
        median = float(np.median(values))
        return {
            "median": median,
            "p95": float(np.percentile(values, 95)),
            "maximum": float(values.max()),
            "p95_over_median": float(np.percentile(values, 95) / median)
            if median > 1e-12
            else None,
            "maximum_over_median": float(values.max() / median)
            if median > 1e-12
            else None,
        }

    camera = next(iter(reconstruction.cameras.values()))
    poses = {
        "schema": "artifixer.roomtour_standard_colmap_poses_v1",
        "schema_version": 1,
        "solver_id": solver_id,
        "pose_image_count": len(records),
        "camera": {
            "model": str(camera.model_name),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in np.asarray(camera.params)],
        },
        "poses": records,
    }
    trajectory = {
        "method": "ordered_pose_step_diagnostics_only",
        "translation_steps": distribution(translations),
        "rotation_steps_degrees": distribution(rotations),
    }
    return poses, trajectory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--colmap-container", type=Path, required=True)
    parser.add_argument("--colmap-bind", type=Path, required=True)
    parser.add_argument("--colmap-cudnn-dir", type=Path, required=True)
    parser.add_argument(
        "--matching-mode",
        choices=("exhaustive", "sequential"),
        default="exhaustive",
        help="Exhaustive is the production default for this 150-image set.",
    )
    parser.add_argument("--sequential-overlap", type=int, default=DEFAULT_SEQUENTIAL_OVERLAP)
    args = parser.parse_args(argv)

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise GeometryError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    images_dir = args.images_dir.expanduser().resolve()
    manifest_path = args.source_manifest.expanduser().resolve()
    names, manifest = load_source_frames(images_dir, manifest_path)
    expected_count = len(names)

    commands = Commands(
        output / "colmap_commands.jsonl",
        container=args.colmap_container.expanduser().resolve(),
        bind=args.colmap_bind.expanduser().resolve(),
        cudnn_dir=args.colmap_cudnn_dir.expanduser().resolve(),
    )
    database = output / "database.db"
    commands.run(
        feature_extractor_arguments(
            database,
            images_dir,
            camera_model=DEFAULT_CAMERA_MODEL,
            camera_params=DEFAULT_CAMERA_PARAMS,
        )
    )
    if args.matching_mode == "exhaustive":
        commands.run(exhaustive_matcher_arguments(database))
    else:
        commands.run(sequential_matcher_arguments(database, overlap=args.sequential_overlap))
    temporal_matches = matching_diagnostics(database, names)
    write_json(output / "temporal_matches.json", temporal_matches)

    candidates = output / "sparse_candidates"
    candidates.mkdir()
    commands.run(mapper_arguments(database, images_dir, candidates))
    model_path, model = largest_model(candidates)
    initial_count = model.num_reg_images()
    completion_used = initial_count < expected_count
    if completion_used:
        model_path, model = register_remaining_images(
            commands,
            database=database,
            images_dir=images_dir,
            model_path=model_path,
            output_root=output,
            expected_count=expected_count,
        )
    if model.num_reg_images() != expected_count:
        raise GeometryError(
            f"standard COLMAP registered {model.num_reg_images()}/{expected_count} images"
        )

    final_sparse = output / "final_sparse"
    final_model = final_bundle_adjustment(
        commands, input_model=model_path, output_model=final_sparse
    )
    final_names = registered_image_names(final_model)
    if final_names != sorted(names):
        raise GeometryError("final bundle adjustment lost or renamed source frames")
    if len(final_model.cameras) != 1:
        raise GeometryError("single-camera video produced more than one COLMAP camera")
    camera = next(iter(final_model.cameras.values()))
    if str(camera.model_name) != DEFAULT_CAMERA_MODEL:
        raise GeometryError("final COLMAP camera model is not SIMPLE_RADIAL")

    solver_id = f"colmap_incremental_all_frames_{args.matching_mode}"
    poses, trajectory = export_poses(final_model, names, solver_id=solver_id)
    poses_path = output / "poses.json"
    write_json(poses_path, poses)
    reprojection = reprojection_metrics(final_model, names)
    metrics = {
        "schema": "artifixer.roomtour_standard_colmap_metrics_v1",
        "schema_version": 1,
        "status": "PASS",
        "verdict": "PASS_STANDARD_COLMAP_ALL_FRAMES",
        "solver_id": solver_id,
        "source_manifest_sha256": digest(manifest_path),
        "source_sampling_fps": float(manifest["source"]["sampling_fps"]),
        "source_frame_count": len(names),
        "registered_images_initial_mapper": initial_count,
        "registered_images": final_model.num_reg_images(),
        "completion_pass_used": completion_used,
        "point3D_count": len(final_model.points3D),
        "shared_camera": True,
        "camera": poses["camera"],
        "matching": {
            "extractor": "ALIKED_N16ROT",
            "matcher": "ALIKED_LIGHTGLUE",
            "pair_selection": args.matching_mode,
            "ordering": "lexicographic_source_frame_order",
            "sequential_overlap": args.sequential_overlap
            if args.matching_mode == "sequential"
            else None,
            "quadratic_overlap": args.matching_mode == "sequential",
            "loop_detection": False,
            "reason": f"all-pairs matching is appropriate for {expected_count} images"
            if args.matching_mode == "exhaustive"
            else "continuous one-way clip has no observed return loop",
        },
        "temporal_match_summary": {
            key: value for key, value in temporal_matches.items() if key != "links"
        },
        "reprojection": reprojection,
        "trajectory": trajectory,
        "poses_sha256": digest(poses_path),
        "output_contract": {
            "colmap_sparse_subdirectory": "final_sparse",
            "compatible_consumer": "data_processing/prepare_colmap_artifixer_inputs.py",
            "all_frames_used_for_geometry": True,
            "custom_holdout_rejection": False,
        },
    }
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
