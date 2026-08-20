#!/usr/bin/env python3
"""Run the preregistered COLMAP incremental geometry benchmark candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


EXPECTED_COUNT = 190
COLMAP_MAX_IMAGE_ID = 2**31 - 1


class GeometryError(RuntimeError):
    """Raised when a preregistered technical or scientific gate fails."""


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GeometryError(f"JSON document is not an object: {path}")
    return value


class Commands:
    def __init__(
        self,
        log_path: Path,
        *,
        container: Path,
        bind: Path,
        cudnn_dir: Path,
    ) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.prefix = [
            "singularity",
            "exec",
            "--nv",
            "--bind",
            f"{bind}:{bind}",
            "--bind",
            f"{cudnn_dir}:/opt/roomtour-cudnn:ro",
            "--env",
            "LD_LIBRARY_PATH=/opt/roomtour-cudnn:/usr/local/cuda/lib64:/.singularity.d/libs",
            str(container),
            "colmap",
        ]

    def run(self, arguments: Sequence[str]) -> None:
        command = [*self.prefix, *arguments]
        command_name = arguments[0] if arguments else "unknown"
        print(f"COLMAP_COMMAND_START={command_name}", flush=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(command, separators=(",", ":")) + "\n")
            stream.flush()
            try:
                subprocess.run(command, check=True, stdout=stream, stderr=stream)
            except subprocess.CalledProcessError:
                payload = self.log_path.read_bytes()[-65536:]
                print("COLMAP_COMMAND_FAILED_LAST_LOG_BYTES_BEGIN", file=sys.stderr)
                print(payload.decode("utf-8", errors="replace"), file=sys.stderr)
                print("COLMAP_COMMAND_FAILED_LAST_LOG_BYTES_END", file=sys.stderr)
                raise
        print(f"COLMAP_COMMAND_COMPLETE={command_name}", flush=True)


def load_inputs(
    images_dir: Path, input_manifest_path: Path, protocol_path: Path
) -> tuple[list[str], list[str], dict[str, Any]]:
    manifest = read_json(input_manifest_path)
    protocol = read_json(protocol_path)
    if (
        manifest.get("status") != "PASS"
        or manifest.get("verdict") != "PASS_COMPACT_PERSPECTIVE_INPUTS"
        or manifest.get("observation_count") != EXPECTED_COUNT
        or manifest.get("ground_truth_erp_accessed") is not False
        or protocol.get("status") != "LOCKED"
        or protocol.get("verdict")
        != "PREREGISTERED_CONTINUOUS_PERSPECTIVE_GEOMETRY_V1"
        or protocol.get("upstream", {}).get("colmap_input_manifest_sha256")
        != digest(input_manifest_path)
    ):
        raise GeometryError("input manifest or geometry protocol is not exact PASS/LOCKED")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_COUNT:
        raise GeometryError("input manifest does not contain 190 records")
    names = []
    train_names = []
    for index, record in enumerate(records):
        name = f"frame_{index:06d}.jpg"
        if record.get("index") != index or record.get("name") != name:
            raise GeometryError("COLMAP input order or name drifted")
        path = images_dir / name
        if (
            not path.is_file()
            or path.stat().st_size != record.get("jpeg_size_bytes")
            or digest(path) != record.get("jpeg_sha256")
        ):
            raise GeometryError(f"COLMAP input image drifted: {path}")
        names.append(name)
        if not record.get("appearance_holdout"):
            train_names.append(name)
    entries = sorted(path.name for path in images_dir.iterdir() if path.is_file())
    if entries != names or len(train_names) != 171:
        raise GeometryError("COLMAP image root or 171/19 split drifted")
    return names, train_names, protocol


def pair_id(first: int, second: int) -> int:
    lower, upper = sorted((first, second))
    return lower * COLMAP_MAX_IMAGE_ID + upper


def _array(blob: bytes | None, dtype: str, rows: int, cols: int) -> np.ndarray:
    if blob is None or rows <= 0 or cols <= 0:
        return np.empty((0, cols), dtype=np.dtype(dtype))
    value = np.frombuffer(blob, dtype=dtype)
    if value.size != rows * cols:
        raise GeometryError("COLMAP database array has inconsistent shape")
    return value.reshape(rows, cols)


def _grid_coverage(points: np.ndarray, width: int, height: int) -> int:
    if len(points) == 0:
        return 0
    columns = np.clip((points[:, 0] * 8 / width).astype(np.int64), 0, 7)
    rows = np.clip((points[:, 1] * 5 / height).astype(np.int64), 0, 4)
    return len(set((int(row), int(column)) for row, column in zip(rows, columns, strict=True)))


def consecutive_link_report(
    database: Path,
    names: Sequence[str],
    *,
    width: int,
    height: int,
    minimum_inliers: int,
    minimum_ratio: float,
    minimum_cells: int,
) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    try:
        image_ids = {
            str(name): int(image_id)
            for image_id, name in connection.execute("SELECT image_id,name FROM images")
        }
        keypoints = {}
        for image_id, rows, cols, blob in connection.execute(
            "SELECT image_id,rows,cols,data FROM keypoints"
        ):
            keypoints[int(image_id)] = _array(blob, "<f4", int(rows), int(cols))[:, :2]
        raw_rows = {
            int(value): int(rows)
            for value, rows in connection.execute("SELECT pair_id,rows FROM matches")
        }
        geometries = {
            int(value): (int(rows), blob)
            for value, rows, blob in connection.execute(
                "SELECT pair_id,rows,data FROM two_view_geometries"
            )
        }
    finally:
        connection.close()
    reports = []
    failures = []
    for first_name, second_name in zip(names, names[1:]):
        first_id = image_ids[first_name]
        second_id = image_ids[second_name]
        identifier = pair_id(first_id, second_id)
        inlier_rows, blob = geometries.get(identifier, (0, None))
        inliers = _array(blob, "<u4", inlier_rows, 2)
        if first_id < second_id:
            first_indices, second_indices = inliers[:, 0], inliers[:, 1]
        else:
            first_indices, second_indices = inliers[:, 1], inliers[:, 0]
        first_points = keypoints[first_id][first_indices] if len(inliers) else np.empty((0, 2))
        second_points = keypoints[second_id][second_indices] if len(inliers) else np.empty((0, 2))
        raw = raw_rows.get(identifier, 0)
        ratio = inlier_rows / raw if raw else 0.0
        first_cells = _grid_coverage(first_points, width, height)
        second_cells = _grid_coverage(second_points, width, height)
        passed = (
            inlier_rows >= minimum_inliers
            and ratio >= minimum_ratio
            and first_cells >= minimum_cells
            and second_cells >= minimum_cells
        )
        record = {
            "pair": [first_name, second_name],
            "raw_matches": raw,
            "true_ransac_inliers": inlier_rows,
            "inlier_ratio": ratio,
            "covered_grid_cells": [first_cells, second_cells],
            "pass": passed,
        }
        reports.append(record)
        if not passed:
            failures.append(record)
    return {
        "pair_count": len(reports),
        "thresholds": {
            "minimum_inliers": minimum_inliers,
            "minimum_ratio": minimum_ratio,
            "minimum_cells": minimum_cells,
            "grid_cell_count": 40,
        },
        "failures": failures,
        "links": reports,
        "pass": not failures and len(reports) == EXPECTED_COUNT - 1,
    }


def largest_model(root: Path) -> tuple[Path, Any]:
    import pycolmap

    candidates = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        try:
            reconstruction = pycolmap.Reconstruction(path)
        except Exception:
            continue
        candidates.append((reconstruction.num_reg_images(), path.name, path, reconstruction))
    if not candidates:
        raise GeometryError(f"COLMAP produced no readable sparse model under {root}")
    _, _, path, reconstruction = max(candidates, key=lambda value: (value[0], value[1]))
    return path, reconstruction


def run_mapper(
    commands: Commands,
    *,
    database: Path,
    images_dir: Path,
    output: Path,
    image_list: Path | None,
) -> tuple[Path, Any]:
    output.mkdir(parents=True, exist_ok=False)
    arguments = [
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
        "--Mapper.min_num_matches",
        "15",
        "--Mapper.init_min_num_inliers",
        "100",
        "--Mapper.abs_pose_min_num_inliers",
        "100",
        "--Mapper.abs_pose_max_error",
        "4",
        "--Mapper.filter_max_reproj_error",
        "4",
        "--Mapper.filter_min_tri_angle",
        "1.5",
        "--Mapper.ba_refine_focal_length",
        "0",
        "--Mapper.ba_refine_principal_point",
        "0",
        "--Mapper.ba_refine_extra_params",
        "0",
    ]
    if image_list is not None:
        arguments.extend(["--Mapper.image_list_path", str(image_list)])
    commands.run(arguments)
    return largest_model(output)


def reprojection_metrics(reconstruction: Any, validation_names: Iterable[str]) -> dict[str, Any]:
    image_by_name = {
        image.name: image
        for image in reconstruction.images.values()
        if image.has_pose
    }
    all_errors = []
    per_image = []
    for name in validation_names:
        image = image_by_name.get(name)
        if image is None:
            per_image.append({"name": name, "localized": False, "observation_count": 0})
            continue
        camera = reconstruction.cameras[image.camera_id]
        errors = []
        for point2d in image.points2D:
            if not point2d.has_point3D():
                continue
            point3d = reconstruction.points3D[point2d.point3D_id]
            point_camera = image.cam_from_world() * point3d.xyz
            if float(point_camera[2]) <= 0:
                continue
            projected = camera.img_from_cam(point_camera)
            if projected is None:
                continue
            error = float(np.linalg.norm(np.asarray(projected) - point2d.xy))
            if math.isfinite(error):
                errors.append(error)
        all_errors.extend(errors)
        per_image.append(
            {
                "name": name,
                "localized": True,
                "observation_count": len(errors),
                "median_px": float(np.median(errors)) if errors else None,
                "p95_px": float(np.percentile(errors, 95)) if errors else None,
            }
        )
    localized = sum(bool(record["localized"]) for record in per_image)
    if localized != len(per_image) or not all_errors:
        raise GeometryError(f"localized {localized}/{len(per_image)} validation images")
    return {
        "localized_count": localized,
        "expected_count": len(per_image),
        "observation_count": len(all_errors),
        "median_px": float(np.median(all_errors)),
        "p95_px": float(np.percentile(all_errors, 95)),
        "images": per_image,
    }


def pose_matrix(image: Any) -> np.ndarray:
    transform = image.cam_from_world
    if callable(transform):
        transform = transform()
    matrix = transform.matrix
    if callable(matrix):
        matrix = matrix()
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape == (3, 4):
        value = np.vstack([value, [0.0, 0.0, 0.0, 1.0]])
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise GeometryError(f"invalid pose matrix for {image.name}")
    return value


def rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    cosine = (float(np.trace(first.T @ second)) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def export_poses_and_trajectory(reconstruction: Any, names: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_name = {image.name: image for image in reconstruction.images.values() if image.has_pose}
    if sorted(by_name) != sorted(names):
        raise GeometryError("final COLMAP pose set is incomplete")
    records = []
    matrices = []
    for name in names:
        world_to_camera = pose_matrix(by_name[name])
        camera_to_world = np.linalg.inv(world_to_camera)
        if (
            not np.isfinite(camera_to_world).all()
            or np.linalg.det(camera_to_world[:3, :3]) <= 0
        ):
            raise GeometryError(f"non-proper final camera pose: {name}")
        matrices.append(camera_to_world)
        records.append(
            {
                "name": name,
                "world_to_camera": world_to_camera.tolist(),
                "camera_to_world": camera_to_world.tolist(),
            }
        )
    matrices_array = np.stack(matrices)
    centers = matrices_array[:, :3, 3]
    translations = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    rotations = np.asarray(
        [
            rotation_angle(matrices_array[i, :3, :3], matrices_array[i + 1, :3, :3])
            for i in range(len(matrices_array) - 1)
        ]
    )
    translation_median = float(np.median(translations))
    rotation_median = float(np.median(rotations))
    translation_ratio = float(translations.max() / translation_median) if translation_median > 0 else math.inf
    rotation_ratio = float(rotations.max() / rotation_median) if rotation_median > 0 else math.inf
    trajectory = {
        "translation_step_median": translation_median,
        "translation_step_max_over_median": translation_ratio,
        "rotation_step_median_deg": rotation_median,
        "rotation_step_max_over_median": rotation_ratio,
        "maximum_allowed_ratio": 5.0,
        "pass": translation_ratio <= 5.0 and rotation_ratio <= 5.0,
    }
    camera = next(iter(reconstruction.cameras.values()))
    pose_document = {
        "schema": "artifixer.erp_benchmark_pose_sequence_v1",
        "schema_version": 1,
        "solver_id": "colmap_incremental",
        "pose_image_count": EXPECTED_COUNT,
        "camera": {
            "model": str(camera.model_name),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in np.asarray(camera.params)],
        },
        "poses": records,
    }
    return pose_document, trajectory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--commands-log", type=Path, required=True)
    parser.add_argument("--colmap-container", type=Path, required=True)
    parser.add_argument("--colmap-bind", type=Path, required=True)
    parser.add_argument("--colmap-cudnn-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (args.output_dir, args.metrics, args.poses, args.commands_log):
        if path.expanduser().resolve().exists():
            raise GeometryError(f"refusing to replace solver output: {path}")
    images_dir = args.images_dir.expanduser().resolve()
    input_manifest = args.input_manifest.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    names, train_names, protocol = load_inputs(images_dir, input_manifest, protocol_path)
    output.mkdir(parents=True)
    camera = protocol["camera"]
    if (
        camera.get("model") != "PINHOLE"
        or camera.get("shared") is not True
        or camera.get("width") != 1024
        or camera.get("height") != 1024
        or camera.get("params") != [512.0000000000001, 512.0000000000001, 511.5, 511.5]
        or camera.get("focal_frozen") is not True
        or camera.get("principal_point_frozen") is not True
        or camera.get("extra_params_frozen") is not True
    ):
        raise GeometryError("frozen PINHOLE protocol drifted")
    camera_params = ",".join(format(float(value), ".17g") for value in camera["params"])
    commands = Commands(
        args.commands_log.expanduser().resolve(),
        container=args.colmap_container.expanduser().resolve(),
        bind=args.colmap_bind.expanduser().resolve(),
        cudnn_dir=args.colmap_cudnn_dir.expanduser().resolve(),
    )
    database = output / "database.db"
    commands.run(
        [
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_params",
            camera_params,
            "--FeatureExtraction.type",
            "ALIKED_N16ROT",
            "--AlikedExtraction.max_num_features",
            "8192",
            "--FeatureExtraction.gpu_index",
            "0",
            "--default_random_seed",
            "0",
        ]
    )
    commands.run(
        [
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
    )
    link_protocol = protocol["validation"]["consecutive_links"]
    links = consecutive_link_report(
        database,
        names,
        width=int(camera["width"]),
        height=int(camera["height"]),
        minimum_inliers=int(link_protocol["true_ransac_inliers_per_pair_min"]),
        minimum_ratio=float(link_protocol["inlier_ratio_per_pair_min"]),
        minimum_cells=int(link_protocol["covered_grid_cells_per_image_min"]),
    )
    (output / "consecutive_links.json").write_text(
        json.dumps(links, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not links["pass"]:
        raise GeometryError(
            f"{len(links['failures'])}/{EXPECTED_COUNT - 1} consecutive links failed locked gates"
        )
    train_list = output / "appearance_train_names.txt"
    train_list.write_text("".join(f"{name}\n" for name in train_names), encoding="utf-8")
    train_model_path, train_model = run_mapper(
        commands,
        database=database,
        images_dir=images_dir,
        output=output / "train_sparse_candidates",
        image_list=train_list,
    )
    if train_model.num_reg_images() != 171:
        raise GeometryError(f"train-only mapper registered {train_model.num_reg_images()}/171 images")
    localized_root = output / "localized_all"
    localized_root.mkdir()
    commands.run(
        [
            "image_registrator",
            "--database_path",
            str(database),
            "--input_path",
            str(train_model_path),
            "--output_path",
            str(localized_root),
            "--Mapper.fix_existing_frames",
            "1",
            "--Mapper.abs_pose_max_error",
            "4",
            "--Mapper.abs_pose_min_num_inliers",
            "100",
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
            "--default_random_seed",
            "0",
        ]
    )
    import pycolmap

    localized = pycolmap.Reconstruction(localized_root)
    holdout_names = [name for name in names if name not in set(train_names)]
    holdout_metrics = reprojection_metrics(localized, holdout_names)
    holdout_gate = protocol["validation"]["holdout_pose_localization"]
    if (
        localized.num_reg_images() != EXPECTED_COUNT
        or holdout_metrics["median_px"] > holdout_gate["median_reprojection_px_max"]
        or holdout_metrics["p95_px"] > holdout_gate["p95_reprojection_px_max"]
    ):
        raise GeometryError("appearance-holdout pose localization failed locked gates")
    final_path, final_model = run_mapper(
        commands,
        database=database,
        images_dir=images_dir,
        output=output / "final_sparse_candidates",
        image_list=None,
    )
    if final_model.num_reg_images() != EXPECTED_COUNT:
        raise GeometryError(f"final mapper registered {final_model.num_reg_images()}/190 images")
    cameras = list(final_model.cameras.values())
    if len(cameras) != 1 or str(cameras[0].model_name) != "PINHOLE":
        raise GeometryError("final shared PINHOLE camera drifted")
    observed_params = np.asarray(cameras[0].params, dtype=np.float64)
    if not np.allclose(observed_params, np.asarray(camera["params"]), atol=1e-9, rtol=0):
        raise GeometryError("final frozen camera intrinsics drifted")
    final_metrics = reprojection_metrics(final_model, names)
    final_gate = protocol["validation"]["final_incremental"]
    if (
        final_metrics["median_px"] > final_gate["median_reprojection_px_max"]
        or final_metrics["p95_px"] > final_gate["p95_reprojection_px_max"]
    ):
        raise GeometryError("final reprojection metrics failed locked gates")
    poses, trajectory = export_poses_and_trajectory(final_model, names)
    if not trajectory["pass"]:
        raise GeometryError("final trajectory contains a >5x undeclared jump")
    args.poses.expanduser().resolve().write_text(
        json.dumps(poses, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    final_sparse = output / "final_sparse"
    final_sparse.mkdir()
    final_model.write(final_sparse)
    metrics = {
        "schema": "artifixer.erp_benchmark_colmap_metrics_v1",
        "schema_version": 1,
        "technical_status": "PASS",
        "solver_id": "colmap_incremental",
        "registered_images": final_model.num_reg_images(),
        "point3D_count": len(final_model.points3D),
        "shared_camera": True,
        "camera_model": "PINHOLE",
        "intrinsics_frozen_and_identical": True,
        "consecutive_links": links,
        "appearance_holdout_localization": holdout_metrics,
        "final_reprojection": final_metrics,
        "trajectory": trajectory,
        "poses_sha256": digest(args.poses.expanduser().resolve()),
        "ground_truth_erp_accessed": False,
        "scientific_consensus_issued": False,
    }
    args.metrics.expanduser().resolve().write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
