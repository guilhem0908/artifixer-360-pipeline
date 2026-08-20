#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Rotate world-anchored ERP frames into each source camera's view frame.

The input panorama is expressed in the coordinate frame of one reference
camera (normally source frame zero).  For every timestamp this script applies
the complete relative camera rotation on the sphere, so longitude zero follows
the source optical axis and the ERP vertical follows the source camera up axis.
Camera centres are irrelevant because an ERP rotation is purely angular.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from scripts.generate_cubemap_trajectories import normalized_source_poses
except ModuleNotFoundError:
    from generate_cubemap_trajectories import normalized_source_poses


OPENGL_TO_CV = np.diag((1.0, -1.0, -1.0))


def camera_to_reference_cv_rotation(
    reference_c2w: np.ndarray, camera_c2w: np.ndarray
) -> np.ndarray:
    """Map a direction in the current OpenCV camera frame to reference CV."""
    reference = np.asarray(reference_c2w, dtype=np.float64)
    camera = np.asarray(camera_c2w, dtype=np.float64)
    if reference.shape != (4, 4) or camera.shape != (4, 4):
        raise ValueError("camera poses must be 4x4")
    relative_gl = reference[:3, :3].T @ camera[:3, :3]
    relative_cv = OPENGL_TO_CV @ relative_gl @ OPENGL_TO_CV
    if not np.allclose(relative_cv.T @ relative_cv, np.eye(3), atol=1e-5):
        raise ValueError("relative camera rotation is not orthonormal")
    if np.linalg.det(relative_cv) < 0.999:
        raise ValueError("relative camera rotation contains a reflection")
    return relative_cv


def erp_remap(width: int, height: int, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if width != 2 * height or width < 4:
        raise ValueError("ERP geometry must have a positive exact 2:1 aspect ratio")
    longitude = ((np.arange(width, dtype=np.float64) + 0.5) / width) * (2 * np.pi) - np.pi
    latitude = np.pi / 2 - ((np.arange(height, dtype=np.float64) + 0.5) / height) * np.pi
    lon, lat = np.meshgrid(longitude, latitude)
    cos_lat = np.cos(lat)
    directions = np.stack(
        (cos_lat * np.sin(lon), -np.sin(lat), cos_lat * np.cos(lon)), axis=0
    ).reshape(3, -1)
    source = (np.asarray(rotation, dtype=np.float64) @ directions).reshape(3, height, width)
    source /= np.maximum(np.linalg.norm(source, axis=0, keepdims=True), 1e-12)
    source_lon = np.arctan2(source[0], source[2])
    source_lat = -np.arcsin(np.clip(source[1], -1.0, 1.0))
    map_x = ((source_lon + np.pi) / (2 * np.pi)) * width - 0.5
    map_y = ((np.pi / 2 - source_lat) / np.pi) * height - 0.5
    map_x = np.mod(map_x, width).astype(np.float32)
    map_y = np.clip(map_y, 0, height - 1).astype(np.float32)
    return map_x, map_y


def reorient(image: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("ERP input must be a three-channel image")
    height, width = image.shape[:2]
    map_x, map_y = erp_remap(width, height, rotation)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_WRAP,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-transforms", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-frame", type=int, default=0)
    args = parser.parse_args()

    source = json.loads(args.source_transforms.expanduser().read_text())
    poses = normalized_source_poses(source)
    if not 0 <= args.reference_frame < len(poses):
        raise SystemExit(f"reference frame outside [0, {len(poses)})")
    paths = sorted(args.input_dir.expanduser().glob("[0-9][0-9][0-9][0-9][0-9].png"))
    if len(paths) != len(poses):
        raise SystemExit(f"expected {len(poses)} ERP frames, found {len(paths)}")
    expected_names = [f"{index:05d}.png" for index in range(len(poses))]
    if [path.name for path in paths] != expected_names:
        raise SystemExit("ERP frame order or naming drifted")

    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=False)
    reference = np.asarray(poses[args.reference_frame], dtype=np.float64)
    records = []
    for index, (path, pose) in enumerate(zip(paths, poses, strict=True)):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"cannot read {path}")
        rotation = camera_to_reference_cv_rotation(reference, np.asarray(pose))
        aligned = reorient(image, rotation)
        destination = output / f"{index:05d}.png"
        if not cv2.imwrite(str(destination), aligned, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
            raise SystemExit(f"cannot write {destination}")
        records.append(
            {
                "frame": index,
                "source": path.name,
                "rotation_camera_cv_to_reference_cv": rotation.tolist(),
            }
        )
        print(f"[align {index + 1:03d}/{len(paths):03d}] {destination.name}", flush=True)
    metadata = {
        "schema": "artifixer.erp_source_camera_alignment_v1",
        "frame_count": len(records),
        "reference_frame": args.reference_frame,
        "orientation_policy": "longitude_zero_and_erp_up_follow_source_camera_c2w",
        "records": records,
    }
    (output.parent / "source_camera_alignment.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
