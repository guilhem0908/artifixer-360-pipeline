#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Generate six synchronized cubemap trajectories from OpenGL C2W poses.

Each output face keeps the source frame order and camera center.  The combined
trajectory stores one contiguous frame block per face in F/R/B/L/U/D order so
rendered images can be split without inspecting per-frame metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


FACE_ORDER = ("F", "R", "B", "L", "U", "D")
DEFAULT_SIZE = 1216
DEFAULT_FOV_DEGREES = 100.0

# OpenGL camera axes are +X right, +Y up, and -Z forward.  These matrices are
# post-multiplied onto each source C2W pose, i.e. face_c2w = source_c2w @ local.
# Keeping them integer-valued avoids small sin(pi/2) residuals at face seams.
LOCAL_ROTATIONS: dict[str, np.ndarray] = {
    "F": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    ),
    "R": np.array(
        [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    ),
    "B": np.array(
        [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    ),
    "L": np.array(
        [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    ),
    "U": np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    ),
    "D": np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    ),
}


def cubemap_intrinsics(size: int, fov_degrees: float) -> dict[str, object]:
    """Return centered square pinhole intrinsics for the requested field of view."""
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"size must be a positive integer, got {size!r}")
    if not math.isfinite(fov_degrees) or not 0.0 < fov_degrees < 180.0:
        raise ValueError(f"fov_degrees must be finite and between 0 and 180, got {fov_degrees!r}")

    focal_length = (size / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
    principal_point = size / 2.0
    return {
        "camera_model": "OPENCV",
        "w": size,
        "h": size,
        "fl_x": focal_length,
        "fl_y": focal_length,
        "cx": principal_point,
        "cy": principal_point,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }


def _homogeneous_matrix(value: object, label: str, *, allow_3x4: bool = False) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain numeric values") from error
    if allow_3x4 and matrix.shape == (3, 4):
        matrix = np.vstack((matrix, np.array([0.0, 0.0, 0.0, 1.0])))
    if matrix.shape != (4, 4):
        expected = "3x4 or 4x4" if allow_3x4 else "4x4"
        raise ValueError(f"{label} must be {expected}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} contains non-finite values")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8, rtol=0.0):
        raise ValueError(f"{label} has an invalid homogeneous row")
    return matrix


def normalized_source_poses(source: Mapping[str, Any]) -> list[np.ndarray]:
    """Read source OpenGL C2W poses and materialize Nerfstudio applied_transform."""
    frames = source.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("input transforms must contain a non-empty frames list")

    applied_transform = np.eye(4, dtype=np.float64)
    if "applied_transform" in source:
        applied_transform = _homogeneous_matrix(
            source["applied_transform"], "applied_transform", allow_3x4=True
        )
        try:
            inverse_applied_transform = np.linalg.inv(applied_transform)
        except np.linalg.LinAlgError as error:
            raise ValueError("applied_transform must be invertible") from error
    else:
        inverse_applied_transform = applied_transform

    poses: list[np.ndarray] = []
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"frame {frame_index} must be an object")
        if "transform_matrix" not in frame:
            raise ValueError(f"frame {frame_index} is missing transform_matrix")
        pose = _homogeneous_matrix(frame["transform_matrix"], f"frame {frame_index} transform_matrix")
        poses.append(inverse_applied_transform @ pose)
    return poses


def generate_trajectories(
    source: Mapping[str, Any],
    *,
    size: int = DEFAULT_SIZE,
    fov_degrees: float = DEFAULT_FOV_DEGREES,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Build per-face and combined transforms dictionaries in canonical block order."""
    intrinsics = cubemap_intrinsics(size, fov_degrees)
    source_poses = normalized_source_poses(source)
    trajectories: dict[str, dict[str, object]] = {}

    for face in FACE_ORDER:
        local_rotation = LOCAL_ROTATIONS[face]
        frames = [
            {"transform_matrix": (source_pose @ local_rotation).tolist()}
            for source_pose in source_poses
        ]
        trajectories[face] = {**intrinsics, "frames": frames}

    combined = {
        **intrinsics,
        "frames": [
            frame
            for face in FACE_ORDER
            for frame in trajectories[face]["frames"]
        ],
    }
    return trajectories, combined


def build_manifest(
    *,
    source_path: Path,
    output_dir: Path,
    frames_per_face: int,
    size: int,
    fov_degrees: float,
) -> dict[str, object]:
    """Describe cubemap conventions, output files, and combined frame blocks."""
    intrinsics = cubemap_intrinsics(size, fov_degrees)
    blocks = []
    face_outputs: dict[str, str] = {}
    for face_index, face in enumerate(FACE_ORDER):
        start = face_index * frames_per_face
        stop = start + frames_per_face
        face_outputs[face] = f"{face}.json"
        blocks.append(
            {
                "face": face,
                "start": start,
                "stop": stop,
                "count": frames_per_face,
                "trajectory": f"{face}.json",
            }
        )

    return {
        "schema_version": 1,
        "source": str(source_path.expanduser().resolve()),
        "output_dir": str(output_dir.expanduser().resolve()),
        "pose_convention": "OpenGL camera-to-world (+X right, +Y up, -Z forward)",
        "composition": "face_c2w = source_c2w @ local_rotation",
        "rotation_convention": "opengl_face_c2w_postmultiply",
        "face_order": list(FACE_ORDER),
        "frames_per_face": frames_per_face,
        "total_frames": frames_per_face * len(FACE_ORDER),
        "size": size,
        "fov": float(fov_degrees),
        "fov_degrees": float(fov_degrees),
        "fx": intrinsics["fl_x"],
        "fy": intrinsics["fl_y"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "local_rotations": {
            face: LOCAL_ROTATIONS[face].tolist() for face in FACE_ORDER
        },
        "outputs": {
            "faces": face_outputs,
            "combined": "combined.json",
            "manifest": "manifest.json",
        },
        "blocks": blocks,
    }


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_cubemap_trajectories(
    source_path: Path,
    output_dir: Path,
    *,
    size: int = DEFAULT_SIZE,
    fov_degrees: float = DEFAULT_FOV_DEGREES,
) -> dict[str, object]:
    """Generate and write all trajectory products, returning the manifest."""
    source_path = source_path.expanduser()
    output_dir = output_dir.expanduser()
    try:
        source = json.loads(source_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {source_path}: {error}") from error
    if not isinstance(source, Mapping):
        raise ValueError("input transforms JSON must be an object")

    trajectories, combined = generate_trajectories(
        source, size=size, fov_degrees=fov_degrees
    )
    frames_per_face = len(trajectories[FACE_ORDER[0]]["frames"])
    manifest = build_manifest(
        source_path=source_path,
        output_dir=output_dir,
        frames_per_face=frames_per_face,
        size=size,
        fov_degrees=fov_degrees,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for face in FACE_ORDER:
        _write_json(output_dir / f"{face}.json", trajectories[face])
    _write_json(output_dir / "combined.json", combined)
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source transforms-style JSON")
    parser.add_argument("output_dir", type=Path, help="directory for all generated JSON files")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="square face size in pixels")
    parser.add_argument(
        "--fov-degrees",
        type=float,
        default=DEFAULT_FOV_DEGREES,
        help="horizontal and vertical pinhole field of view",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        manifest = write_cubemap_trajectories(
            args.input,
            args.output_dir,
            size=args.size,
            fov_degrees=args.fov_degrees,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    print(f"output_dir={Path(manifest['output_dir'])}")
    print(f"face_order={','.join(manifest['face_order'])}")
    print(f"frames_per_face={manifest['frames_per_face']}")
    print(f"combined_frames={manifest['total_frames']}")
    print(f"size={manifest['size']}")
    print(f"fov_degrees={manifest['fov_degrees']}")
    print("camera_centers_preserved=true")


if __name__ == "__main__":
    main()
