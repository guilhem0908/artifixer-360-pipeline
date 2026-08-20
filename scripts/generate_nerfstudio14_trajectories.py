#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Generate Nerfstudio-style 14-view trajectories from OpenGL C2W poses.

The sampling follows Nerfstudio's 14 planar projections: six level cameras at
60-degree yaw intervals, four cameras pitched 45 degrees up, and four pitched
45 degrees down.  Every view preserves the source camera centre and frame
order.  The combined trajectory is view-major so one 154-frame ArtiFixer chunk
corresponds to exactly one direction.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.generate_cubemap_trajectories import normalized_source_poses
except ModuleNotFoundError:  # Direct execution as ``python scripts/...py``.
    from generate_cubemap_trajectories import normalized_source_poses


DEFAULT_SIZE = 768
DEFAULT_FOV_DEGREES = 110.0
VIEW_SPECS: tuple[tuple[str, float, float, str], ...] = (
    ("H000", 0.0, 0.0, "horizontal_a"),
    ("H060", 60.0, 0.0, "horizontal_b"),
    ("H120", 120.0, 0.0, "horizontal_a"),
    ("H180", 180.0, 0.0, "horizontal_b"),
    ("H240", 240.0, 0.0, "horizontal_a"),
    ("H300", 300.0, 0.0, "horizontal_b"),
    ("U000", 0.0, 45.0, "upper"),
    ("U090", 90.0, 45.0, "upper"),
    ("U180", 180.0, 45.0, "upper"),
    ("U270", 270.0, 45.0, "upper"),
    ("D000", 0.0, -45.0, "lower"),
    ("D090", 90.0, -45.0, "lower"),
    ("D180", 180.0, -45.0, "lower"),
    ("D270", 270.0, -45.0, "lower"),
)


def pinhole_intrinsics(size: int, fov_degrees: float) -> dict[str, object]:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"size must be a positive integer, got {size!r}")
    if not math.isfinite(fov_degrees) or not 0.0 < fov_degrees < 180.0:
        raise ValueError(f"fov_degrees must be between 0 and 180, got {fov_degrees!r}")
    focal = (size / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
    return {
        "camera_model": "OPENCV",
        "w": size,
        "h": size,
        "fl_x": focal,
        "fl_y": focal,
        "cx": size / 2.0,
        "cy": size / 2.0,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }


def local_rotation(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    """Return an OpenGL camera-local rotation (positive yaw right, pitch up)."""
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    yaw_right = np.array(
        [
            [math.cos(yaw), 0.0, -math.sin(yaw), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [math.sin(yaw), 0.0, math.cos(yaw), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pitch_up = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch), 0.0],
            [0.0, math.sin(pitch), math.cos(pitch), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return yaw_right @ pitch_up


def generate(
    source: Mapping[str, Any], *, size: int = DEFAULT_SIZE, fov_degrees: float = DEFAULT_FOV_DEGREES
) -> tuple[dict[str, dict[str, object]], dict[str, object], dict[str, object]]:
    intrinsics = pinhole_intrinsics(size, fov_degrees)
    source_poses = normalized_source_poses(source)
    view_order = [name for name, _, _, _ in VIEW_SPECS]
    rotations = {
        name: local_rotation(yaw, pitch) for name, yaw, pitch, _ in VIEW_SPECS
    }
    trajectories: dict[str, dict[str, object]] = {}
    blocks: list[dict[str, object]] = []
    groups: dict[str, list[str]] = {}
    frames_per_view = len(source_poses)
    for view_index, (name, yaw, pitch, group) in enumerate(VIEW_SPECS):
        frames = [
            {"transform_matrix": (source_pose @ rotations[name]).tolist()}
            for source_pose in source_poses
        ]
        trajectories[name] = {**intrinsics, "frames": frames}
        groups.setdefault(group, []).append(name)
        blocks.append(
            {
                "view": name,
                "yaw_degrees": yaw,
                "pitch_degrees": pitch,
                "group": group,
                "start": view_index * frames_per_view,
                "stop": (view_index + 1) * frames_per_view,
                "count": frames_per_view,
                "trajectory": f"{name}.json",
            }
        )
    combined = {
        **intrinsics,
        "frames": [frame for name in view_order for frame in trajectories[name]["frames"]],
    }
    manifest = {
        "schema_version": 2,
        "layout": "nerfstudio_equirectangular_planar_14",
        "pose_convention": "OpenGL camera-to-world (+X right, +Y up, -Z forward)",
        "composition": "view_c2w = source_c2w @ local_rotation",
        "rotation_convention": "opengl_view_c2w_postmultiply",
        "view_order": view_order,
        "frames_per_view": frames_per_view,
        "total_frames": frames_per_view * len(view_order),
        "size": size,
        "fov_degrees": float(fov_degrees),
        "fx": intrinsics["fl_x"],
        "fy": intrinsics["fl_y"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "groups": groups,
        "local_rotations": {name: rotations[name].tolist() for name in view_order},
        "blocks": blocks,
        "outputs": {
            "views": {name: f"{name}.json" for name in view_order},
            "combined": "combined.json",
            "manifest": "manifest.json",
        },
    }
    return trajectories, combined, manifest


def write_trajectories(source_path: Path, output_dir: Path, *, size: int, fov_degrees: float) -> dict[str, object]:
    source = json.loads(source_path.expanduser().read_text())
    if not isinstance(source, Mapping):
        raise ValueError("input transforms JSON must be an object")
    trajectories, combined, manifest = generate(source, size=size, fov_degrees=fov_degrees)
    manifest["source"] = str(source_path.expanduser().resolve())
    manifest["output_dir"] = str(output_dir.expanduser().resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, trajectory in trajectories.items():
        (output_dir / f"{name}.json").write_text(json.dumps(trajectory, indent=2) + "\n")
    (output_dir / "combined.json").write_text(json.dumps(combined, indent=2) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--fov-degrees", type=float, default=DEFAULT_FOV_DEGREES)
    args = parser.parse_args()
    try:
        manifest = write_trajectories(
            args.input, args.output_dir, size=args.size, fov_degrees=args.fov_degrees
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from error
    print(f"view_order={','.join(manifest['view_order'])}")
    print(f"groups={json.dumps(manifest['groups'], separators=(',', ':'))}")
    print(f"frames_per_view={manifest['frames_per_view']}")
    print(f"total_frames={manifest['total_frames']}")
    print(f"size={manifest['size']} fov_degrees={manifest['fov_degrees']}")
    print("camera_centres_preserved=true")


if __name__ == "__main__":
    main()
