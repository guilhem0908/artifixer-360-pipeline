#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Generate 14 world-locked overlapping perspective trajectories.

Camera centres follow the input trajectory, while all view orientations remain
fixed in one world coordinate system anchored by the first input pose.  This
scene-agnostic policy prevents a source-camera U-turn from changing which world
direction a named frustum represents.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.generate_cubemap_trajectories import normalized_source_poses
    from scripts.generate_nerfstudio14_trajectories import (
        DEFAULT_FOV_DEGREES,
        DEFAULT_SIZE,
        VIEW_SPECS,
        local_rotation,
        pinhole_intrinsics,
    )
except ModuleNotFoundError:  # Direct execution as ``python scripts/...py``.
    from generate_cubemap_trajectories import normalized_source_poses
    from generate_nerfstudio14_trajectories import (
        DEFAULT_FOV_DEGREES,
        DEFAULT_SIZE,
        VIEW_SPECS,
        local_rotation,
        pinhole_intrinsics,
    )


def generate(
    source: Mapping[str, Any],
    *,
    size: int = DEFAULT_SIZE,
    fov_degrees: float = DEFAULT_FOV_DEGREES,
) -> tuple[dict[str, dict[str, object]], dict[str, object], dict[str, object]]:
    source_poses = normalized_source_poses(source)
    if not source_poses:
        raise ValueError("source trajectory contains no poses")
    intrinsics = pinhole_intrinsics(size, fov_degrees)
    reference = np.asarray(source_poses[0], dtype=np.float64)
    reference_rotation = reference[:3, :3]
    frames_per_view = len(source_poses)
    view_order = [name for name, _, _, _ in VIEW_SPECS]
    local_rotations = {
        name: local_rotation(yaw, pitch) for name, yaw, pitch, _ in VIEW_SPECS
    }
    trajectories: dict[str, dict[str, object]] = {}
    blocks: list[dict[str, object]] = []
    groups: dict[str, list[str]] = {}

    for view_index, (name, yaw, pitch, group) in enumerate(VIEW_SPECS):
        fixed_rotation = reference_rotation @ local_rotations[name][:3, :3]
        frames = []
        for source_pose in source_poses:
            output = np.eye(4, dtype=np.float64)
            output[:3, :3] = fixed_rotation
            output[:3, 3] = np.asarray(source_pose, dtype=np.float64)[:3, 3]
            frames.append({"transform_matrix": output.tolist()})
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
        "world_locked": True,
        "reference_policy": "first_frame",
    }
    manifest = {
        "schema_version": 3,
        "layout": "nerfstudio_equirectangular_planar_14_world_locked",
        "pose_convention": "OpenGL camera-to-world (+X right, +Y up, -Z forward)",
        "composition": "view_c2w_rotation = first_source_c2w_rotation @ local_rotation",
        "rotation_convention": "opengl_view_c2w_postmultiply",
        "world_locked": True,
        "reference_policy": "first_frame",
        "semantic_rules": False,
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
        # The stitcher operates in the first-pose coordinate system, so these
        # are deliberately local rather than world rotations.
        "local_rotations": {name: local_rotations[name].tolist() for name in view_order},
        "blocks": blocks,
        "outputs": {
            "views": {name: f"{name}.json" for name in view_order},
            "combined": "combined.json",
            "manifest": "manifest.json",
        },
    }
    return trajectories, combined, manifest


def write_trajectories(
    source_path: Path,
    output_dir: Path,
    *,
    size: int,
    fov_degrees: float,
) -> dict[str, object]:
    source = json.loads(source_path.expanduser().read_text())
    if not isinstance(source, Mapping):
        raise ValueError("input transforms JSON must be an object")
    trajectories, combined, manifest = generate(
        source, size=size, fov_degrees=fov_degrees
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, trajectory in trajectories.items():
        (output_dir / f"{name}.json").write_text(json.dumps(trajectory, indent=2) + "\n")
    (output_dir / "combined.json").write_text(json.dumps(combined, indent=2) + "\n")
    manifest["source"] = str(source_path.expanduser().resolve())
    manifest["output_dir"] = str(output_dir.expanduser().resolve())
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
    print("world_locked=true reference_policy=first_frame semantic_rules=false")
    print("camera_centres_preserved=true")


if __name__ == "__main__":
    main()
