#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Generate a smooth world-locked multi-pass trajectory for ArtiFixer.

Camera centres are copied from a contiguous source segment.  Each lane keeps
one fixed world orientation and alternates the direction in which the source
centres are visited.  Consecutive lanes are joined by stationary, gradual
turns at their shared endpoint.  Target frames never carry ``file_path`` so
they cannot be mistaken for real observations.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.generate_cubemap_trajectories import normalized_source_poses
    from scripts.generate_nerfstudio14_trajectories import (
        DEFAULT_FOV_DEGREES,
        DEFAULT_SIZE,
        local_rotation,
        pinhole_intrinsics,
    )
except ModuleNotFoundError:  # Direct execution as ``python scripts/...py``.
    from generate_cubemap_trajectories import normalized_source_poses
    from generate_nerfstudio14_trajectories import (
        DEFAULT_FOV_DEGREES,
        DEFAULT_SIZE,
        local_rotation,
        pinhole_intrinsics,
    )


Orientation = tuple[float, float]


def parse_orientation(value: str) -> Orientation:
    try:
        yaw_text, pitch_text = value.split(":", maxsplit=1)
        yaw, pitch = float(yaw_text), float(pitch_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"orientation must be YAW:PITCH in degrees, got {value!r}"
        ) from error
    if not math.isfinite(yaw) or not math.isfinite(pitch):
        raise argparse.ArgumentTypeError(f"orientation must be finite, got {value!r}")
    if not -90.0 <= pitch <= 90.0:
        raise argparse.ArgumentTypeError(f"pitch must be in [-90, 90], got {pitch}")
    return yaw, pitch


def rotation_distance_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _turn_orientations(
    start: Orientation,
    stop: Orientation,
    *,
    max_turn_degrees: float,
) -> list[Orientation]:
    """Return intermediate orientations, excluding both lane endpoints."""
    if not math.isfinite(max_turn_degrees) or max_turn_degrees <= 0.0:
        raise ValueError("max_turn_degrees must be positive and finite")
    start_rotation = local_rotation(*start)[:3, :3]
    stop_rotation = local_rotation(*stop)[:3, :3]
    angular_distance = rotation_distance_degrees(start_rotation, stop_rotation)
    steps = max(1, math.ceil((angular_distance - 1e-9) / max_turn_degrees))
    while True:
        values = [
            (
                start[0] + (stop[0] - start[0]) * step / steps,
                start[1] + (stop[1] - start[1]) * step / steps,
            )
            for step in range(steps + 1)
        ]
        rotations = [local_rotation(*value)[:3, :3] for value in values]
        maximum_step = max(
            rotation_distance_degrees(first, second)
            for first, second in zip(rotations, rotations[1:])
        )
        if maximum_step <= max_turn_degrees + 1e-7:
            return values[1:-1]
        steps += 1


def generate(
    source: Mapping[str, Any],
    *,
    reference_frame: int,
    start_frame: int,
    frame_count: int,
    orientations: Sequence[Orientation],
    max_turn_degrees: float,
    orientation_names: Sequence[str] | None = None,
    alternate: bool = True,
    size: int = DEFAULT_SIZE,
    fov_degrees: float = DEFAULT_FOV_DEGREES,
) -> tuple[dict[str, object], dict[str, object]]:
    poses = normalized_source_poses(source)
    if not 0 <= reference_frame < len(poses):
        raise ValueError(f"reference_frame must be in [0, {len(poses) - 1}]")
    if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0:
        raise ValueError("start_frame must be a non-negative integer")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    stop_frame = start_frame + frame_count
    if stop_frame > len(poses):
        raise ValueError(
            f"source segment [{start_frame}, {stop_frame}) exceeds {len(poses)} poses"
        )
    if not orientations:
        raise ValueError("at least one orientation is required")
    if orientation_names is None:
        orientation_names = [
            f"Y{yaw % 360:06.2f}_P{pitch:+06.2f}" for yaw, pitch in orientations
        ]
    if len(orientation_names) != len(orientations):
        raise ValueError("orientation_names must match the orientation count")
    if len(set(orientation_names)) != len(orientation_names) or any(
        not name for name in orientation_names
    ):
        raise ValueError("orientation_names must be non-empty and unique")

    reference_rotation = np.asarray(poses[reference_frame], dtype=np.float64)[:3, :3]
    intrinsics = pinhole_intrinsics(size, fov_degrees)
    frames: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    lanes: list[dict[str, object]] = []
    turns: list[dict[str, object]] = []
    previous_endpoint: int | None = None

    def append_frame(
        source_index: int,
        orientation: Orientation,
        *,
        kind: str,
        lane_index: int | None,
    ) -> None:
        source_pose = np.asarray(poses[source_index], dtype=np.float64)
        output = np.eye(4, dtype=np.float64)
        output[:3, :3] = reference_rotation @ local_rotation(*orientation)[:3, :3]
        output[:3, 3] = source_pose[:3, 3]
        target_index = len(frames)
        frames.append({"transform_matrix": output.tolist()})
        entries.append(
            {
                "target_index": target_index,
                "source_index": source_index,
                "kind": kind,
                "lane_index": lane_index,
                "yaw_degrees": orientation[0],
                "pitch_degrees": orientation[1],
            }
        )

    for lane_index, orientation in enumerate(orientations):
        reverse = bool(alternate and lane_index % 2)
        source_indices = list(range(start_frame, stop_frame))
        if reverse:
            source_indices.reverse()

        if previous_endpoint is not None:
            if source_indices[0] != previous_endpoint:
                raise AssertionError("adjacent lanes must share their endpoint centre")
            intermediate = _turn_orientations(
                orientations[lane_index - 1],
                orientation,
                max_turn_degrees=max_turn_degrees,
            )
            turn_start = len(frames)
            for turn_orientation in intermediate:
                append_frame(
                    previous_endpoint,
                    turn_orientation,
                    kind="stationary_turn",
                    lane_index=None,
                )
            turns.append(
                {
                    "from_lane": lane_index - 1,
                    "to_lane": lane_index,
                    "source_index": previous_endpoint,
                    "start": turn_start,
                    "stop": len(frames),
                    "intermediate_count": len(intermediate),
                    "from_orientation": list(orientations[lane_index - 1]),
                    "to_orientation": list(orientation),
                    "from_name": orientation_names[lane_index - 1],
                    "to_name": orientation_names[lane_index],
                }
            )

        lane_start = len(frames)
        for source_index in source_indices:
            append_frame(
                source_index,
                orientation,
                kind="lane",
                lane_index=lane_index,
            )
        lanes.append(
            {
                "lane_index": lane_index,
                "name": orientation_names[lane_index],
                "orientation": list(orientation),
                "direction": "reverse" if reverse else "forward",
                "source_start": source_indices[0],
                "source_stop": source_indices[-1],
                "start": lane_start,
                "stop": len(frames),
                "count": len(source_indices),
            }
        )
        previous_endpoint = source_indices[-1]

    matrices = np.asarray([frame["transform_matrix"] for frame in frames], dtype=np.float64)
    maximum_angular_step = max(
        (
            rotation_distance_degrees(first[:3, :3], second[:3, :3])
            for first, second in zip(matrices, matrices[1:])
        ),
        default=0.0,
    )
    if maximum_angular_step > max_turn_degrees + 1e-6:
        raise AssertionError(
            f"generated angular step {maximum_angular_step:.6f} exceeds {max_turn_degrees}"
        )

    trajectory: dict[str, object] = {
        **intrinsics,
        "frames": frames,
        "world_locked": True,
        "reference_frame": reference_frame,
        "pose_convention": "OpenGL camera-to-world (+X right, +Y up, -Z forward)",
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "layout": "world_locked_spherical_lane_snake",
        "source_frame_count": len(poses),
        "source_segment": {"start": start_frame, "stop": stop_frame, "count": frame_count},
        "reference_frame": reference_frame,
        "world_locked": True,
        "alternate": alternate,
        "orientations": [list(value) for value in orientations],
        "orientation_names": list(orientation_names),
        "max_turn_degrees": max_turn_degrees,
        "maximum_generated_angular_step_degrees": maximum_angular_step,
        "target_count": len(frames),
        "lanes": lanes,
        "turns": turns,
        "frames": entries,
        "intrinsics": intrinsics,
    }
    return trajectory, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument(
        "--orientation",
        dest="orientations",
        action="append",
        type=parse_orientation,
        required=True,
        metavar="YAW:PITCH",
    )
    parser.add_argument(
        "--orientation-name",
        dest="orientation_names",
        action="append",
        default=None,
        help="Stable name for the corresponding --orientation; may be repeated.",
    )
    parser.add_argument("--max-turn-degrees", type=float, default=5.0)
    parser.add_argument("--no-alternate", dest="alternate", action="store_false")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--fov-degrees", type=float, default=DEFAULT_FOV_DEGREES)
    args = parser.parse_args()

    source = json.loads(args.input.expanduser().read_text())
    if not isinstance(source, Mapping):
        raise SystemExit("error: input transforms JSON must be an object")
    try:
        trajectory, manifest = generate(
            source,
            reference_frame=args.reference_frame,
            start_frame=args.start_frame,
            frame_count=args.frame_count,
            orientations=args.orientations,
            orientation_names=args.orientation_names,
            max_turn_degrees=args.max_turn_degrees,
            alternate=args.alternate,
            size=args.size,
            fov_degrees=args.fov_degrees,
        )
    except (ValueError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trajectory, indent=2) + "\n")
    manifest_path = args.manifest or args.output.with_name(f"{args.output.stem}_manifest.json")
    manifest["source"] = str(args.input.expanduser().resolve())
    manifest["trajectory"] = str(args.output.expanduser().resolve())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"target_count={manifest['target_count']}")
    print(f"lane_count={len(manifest['lanes'])} turn_count={len(manifest['turns'])}")
    print(f"maximum_angular_step_degrees={manifest['maximum_generated_angular_step_degrees']:.6f}")
    print("world_locked=true camera_centres_preserved=true")


if __name__ == "__main__":
    main()
