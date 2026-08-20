#!/usr/bin/env python3
"""Generate a smooth, forward-facing 14-view trajectory from COLMAP poses.

The camera centres are cleaned, smoothed, and resampled at uniform arc length.
The H000 camera then follows the local path tangent with a stable scene up
vector.  The other thirteen frustums retain fixed rotations relative to H000,
so every timestamp still forms one closed panoramic rig.
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


def _normalize(vector: np.ndarray, *, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"cannot normalize {label}")
    return np.asarray(vector, dtype=np.float64) / norm


def step_statistics(centres: np.ndarray) -> dict[str, float]:
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    positive = steps[steps > 1e-12]
    if not positive.size:
        return {
            "median_positive": 0.0,
            "p95": 0.0,
            "maximum": 0.0,
            "maximum_over_median": 0.0,
        }
    median = float(np.median(positive))
    return {
        "median_positive": median,
        "p95": float(np.percentile(positive, 95)),
        "maximum": float(np.max(positive)),
        "maximum_over_median": float(np.max(positive) / median),
    }


def repair_isolated_spikes(
    centres: np.ndarray,
    *,
    jump_ratio: float = 5.0,
    reversal_cosine: float = -0.5,
) -> tuple[np.ndarray, list[int]]:
    """Replace isolated COLMAP centre spikes bracketed by opposite large jumps."""
    if not np.isfinite(jump_ratio) or jump_ratio <= 1.0:
        raise ValueError("jump_ratio must be finite and greater than one")
    result = np.asarray(centres, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 3:
        raise ValueError("centres must have shape [N,3] with N >= 3")
    positive = np.linalg.norm(np.diff(result, axis=0), axis=1)
    positive = positive[positive > 1e-12]
    if not positive.size:
        return result, []
    threshold = float(np.median(positive)) * jump_ratio
    repaired: list[int] = []
    for index in range(1, len(result) - 1):
        incoming = result[index] - result[index - 1]
        outgoing = result[index + 1] - result[index]
        incoming_norm = float(np.linalg.norm(incoming))
        outgoing_norm = float(np.linalg.norm(outgoing))
        if incoming_norm <= threshold or outgoing_norm <= threshold:
            continue
        cosine = float(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm))
        direct = float(np.linalg.norm(result[index + 1] - result[index - 1]))
        if cosine <= reversal_cosine and direct <= 2.0 * threshold:
            result[index] = 0.5 * (result[index - 1] + result[index + 1])
            repaired.append(index)
    return result, repaired


def binomial_smooth(
    centres: np.ndarray,
    *,
    iterations: int,
) -> np.ndarray:
    """Apply a short symmetric low-pass while preserving both endpoints."""
    if iterations < 0:
        raise ValueError("smooth iterations must be non-negative")
    result = np.asarray(centres, dtype=np.float64).copy()
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
    for _ in range(iterations):
        padded = np.pad(result, ((2, 2), (0, 0)), mode="edge")
        filtered = sum(kernel[offset] * padded[offset : offset + len(result)] for offset in range(5))
        filtered[0] = result[0]
        filtered[-1] = result[-1]
        result = filtered
    return result


def resample_uniform_arclength(centres: np.ndarray, count: int) -> np.ndarray:
    """Resample a polyline so synthetic camera speed is approximately constant."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("output frame count must be an integer >= 2")
    centres = np.asarray(centres, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-12))
    cumulative = cumulative[keep]
    unique_centres = centres[keep]
    if len(unique_centres) < 2 or cumulative[-1] <= 1e-12:
        raise ValueError("trajectory centres do not span a usable path")
    samples = np.linspace(0.0, float(cumulative[-1]), count)
    return np.stack(
        [np.interp(samples, cumulative, unique_centres[:, axis]) for axis in range(3)],
        axis=1,
    )


def smooth_forward_centres(
    centres: np.ndarray,
    *,
    output_frame_count: int,
    jump_ratio: float,
    smooth_iterations: int,
) -> tuple[np.ndarray, dict[str, object]]:
    repaired, repaired_indices = repair_isolated_spikes(centres, jump_ratio=jump_ratio)
    filtered = binomial_smooth(repaired, iterations=smooth_iterations)
    uniform = resample_uniform_arclength(filtered, output_frame_count)
    # A second pass after arc-length sampling removes polyline-corner
    # acceleration.  This is the pass that prevents a one-frame heading snap;
    # the final resample restores uniform speed and the requested endpoints.
    uniform = binomial_smooth(uniform, iterations=smooth_iterations)
    uniform = resample_uniform_arclength(uniform, output_frame_count)
    return uniform, {
        "jump_ratio": float(jump_ratio),
        "smooth_iterations": int(smooth_iterations),
        "repaired_isolated_source_indices": repaired_indices,
        "raw_steps": step_statistics(np.asarray(centres, dtype=np.float64)),
        "smoothed_steps": step_statistics(uniform),
    }


def robust_world_up(rotations: np.ndarray) -> np.ndarray:
    """Estimate one stable up direction from COLMAP OpenGL camera rotations."""
    candidates = np.asarray(rotations, dtype=np.float64)[:, :3, 1]
    reference = _normalize(candidates[0], label="reference camera up")
    aligned = np.stack(
        [value if float(np.dot(value, reference)) >= 0.0 else -value for value in candidates]
    )
    return _normalize(np.median(aligned, axis=0), label="robust world up")


def forward_facing_rotations(
    centres: np.ndarray,
    *,
    world_up: np.ndarray,
    heading_span: int = 3,
) -> np.ndarray:
    """Construct OpenGL c2w rotations whose -Z axes follow path tangents."""
    if heading_span <= 0:
        raise ValueError("heading_span must be positive")
    centres = np.asarray(centres, dtype=np.float64)
    world_up = _normalize(world_up, label="world up")
    rotations = []
    previous_right: np.ndarray | None = None
    for index in range(len(centres)):
        lo = max(0, index - heading_span)
        hi = min(len(centres) - 1, index + heading_span)
        forward = _normalize(centres[hi] - centres[lo], label=f"path tangent {index}")
        right_candidate = np.cross(forward, world_up)
        if np.linalg.norm(right_candidate) <= 1e-8:
            if previous_right is None:
                fallback = np.asarray([1.0, 0.0, 0.0])
                if abs(float(np.dot(fallback, forward))) > 0.9:
                    fallback = np.asarray([0.0, 0.0, 1.0])
                right_candidate = fallback - np.dot(fallback, forward) * forward
            else:
                right_candidate = previous_right - np.dot(previous_right, forward) * forward
        right = _normalize(right_candidate, label=f"camera right {index}")
        if previous_right is not None and float(np.dot(right, previous_right)) < 0.0:
            right = -right
        camera_up = _normalize(np.cross(right, forward), label=f"camera up {index}")
        rotation = np.stack((right, camera_up, -forward), axis=1)
        if float(np.linalg.det(rotation)) < 0.999999:
            raise AssertionError("forward-facing camera rotation is not right handed")
        rotations.append(rotation)
        previous_right = right
    return np.stack(rotations)


def rotation_step_degrees(rotations: np.ndarray) -> np.ndarray:
    values = []
    for first, second in zip(rotations, rotations[1:]):
        relative = first.T @ second
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        values.append(math.degrees(math.acos(cosine)))
    return np.asarray(values, dtype=np.float64)


def generate(
    source: Mapping[str, Any],
    *,
    output_frame_count: int | None = None,
    jump_ratio: float = 5.0,
    smooth_iterations: int = 8,
    heading_span: int = 3,
    size: int = DEFAULT_SIZE,
    fov_degrees: float = DEFAULT_FOV_DEGREES,
) -> tuple[dict[str, dict[str, object]], dict[str, object], dict[str, object]]:
    poses = np.asarray(normalized_source_poses(source), dtype=np.float64)
    if len(poses) < 2:
        raise ValueError("source trajectory must contain at least two poses")
    output_frame_count = len(poses) if output_frame_count is None else output_frame_count
    centres, smoothing = smooth_forward_centres(
        poses[:, :3, 3],
        output_frame_count=output_frame_count,
        jump_ratio=jump_ratio,
        smooth_iterations=smooth_iterations,
    )
    world_up = robust_world_up(poses[:, :3, :3])
    bases = forward_facing_rotations(centres, world_up=world_up, heading_span=heading_span)
    intrinsics = pinhole_intrinsics(size, fov_degrees)
    view_order = [name for name, _, _, _ in VIEW_SPECS]
    local_rotations = {name: local_rotation(yaw, pitch) for name, yaw, pitch, _ in VIEW_SPECS}
    trajectories: dict[str, dict[str, object]] = {}
    blocks: list[dict[str, object]] = []
    groups: dict[str, list[str]] = {}
    for view_index, (name, yaw, pitch, group) in enumerate(VIEW_SPECS):
        frames = []
        for centre, base_rotation in zip(centres, bases):
            output = np.eye(4, dtype=np.float64)
            output[:3, :3] = base_rotation @ local_rotations[name][:3, :3]
            output[:3, 3] = centre
            frames.append({"transform_matrix": output.tolist()})
        trajectories[name] = {**intrinsics, "frames": frames}
        groups.setdefault(group, []).append(name)
        blocks.append(
            {
                "view": name,
                "yaw_degrees": yaw,
                "pitch_degrees": pitch,
                "group": group,
                "start": view_index * output_frame_count,
                "stop": (view_index + 1) * output_frame_count,
                "count": output_frame_count,
                "trajectory": f"{name}.json",
            }
        )
    combined = {
        **intrinsics,
        "frames": [frame for name in view_order for frame in trajectories[name]["frames"]],
        "forward_facing": True,
        "centre_policy": "robust_smooth_uniform_arclength",
    }
    angular_steps = rotation_step_degrees(bases)
    tangents = np.gradient(centres, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    forward_dot = np.sum((-bases[:, :, 2]) * tangents, axis=1)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "layout": "nerfstudio_equirectangular_planar_14_forward_facing",
        "pose_convention": "OpenGL camera-to-world (+X right, +Y up, -Z forward)",
        "composition": "view_c2w_rotation = smoothed_path_tangent_c2w @ local_rotation",
        "forward_facing": True,
        "centre_policy": "repair_isolated_spikes+binomial_smooth+uniform_arclength",
        "orientation_policy": "H000_negative_z_follows_smoothed_path_tangent",
        "source_frame_count": len(poses),
        "frames_per_view": output_frame_count,
        "total_frames": output_frame_count * len(view_order),
        "view_order": view_order,
        "size": size,
        "fov_degrees": float(fov_degrees),
        "fx": intrinsics["fl_x"],
        "fy": intrinsics["fl_y"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "world_up": world_up.tolist(),
        "heading_span": int(heading_span),
        "smoothing": smoothing,
        "orientation_metrics": {
            "forward_dot_median": float(np.median(forward_dot)),
            "forward_dot_minimum": float(np.min(forward_dot)),
            "rotation_step_median_degrees": float(np.median(angular_steps)),
            "rotation_step_p95_degrees": float(np.percentile(angular_steps, 95)),
            "rotation_step_maximum_degrees": float(np.max(angular_steps)),
        },
        "groups": groups,
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
    **kwargs: object,
) -> dict[str, object]:
    source = json.loads(source_path.expanduser().read_text())
    if not isinstance(source, Mapping):
        raise ValueError("input transforms JSON must be an object")
    trajectories, combined, manifest = generate(source, **kwargs)
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
    parser.add_argument("--output-frame-count", type=int)
    parser.add_argument("--jump-ratio", type=float, default=5.0)
    parser.add_argument("--smooth-iterations", type=int, default=8)
    parser.add_argument("--heading-span", type=int, default=3)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--fov-degrees", type=float, default=DEFAULT_FOV_DEGREES)
    args = parser.parse_args()
    try:
        manifest = write_trajectories(
            args.input,
            args.output_dir,
            output_frame_count=args.output_frame_count,
            jump_ratio=args.jump_ratio,
            smooth_iterations=args.smooth_iterations,
            heading_span=args.heading_span,
            size=args.size,
            fov_degrees=args.fov_degrees,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from error
    print(f"frames_per_view={manifest['frames_per_view']}")
    print(f"total_frames={manifest['total_frames']}")
    print(f"repaired_indices={manifest['smoothing']['repaired_isolated_source_indices']}")
    print(f"smoothed_steps={json.dumps(manifest['smoothing']['smoothed_steps'], sort_keys=True)}")
    print(f"orientation={json.dumps(manifest['orientation_metrics'], sort_keys=True)}")


if __name__ == "__main__":
    main()
