#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Build scene-agnostic, 3D-consistent RGB targets for ArtiFixer3D.

Every valid pixel is backprojected with the depth rendered by a reference 3DGS.
Observations falling in the same world-space voxel share one canonical colour.
No scene semantics, trajectory direction, or hand-picked frame is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


OPENGL_TO_OPENCV = np.diag((1.0, -1.0, -1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--opacity-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--voxel-pixels", type=float, default=2.5)
    parser.add_argument("--depth-percentile", type=float, default=99.5)
    parser.add_argument("--low-opacity", type=float, default=0.78)
    parser.add_argument("--high-opacity", type=float, default=0.98)
    parser.add_argument("--singleton-weight", type=float, default=0.25)
    parser.add_argument("--support-for-full-weight", type=int, default=4)
    parser.add_argument(
        "--minimum-distinct-frame-support",
        type=int,
        default=1,
        help=(
            "Do not supervise a pixel unless its world-space voxel is seen in "
            "at least this many distinct target frames."
        ),
    )
    return parser.parse_args()


def numbered_files(directory: Path, suffix: str) -> list[Path]:
    files = sorted(directory.glob(f"[0-9][0-9][0-9][0-9][0-9]{suffix}"))
    if not files:
        raise FileNotFoundError(f"no numbered {suffix} files in {directory}")
    expected = [f"{index:05d}{suffix}" for index in range(len(files))]
    names = [path.name for path in files]
    if names != expected:
        raise ValueError(f"non-contiguous files in {directory}: expected 00000..{len(files)-1:05d}")
    return files


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_depth(path: Path) -> np.ndarray:
    depth = np.load(path)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"{path} has unsupported depth shape {depth.shape}")
    return depth.astype(np.float32, copy=False)


def frame_intrinsics(trajectory: dict[str, object], frame: dict[str, object]) -> tuple[float, float, float, float]:
    def value(name: str) -> float:
        if name in frame:
            return float(frame[name])
        return float(trajectory[name])

    return value("fl_x"), value("fl_y"), value("cx"), value("cy")


def camera(frame: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"expected 4x4 camera-to-world transform, got {pose.shape}")
    return pose[:3, :3] @ OPENGL_TO_OPENCV, pose[:3, 3]


def world_points(
    depth: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    intrinsics: tuple[float, float, float, float],
    yy: np.ndarray,
    xx: np.ndarray,
) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    rays = np.stack(((xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)), axis=-1).astype(np.float64)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    rotation, translation = pose
    return translation + (rays * depth[..., None]) @ rotation.T


def conservative_target(
    render: np.ndarray,
    prediction: np.ndarray,
    opacity: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    confidence = opacity.astype(np.float32) / 255.0
    alpha = np.clip((high - confidence) / (high - low), 0.0, 1.0)
    target = render.astype(np.float32) * (1.0 - alpha[..., None]) + prediction.astype(np.float32) * alpha[..., None]
    return np.clip(target, 0, 255).astype(np.uint8)


def pack_voxels(voxels: np.ndarray, minimum: np.ndarray, dimensions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = voxels.astype(np.int64) - minimum
    inside = np.all((shifted >= 0) & (shifted < dimensions), axis=1)
    packed = np.zeros(len(voxels), dtype=np.int64)
    packed[inside] = (shifted[inside, 0] * dimensions[1] + shifted[inside, 1]) * dimensions[2] + shifted[inside, 2]
    return packed, inside


def main() -> None:
    args = parse_args()
    if args.sample_stride < 1 or args.voxel_pixels <= 0:
        raise ValueError("sample-stride and voxel-pixels must be positive")
    if not 0 < args.depth_percentile < 100:
        raise ValueError("depth-percentile must be in (0,100)")
    if not 0 <= args.low_opacity < args.high_opacity <= 1:
        raise ValueError("require 0 <= low-opacity < high-opacity <= 1")
    if not 0 <= args.singleton_weight <= 1:
        raise ValueError("singleton-weight must be in [0,1]")
    if args.support_for_full_weight < 2:
        raise ValueError("support-for-full-weight must be at least 2")
    if args.minimum_distinct_frame_support < 1:
        raise ValueError("minimum-distinct-frame-support must be at least 1")

    prediction_files = numbered_files(args.prediction_dir, ".png")
    render_files = numbered_files(args.render_dir, ".png")
    opacity_files = numbered_files(args.opacity_dir, ".png")
    depth_files = numbered_files(args.depth_dir, ".npy")
    frame_count = len(prediction_files)
    if not (len(render_files) == len(opacity_files) == len(depth_files) == frame_count):
        raise ValueError("prediction, render, opacity, and depth counts must match")

    trajectory = json.loads(args.transforms.read_text())
    frames = trajectory["frames"]
    if len(frames) < frame_count:
        raise ValueError(f"trajectory has {len(frames)} frames but targets contain {frame_count}")
    frames = frames[:frame_count]
    cameras = [camera(frame) for frame in frames]
    intrinsics = [frame_intrinsics(trajectory, frame) for frame in frames]

    probe = read_image(prediction_files[0])
    height, width = probe.shape[:2]
    for path in (render_files[0], opacity_files[0]):
        if read_image(path, cv2.IMREAD_UNCHANGED).shape[:2] != (height, width):
            raise ValueError("all RGB, opacity, and depth inputs must share one resolution")

    # Robust, scene-scaled voxel size: a fixed number of projected pixels at
    # the median observed depth, independent of the scene's metric units.
    depth_probe = []
    probe_stride = max(args.sample_stride * 4, 8)
    for path in depth_files:
        values = read_depth(path)[::probe_stride, ::probe_stride]
        values = values[np.isfinite(values) & (values > 1e-5)]
        if values.size:
            depth_probe.append(values)
    if not depth_probe:
        raise ValueError("all depth maps are empty")
    depth_probe_values = np.concatenate(depth_probe)
    median_depth = float(np.median(depth_probe_values))
    maximum_depth = float(np.percentile(depth_probe_values, args.depth_percentile))
    median_focal = float(np.median([(fx + fy) * 0.5 for fx, fy, _, _ in intrinsics]))
    voxel_size = median_depth / median_focal * args.voxel_pixels
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError(f"invalid automatically derived voxel size {voxel_size}")

    sample_y, sample_x = np.mgrid[0:height:args.sample_stride, 0:width:args.sample_stride]
    voxel_chunks: list[np.ndarray] = []
    colour_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    frame_chunks: list[np.ndarray] = []
    for index in range(frame_count):
        depth = read_depth(depth_files[index])[:: args.sample_stride, :: args.sample_stride]
        opacity_full = read_image(opacity_files[index], cv2.IMREAD_GRAYSCALE)
        opacity = opacity_full[:: args.sample_stride, :: args.sample_stride]
        target = conservative_target(
            read_image(render_files[index]),
            read_image(prediction_files[index]),
            opacity_full,
            args.low_opacity,
            args.high_opacity,
        )[:: args.sample_stride, :: args.sample_stride]
        valid = np.isfinite(depth) & (depth > 1e-5) & (depth <= maximum_depth)
        if not np.any(valid):
            continue
        points = world_points(depth, cameras[index], intrinsics[index], sample_y, sample_x)
        voxels = np.floor(points[valid] / voxel_size).astype(np.int32)
        # Prefer geometrically confident observations and pixels away from the
        # frustum boundary. Frame index is only a deterministic final tie-break.
        fx, fy, cx, cy = intrinsics[index]
        centrality = 1.0 - np.maximum(np.abs((sample_x - cx) / max(cx, width - cx)), np.abs((sample_y - cy) / max(cy, height - cy)))
        score = 0.8 * (opacity[valid].astype(np.float32) / 255.0) + 0.2 * np.clip(centrality[valid], 0, 1).astype(np.float32)
        score -= index * np.finfo(np.float32).eps
        voxel_chunks.append(voxels)
        colour_chunks.append(target[valid])
        score_chunks.append(score)
        frame_chunks.append(np.full(len(voxels), index, dtype=np.int32))
        print(f"[atlas collect {index + 1:04d}/{frame_count:04d}] observations={len(voxels)}", flush=True)

    voxels = np.concatenate(voxel_chunks)
    colours = np.concatenate(colour_chunks)
    scores = np.concatenate(score_chunks)
    observation_frames = np.concatenate(frame_chunks)
    minimum = voxels.min(axis=0).astype(np.int64)
    maximum = voxels.max(axis=0).astype(np.int64)
    dimensions = maximum - minimum + 1
    if int(dimensions[0]) * int(dimensions[1]) * int(dimensions[2]) >= np.iinfo(np.int64).max:
        raise OverflowError(f"voxel grid is too large to pack: {dimensions.tolist()}")
    keys, inside = pack_voxels(voxels, minimum, dimensions)
    if not np.all(inside):
        raise AssertionError("collected voxels fell outside their own bounds")

    # Highest-confidence real observation becomes the sharp canonical colour.
    best_order = np.lexsort((-scores, keys))
    ordered_keys = keys[best_order]
    first = np.r_[True, ordered_keys[1:] != ordered_keys[:-1]]
    best_indices = best_order[first]
    atlas_keys = keys[best_indices]
    atlas_colours = colours[best_indices]

    # Count distinct observing frames per voxel; repeated pixels in one image
    # do not falsely increase loop/multiview support.
    pair_order = np.lexsort((observation_frames, keys))
    pair_keys = keys[pair_order]
    pair_frames = observation_frames[pair_order]
    unique_pair = np.r_[True, (pair_keys[1:] != pair_keys[:-1]) | (pair_frames[1:] != pair_frames[:-1])]
    unique_keys, support = np.unique(pair_keys[unique_pair], return_counts=True)
    if not np.array_equal(unique_keys, atlas_keys):
        raise AssertionError("canonical atlas key/support mismatch")

    output_frames = args.output_dir / "frames"
    output_frames.mkdir(parents=True, exist_ok=True)
    per_frame_metrics = []
    full_y, full_x = np.mgrid[0:height, 0:width]
    for index in range(frame_count):
        render = read_image(render_files[index])
        prediction = read_image(prediction_files[index])
        opacity = read_image(opacity_files[index], cv2.IMREAD_GRAYSCALE)
        candidate = conservative_target(render, prediction, opacity, args.low_opacity, args.high_opacity)
        depth = read_depth(depth_files[index])
        valid_depth = np.isfinite(depth) & (depth > 1e-5) & (depth <= maximum_depth)
        safe_depth = np.where(valid_depth, depth, 0.0)
        points = world_points(safe_depth, cameras[index], intrinsics[index], full_y, full_x)
        query_voxels = np.floor(points.reshape(-1, 3) / voxel_size).astype(np.int32)
        query_keys, inside = pack_voxels(query_voxels, minimum, dimensions)
        locations = np.searchsorted(atlas_keys, query_keys)
        found = inside & (locations < len(atlas_keys))
        safe_locations = np.minimum(locations, len(atlas_keys) - 1)
        found &= atlas_keys[safe_locations] == query_keys
        found &= valid_depth.reshape(-1)
        supported = found & (support[safe_locations] >= 2)
        mask_eligible = found & (
            support[safe_locations] >= args.minimum_distinct_frame_support
        )

        output = candidate.reshape(-1, 3).copy()
        output[supported] = atlas_colours[safe_locations[supported]]
        support_value = np.zeros(height * width, dtype=np.float32)
        support_value[found] = support[safe_locations[found]].astype(np.float32)
        support_confidence = np.clip(
            (support_value - 1.0) / (args.support_for_full_weight - 1.0), 0.0, 1.0
        )
        geometry_confidence = opacity.reshape(-1).astype(np.float32) / 255.0
        weight = np.where(
            mask_eligible,
            args.singleton_weight + (1.0 - args.singleton_weight) * np.maximum(support_confidence, geometry_confidence),
            0.0,
        )
        mask = np.clip(weight.reshape(height, width) * 255.0, 0, 255).astype(np.uint8)
        output = output.reshape(height, width, 3)
        cv2.imwrite(str(output_frames / f"{index:05d}.png"), output)
        cv2.imwrite(str(output_frames / f"{index:05d}_mask.png"), mask)
        changed = np.any(output != candidate, axis=2)
        metric = {
            "frame": index,
            "depth_valid_fraction": float(valid_depth.mean()),
            "atlas_found_fraction": float(found.mean()),
            "multi_observation_fraction": float(supported.mean()),
            "mask_eligible_fraction": float(mask_eligible.mean()),
            "changed_fraction": float(changed.mean()),
            "mean_mask_weight": float(mask.mean() / 255.0),
        }
        per_frame_metrics.append(metric)
        print(
            f"[canonical {index + 1:04d}/{frame_count:04d}] "
            f"found={metric['atlas_found_fraction']:.4f} multi={metric['multi_observation_fraction']:.4f} "
            f"changed={metric['changed_fraction']:.4f} mask={metric['mean_mask_weight']:.4f}",
            flush=True,
        )

    summary = {
        "schema_version": 1,
        "method": "world_space_depth_voxel_canonicalization",
        "semantic_rules": False,
        "frame_count": frame_count,
        "resolution": [width, height],
        "sample_stride": args.sample_stride,
        "voxel_pixels": args.voxel_pixels,
        "minimum_distinct_frame_support": args.minimum_distinct_frame_support,
        "voxel_size_scene_units": voxel_size,
        "median_depth": median_depth,
        "maximum_depth": maximum_depth,
        "atlas_observations": int(len(voxels)),
        "atlas_voxels": int(len(atlas_keys)),
        "atlas_multi_frame_voxels": int(np.count_nonzero(support >= 2)),
        "mean_atlas_found_fraction": float(np.mean([item["atlas_found_fraction"] for item in per_frame_metrics])),
        "mean_multi_observation_fraction": float(np.mean([item["multi_observation_fraction"] for item in per_frame_metrics])),
        "mean_mask_eligible_fraction": float(np.mean([item["mask_eligible_fraction"] for item in per_frame_metrics])),
        "mean_changed_fraction": float(np.mean([item["changed_fraction"] for item in per_frame_metrics])),
        "mean_mask_weight": float(np.mean([item["mean_mask_weight"] for item in per_frame_metrics])),
        "frames": per_frame_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
