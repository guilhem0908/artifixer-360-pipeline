#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Enforce appearance loop closure by reprojecting an outbound pass in 3D.

ArtiFixer can assign different semantics to the same surface when a trajectory
revisits it.  This tool treats the outbound pass as canonical, backprojects its
fixed pixels with the rendered depth, and reprojects them into the return pass.
Only depth-consistent samples are transferred, so this is a world-space
constraint rather than a 2D seam blend.
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
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-per-view", type=int, required=True)
    parser.add_argument("--target-view-index", type=int, default=1)
    parser.add_argument("--turn-frame", type=int, default=77)
    parser.add_argument("--source-view-indices", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--nearest-frames", type=int, default=3)
    parser.add_argument("--max-pose-distance", type=float, default=0.65)
    parser.add_argument("--max-relative-depth-error", type=float, default=0.06)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=15.0)
    return parser.parse_args()


def read_image(directory: Path, index: int, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    path = directory / f"{index:05d}.png"
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_depth(directory: Path, index: int) -> np.ndarray:
    path = directory / f"{index:05d}.npy"
    depth = np.load(path)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"{path} has unsupported shape {depth.shape}")
    return depth.astype(np.float32, copy=False)


def camera_from_frame(frame: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"expected 4x4 transform, got {pose.shape}")
    return pose[:3, :3] @ OPENGL_TO_OPENCV, pose[:3, 3]


def best_source_frames(
    centres: np.ndarray,
    target_frame: int,
    turn_frame: int,
    count: int,
    maximum_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    outbound = np.arange(turn_frame + 1)
    distances = np.linalg.norm(centres[outbound] - centres[target_frame], axis=1)
    order = np.argsort(distances)
    selected = order[:count]
    accepted = distances[selected] <= maximum_distance
    return outbound[selected][accepted], distances[selected][accepted]


def source_rays(height: int, width: int, fx: float, fy: float, cx: float, cy: float, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    rays = np.stack(((xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)), axis=-1).astype(np.float64)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return yy, xx, rays


def update_from_source(
    output: np.ndarray,
    best_score: np.ndarray,
    best_source: np.ndarray,
    target_depth: np.ndarray,
    source_image: np.ndarray,
    source_depth: np.ndarray,
    source_camera: tuple[np.ndarray, np.ndarray],
    target_camera: tuple[np.ndarray, np.ndarray],
    rays: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_tolerance: float,
    pose_distance: float,
    source_index: int,
) -> int:
    height, width = target_depth.shape
    sampled_depth = source_depth[yy, xx]
    valid_source = np.isfinite(sampled_depth) & (sampled_depth > 1e-5)
    if not np.any(valid_source):
        return 0

    ray = rays[valid_source]
    distance = sampled_depth[valid_source, None].astype(np.float64)
    r_source, t_source = source_camera
    r_target, t_target = target_camera
    world = t_source + (ray * distance) @ r_source.T
    target_xyz = (world - t_target) @ r_target
    in_front = target_xyz[:, 2] > 1e-5
    target_distance = np.linalg.norm(target_xyz, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_x = fx * target_xyz[:, 0] / target_xyz[:, 2] + cx
        projected_y = fy * target_xyz[:, 1] / target_xyz[:, 2] + cy

    source_colours = source_image[yy[valid_source], xx[valid_source]]
    # Four-neighbour splatting avoids checkerboard holes when source sampling is strided.
    base_x = np.floor(projected_x).astype(np.int32)
    base_y = np.floor(projected_y).astype(np.int32)
    updates = 0
    for offset_x, offset_y in ((0, 0), (1, 0), (0, 1), (1, 1)):
        px = base_x + offset_x
        py = base_y + offset_y
        inside = in_front & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        if not np.any(inside):
            continue
        indices = np.flatnonzero(inside)
        target_reference = target_depth[py[indices], px[indices]]
        valid_depth = np.isfinite(target_reference) & (target_reference > 1e-5)
        relative_error = np.full(indices.shape, np.inf, dtype=np.float64)
        relative_error[valid_depth] = np.abs(target_distance[indices][valid_depth] - target_reference[valid_depth]) / target_reference[valid_depth]
        accepted = relative_error <= depth_tolerance
        if not np.any(accepted):
            continue
        indices = indices[accepted]
        px = px[indices]
        py = py[indices]
        # Reprojection error is primary.  The small pose term breaks ties in
        # favour of the closest outbound observation.
        score = relative_error[accepted] + pose_distance * 1e-3
        flat = py.astype(np.int64) * width + px
        order = np.lexsort((score, flat))
        sorted_flat = flat[order]
        first = np.r_[True, sorted_flat[1:] != sorted_flat[:-1]]
        chosen = order[first]
        px = px[chosen]
        py = py[chosen]
        score = score[chosen]
        indices = indices[chosen]
        improve = score < best_score[py, px]
        if not np.any(improve):
            continue
        px = px[improve]
        py = py[improve]
        indices = indices[improve]
        score = score[improve]
        output[py, px] = source_colours[indices]
        best_score[py, px] = score.astype(np.float32)
        best_source[py, px] = source_index
        updates += int(improve.sum())
    return updates


def annotate(image: np.ndarray, label: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(result, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def main() -> None:
    args = parse_args()
    if args.sample_stride < 1 or args.nearest_frames < 1:
        raise ValueError("sample-stride and nearest-frames must be positive")
    trajectory = json.loads(args.transforms.read_text())
    frames = trajectory["frames"]
    required = args.frames_per_view * (args.target_view_index + 1)
    if len(frames) < required:
        raise ValueError(f"need at least {required} transforms, found {len(frames)}")
    width = int(trajectory.get("w", 0))
    height = int(trajectory.get("h", 0))
    fx = float(trajectory["fl_x"])
    fy = float(trajectory["fl_y"])
    cx = float(trajectory["cx"])
    cy = float(trajectory["cy"])
    if width <= 0 or height <= 0:
        probe = read_image(args.input_dir, 0)
        height, width = probe.shape[:2]

    cameras = [camera_from_frame(frame) for frame in frames[:required]]
    base_centres = np.stack([cameras[index][1] for index in range(args.frames_per_view)])
    yy, xx, rays = source_rays(height, width, fx, fy, cx, cy, args.sample_stride)

    corrected_dir = args.output_dir / "loop_consistent"
    mask_dir = args.output_dir / "loop_masks"
    diagnostic_dir = args.output_dir / "diagnostics"
    for directory in (corrected_dir, mask_dir, diagnostic_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Preserve every view/frame; only return frames of the requested view are constrained.
    for index in range(required):
        image = read_image(args.input_dir, index)
        cv2.imwrite(str(corrected_dir / f"{index:05d}.png"), image)

    metrics: list[dict[str, object]] = []
    diagnostic_frames = {args.turn_frame + 1, 115, 138, 146, 149, 150, args.frames_per_view - 1}
    target_offset = args.target_view_index * args.frames_per_view
    for target_frame in range(args.turn_frame + 1, args.frames_per_view):
        source_frames, pose_distances = best_source_frames(
            base_centres,
            target_frame,
            args.turn_frame,
            args.nearest_frames,
            args.max_pose_distance,
        )
        target_index = target_offset + target_frame
        original = read_image(args.input_dir, target_index)
        output = original.copy()
        target_depth = read_depth(args.depth_dir, target_index)
        best_score = np.full((height, width), np.inf, dtype=np.float32)
        best_source = np.full((height, width), -1, dtype=np.int16)
        attempts = 0
        source_records = []
        for source_frame, pose_distance in zip(source_frames.tolist(), pose_distances.tolist()):
            for source_view_index in args.source_view_indices:
                source_index = source_view_index * args.frames_per_view + source_frame
                attempts += update_from_source(
                    output,
                    best_score,
                    best_source,
                    target_depth,
                    read_image(args.input_dir, source_index),
                    read_depth(args.depth_dir, source_index),
                    cameras[source_index],
                    cameras[target_index],
                    rays,
                    yy,
                    xx,
                    fx,
                    fy,
                    cx,
                    cy,
                    args.max_relative_depth_error,
                    pose_distance,
                    source_index,
                )
                source_records.append({"frame": source_frame, "view_index": source_view_index, "index": source_index, "pose_distance": pose_distance})
        mask = best_source >= 0
        mask_u8 = mask.astype(np.uint8) * 255
        cv2.imwrite(str(corrected_dir / f"{target_index:05d}.png"), output)
        cv2.imwrite(str(mask_dir / f"{target_index:05d}.png"), mask_u8)
        coverage = float(mask.mean())
        mean_change = float(np.abs(output.astype(np.float32) - original.astype(np.float32))[mask].mean()) if np.any(mask) else 0.0
        record = {
            "target_frame": target_frame,
            "target_index": target_index,
            "source_records": source_records,
            "coverage": coverage,
            "mean_absolute_change": mean_change,
            "update_attempts": attempts,
        }
        metrics.append(record)
        if target_frame in diagnostic_frames:
            overlay = original.copy()
            overlay[mask] = (0, 0, 255)
            overlay = cv2.addWeighted(original, 0.55, overlay, 0.45, 0)
            comparison = np.hstack(
                (
                    annotate(original, f"Before return frame {target_frame}"),
                    annotate(output, f"3D loop closure; coverage {coverage:.1%}"),
                    annotate(overlay, "Red = canonical outbound reprojection"),
                )
            )
            cv2.imwrite(str(diagnostic_dir / f"frame_{target_frame:05d}.png"), comparison)
        print(
            f"[loop {target_frame:03d}/{args.frames_per_view - 1:03d}] "
            f"sources={[(int(frame), round(float(distance), 4)) for frame, distance in zip(source_frames, pose_distances)]} "
            f"coverage={coverage:.4f} change={mean_change:.3f}",
            flush=True,
        )

    summary = {
        "frames_per_view": args.frames_per_view,
        "target_view_index": args.target_view_index,
        "turn_frame": args.turn_frame,
        "source_view_indices": args.source_view_indices,
        "nearest_frames": args.nearest_frames,
        "max_pose_distance": args.max_pose_distance,
        "max_relative_depth_error": args.max_relative_depth_error,
        "sample_stride": args.sample_stride,
        "corrected_return_frames": len(metrics),
        "mean_coverage": float(np.mean([item["coverage"] for item in metrics])) if metrics else 0.0,
        "frames": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    video = args.output_dir / "H060_loop_consistent_768x768_15fps.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {video}")
    for frame in range(args.frames_per_view):
        writer.write(read_image(corrected_dir, target_offset + frame))
    writer.release()
    print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=2))
    print(f"VIDEO={video}")


if __name__ == "__main__":
    main()
