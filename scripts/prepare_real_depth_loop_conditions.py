#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Anchor novel frustums to repeated real observations through a 3D depth atlas.

The atlas is built only from original source RGB frames and the reference 3DGS
depth maps.  A real colour is transferred to a target pixel only when the same
world-space voxel was observed by multiple source frames.  This is a generic
loop-closure constraint: it contains no scene labels, turn frame, or hand-picked
surface rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Make `python scripts/<name>.py` work on a plain clone: the repository root must
# be importable for the `scripts.*` / `model_eval.*` imports below.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_depth_canonical_targets import (  # noqa: E402
    camera,
    frame_intrinsics,
    pack_voxels,
    read_depth,
    read_image,
    world_points,
)
from scripts.prepare_real_anchor_conditions import source_path


def numbered_count(directory: Path, suffix: str) -> int:
    return len(list(directory.glob(f"[0-9][0-9][0-9][0-9][0-9]{suffix}")))


def build_real_atlas(
    transforms: dict,
    *,
    image_root: Path,
    source_depth_dir: Path,
    source_opacity_dir: Path,
    anchor_start: int,
    anchor_count: int,
    sample_stride: int,
    voxel_pixels: float,
    minimum_source_opacity: float,
) -> dict[str, np.ndarray | float | int]:
    frames = transforms["frames"]
    source_frames = frames[anchor_start : anchor_start + anchor_count]
    if len(source_frames) != anchor_count:
        raise ValueError("source anchor range exceeds transforms")
    if numbered_count(source_depth_dir, ".npy") != anchor_count:
        raise ValueError("source depth count does not match anchor_count")
    if numbered_count(source_opacity_dir, ".png") != anchor_count:
        raise ValueError("source opacity count does not match anchor_count")

    depth_probe = []
    for index in range(anchor_count):
        values = read_depth(source_depth_dir / f"{index:05d}.npy")[::16, ::16]
        values = values[np.isfinite(values) & (values > 1e-5)]
        if values.size:
            depth_probe.append(values)
    if not depth_probe:
        raise ValueError("source depth atlas is empty")
    median_depth = float(np.median(np.concatenate(depth_probe)))
    median_focal = float(
        np.median([(fx + fy) * 0.5 for fx, fy, _, _ in (frame_intrinsics(transforms, frame) for frame in source_frames)])
    )
    voxel_size = median_depth / median_focal * voxel_pixels
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError(f"invalid voxel size {voxel_size}")

    voxel_chunks, colour_chunks, score_chunks, frame_chunks = [], [], [], []
    for source_index, frame in enumerate(source_frames):
        depth = read_depth(source_depth_dir / f"{source_index:05d}.npy")
        opacity = read_image(source_opacity_dir / f"{source_index:05d}.png", cv2.IMREAD_GRAYSCALE)
        colour = read_image(source_path(image_root, str(frame["file_path"])))
        if colour.shape[:2] != depth.shape or opacity.shape != depth.shape:
            raise ValueError(f"source {source_index} RGB/depth/opacity shape mismatch")
        height, width = depth.shape
        yy, xx = np.mgrid[0:height:sample_stride, 0:width:sample_stride]
        sampled_depth = depth[::sample_stride, ::sample_stride]
        sampled_opacity = opacity[::sample_stride, ::sample_stride]
        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth > 1e-5)
            & (sampled_opacity >= round(255 * minimum_source_opacity))
        )
        if not np.any(valid):
            continue
        points = world_points(
            sampled_depth,
            camera(frame),
            frame_intrinsics(transforms, frame),
            yy,
            xx,
        )
        voxels = np.floor(points[valid] / voxel_size).astype(np.int32)
        fx, fy, cx, cy = frame_intrinsics(transforms, frame)
        centrality = 1.0 - np.maximum(
            np.abs((xx - cx) / max(cx, width - cx)),
            np.abs((yy - cy) / max(cy, height - cy)),
        )
        scores = sampled_opacity[valid].astype(np.float32) / 255.0
        scores += 0.05 * np.clip(centrality[valid], 0, 1).astype(np.float32)
        scores -= source_index * np.finfo(np.float32).eps
        voxel_chunks.append(voxels)
        colour_chunks.append(colour[::sample_stride, ::sample_stride][valid])
        score_chunks.append(scores)
        frame_chunks.append(np.full(len(voxels), source_index, dtype=np.int32))
        print(f"[real atlas {source_index + 1:03d}/{anchor_count:03d}] samples={len(voxels)}", flush=True)
    if not voxel_chunks:
        raise ValueError("no valid real atlas observations")

    voxels = np.concatenate(voxel_chunks)
    colours = np.concatenate(colour_chunks)
    scores = np.concatenate(score_chunks)
    observation_frames = np.concatenate(frame_chunks)
    minimum = voxels.min(axis=0).astype(np.int64)
    maximum = voxels.max(axis=0).astype(np.int64)
    dimensions = maximum - minimum + 1
    keys, inside = pack_voxels(voxels, minimum, dimensions)
    if not np.all(inside):
        raise AssertionError("atlas voxels fell outside their bounds")

    best_order = np.lexsort((-scores, keys))
    ordered_keys = keys[best_order]
    first = np.r_[True, ordered_keys[1:] != ordered_keys[:-1]]
    best = best_order[first]
    atlas_keys = keys[best]
    atlas_colours = colours[best]

    pair_order = np.lexsort((observation_frames, keys))
    pair_keys = keys[pair_order]
    pair_frames = observation_frames[pair_order]
    unique_pair = np.r_[True, (pair_keys[1:] != pair_keys[:-1]) | (pair_frames[1:] != pair_frames[:-1])]
    support_keys, support = np.unique(pair_keys[unique_pair], return_counts=True)
    if not np.array_equal(support_keys, atlas_keys):
        raise AssertionError("atlas support/key mismatch")
    return {
        "keys": atlas_keys,
        "colours": atlas_colours,
        "support": support,
        "minimum": minimum,
        "dimensions": dimensions,
        "voxel_size": voxel_size,
        "observation_count": len(voxels),
    }


def apply_atlas(
    transforms: dict,
    atlas: dict,
    *,
    target_count: int,
    target_depth_dir: Path,
    input_render_dir: Path,
    input_opacity_dir: Path,
    output_render_dir: Path,
    output_opacity_dir: Path,
    minimum_support: int,
    feather_pixels: float,
) -> list[dict[str, float | int]]:
    output_render_dir.mkdir(parents=True, exist_ok=True)
    output_opacity_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    keys = atlas["keys"]
    colours = atlas["colours"]
    support = atlas["support"]
    for index, frame in enumerate(transforms["frames"][:target_count]):
        depth = read_depth(target_depth_dir / f"{index:05d}.npy")
        render = read_image(input_render_dir / f"{index:05d}.png")
        opacity = read_image(input_opacity_dir / f"{index:05d}.png", cv2.IMREAD_GRAYSCALE)
        if render.shape[:2] != depth.shape or opacity.shape != depth.shape:
            raise ValueError(f"target {index} RGB/depth/opacity shape mismatch")
        height, width = depth.shape
        yy, xx = np.mgrid[0:height, 0:width]
        valid_depth = np.isfinite(depth) & (depth > 1e-5)
        points = world_points(
            np.where(valid_depth, depth, 0),
            camera(frame),
            frame_intrinsics(transforms, frame),
            yy,
            xx,
        )
        voxels = np.floor(points.reshape(-1, 3) / float(atlas["voxel_size"])).astype(np.int32)
        query_keys, inside = pack_voxels(voxels, atlas["minimum"], atlas["dimensions"])
        locations = np.searchsorted(keys, query_keys)
        safe = np.minimum(locations, len(keys) - 1)
        found = inside & (locations < len(keys)) & (keys[safe] == query_keys) & valid_depth.reshape(-1)
        anchored = found & (support[safe] >= minimum_support)
        mask = anchored.reshape(height, width)
        if feather_pixels > 0 and np.any(mask):
            alpha = np.clip(cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3) / feather_pixels, 0, 1)
        else:
            alpha = mask.astype(np.float32)
        canonical = render.reshape(-1, 3).copy()
        canonical[anchored] = colours[safe[anchored]]
        canonical = canonical.reshape(height, width, 3)
        output = render.astype(np.float32) * (1 - alpha[..., None]) + canonical.astype(np.float32) * alpha[..., None]
        output_opacity = np.maximum(opacity.astype(np.float32), alpha * 255)
        cv2.imwrite(str(output_render_dir / f"{index:05d}.png"), np.clip(output, 0, 255).astype(np.uint8))
        cv2.imwrite(str(output_opacity_dir / f"{index:05d}.png"), np.clip(output_opacity, 0, 255).astype(np.uint8))
        record = {
            "frame": index,
            "real_loop_coverage": float(mask.mean()),
            "mean_alpha": float(alpha.mean()),
        }
        metrics.append(record)
        print(
            f"[real loop {index + 1:04d}/{target_count:04d}] "
            f"coverage={record['real_loop_coverage']:.4f} alpha={record['mean_alpha']:.4f}",
            flush=True,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--source-depth-dir", type=Path, required=True)
    parser.add_argument("--source-opacity-dir", type=Path, required=True)
    parser.add_argument("--target-depth-dir", type=Path, required=True)
    parser.add_argument("--input-render-dir", type=Path, required=True)
    parser.add_argument("--input-opacity-dir", type=Path, required=True)
    parser.add_argument("--output-render-dir", type=Path, required=True)
    parser.add_argument("--output-opacity-dir", type=Path, required=True)
    parser.add_argument("--anchor-start", type=int, required=True)
    parser.add_argument("--anchor-count", type=int, required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--voxel-pixels", type=float, default=2.0)
    parser.add_argument("--minimum-source-opacity", type=float, default=0.9)
    parser.add_argument("--minimum-support", type=int, default=2)
    parser.add_argument("--feather-pixels", type=float, default=2.0)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    if args.sample_stride < 1 or args.voxel_pixels <= 0 or args.minimum_support < 2:
        raise ValueError("invalid atlas sampling/support arguments")
    if not 0 <= args.minimum_source_opacity <= 1 or args.feather_pixels < 0:
        raise ValueError("invalid opacity/feather arguments")
    transforms = json.loads(args.transforms.read_text())
    atlas = build_real_atlas(
        transforms,
        image_root=args.image_root,
        source_depth_dir=args.source_depth_dir,
        source_opacity_dir=args.source_opacity_dir,
        anchor_start=args.anchor_start,
        anchor_count=args.anchor_count,
        sample_stride=args.sample_stride,
        voxel_pixels=args.voxel_pixels,
        minimum_source_opacity=args.minimum_source_opacity,
    )
    frames = apply_atlas(
        transforms,
        atlas,
        target_count=args.target_count,
        target_depth_dir=args.target_depth_dir,
        input_render_dir=args.input_render_dir,
        input_opacity_dir=args.input_opacity_dir,
        output_render_dir=args.output_render_dir,
        output_opacity_dir=args.output_opacity_dir,
        minimum_support=args.minimum_support,
        feather_pixels=args.feather_pixels,
    )
    summary = {
        "schema_version": 1,
        "method": "multi_real_frame_world_depth_atlas",
        "semantic_rules": False,
        "target_count": args.target_count,
        "anchor_count": args.anchor_count,
        "minimum_support": args.minimum_support,
        "voxel_size_scene_units": float(atlas["voxel_size"]),
        "atlas_observations": int(atlas["observation_count"]),
        "atlas_voxels": int(len(atlas["keys"])),
        "mean_real_loop_coverage": float(np.mean([item["real_loop_coverage"] for item in frames])),
        "mean_alpha": float(np.mean([item["mean_alpha"] for item in frames])),
        "frames": frames,
    }
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
