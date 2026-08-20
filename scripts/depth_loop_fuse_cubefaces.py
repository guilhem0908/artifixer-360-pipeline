#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Enforce multi-view/loop colour consistency using rendered ray-depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def intrinsics(path: Path) -> tuple[int, float, float, float, float]:
    raw = json.loads(path.read_text())
    nested = raw.get("intrinsics")
    src = nested if isinstance(nested, dict) else raw
    size = raw.get("size", src.get("size"))
    width = int(size[0] if isinstance(size, list) else size)

    def number(*names: str) -> float:
        for name in names:
            value = src.get(name, raw.get(name))
            if value is not None:
                return float(value)
        raise ValueError(f"missing intrinsic {names}")

    return width, number("fx", "fl_x"), number("fy", "fl_y"), number("cx"), number("cy")


def rays(width: int, fx: float, fy: float, cx: float, cy: float, device: str) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device) + 0.5,
        torch.arange(width, dtype=torch.float32, device=device) + 0.5,
        indexing="ij",
    )
    result = torch.stack(((xx - cx) / fx, (yy - cy) / fy, torch.ones_like(xx)), dim=-1)
    return result / torch.linalg.norm(result, dim=-1, keepdim=True)


def depth_edge_mask(depth: torch.Tensor, ratio: float) -> torch.Tensor:
    gradient = torch.zeros_like(depth)
    gradient[:, 1:-1] += torch.abs(depth[:, 2:] - depth[:, :-2])
    gradient[1:-1, :] += torch.abs(depth[2:, :] - depth[:-2, :])
    return torch.isfinite(depth) & (depth > 0) & (gradient < ratio * depth)


def packed_keys(points: torch.Tensor, voxel: float, span: int) -> torch.Tensor:
    offset = span // 2
    indices = torch.floor(points / voxel).to(torch.int64) + offset
    if torch.any(indices < 0) or torch.any(indices >= span):
        raise RuntimeError("voxel coordinate exceeds --pack-span")
    return (indices[:, 0] * span + indices[:, 1]) * span + indices[:, 2]


def load_view(
    index: int, pred_dir: Path, depth_dir: Path, poses: list[np.ndarray], device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    name = f"{index:05d}"
    image = cv2.imread(str(pred_dir / f"{name}.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(pred_dir / f"{name}.png")
    depth = np.squeeze(np.load(depth_dir / f"{name}.npy")).astype(np.float32)
    return (
        torch.from_numpy(image).to(device=device, dtype=torch.float32),
        torch.from_numpy(depth).to(device=device),
        torch.from_numpy(poses[index]).to(device=device, dtype=torch.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=154)
    parser.add_argument("--view-count", type=int, default=6)
    parser.add_argument("--voxel-size", type=float, default=0.006)
    parser.add_argument("--fuse-stride", type=int, default=4)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--min-frame-span", type=int, default=12)
    parser.add_argument("--central-power", type=float, default=8.0)
    parser.add_argument("--front-weight", type=float, default=2.0)
    parser.add_argument("--edge-depth-ratio", type=float, default=0.03)
    parser.add_argument("--blend", type=float, default=0.85)
    parser.add_argument("--pack-span", type=int, default=1 << 21)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sidecar-mode",
        action="store_true",
        help="Write *_color_loop.png and *_color_loop_mask.png sparse training targets.",
    )
    args = parser.parse_args()

    if args.output_dir.exists() and not args.sidecar_mode:
        raise FileExistsError(f"refusing to replace {args.output_dir}")
    if args.sidecar_mode and args.output_dir.exists():
        existing = list(args.output_dir.glob("*_color_loop.png")) + list(
            args.output_dir.glob("*_color_loop_mask.png")
        )
        if existing:
            raise FileExistsError(f"refusing to replace existing color-loop sidecars in {args.output_dir}")
    if not 0.0 <= args.blend <= 1.0:
        raise ValueError("--blend must be in [0,1]")

    width, fx, fy, cx, cy = intrinsics(args.manifest)
    pose_values = [np.asarray(value, dtype=np.float32) for value in json.loads(args.poses.read_text())]
    view_count = args.view_count * args.frame_count
    if len(pose_values) != view_count:
        raise ValueError(f"expected {view_count} poses, got {len(pose_values)}")

    device = args.device
    ray_grid = rays(width, fx, fy, cx, cy, device)
    stride = max(1, args.fuse_stride)
    sampled_rays = ray_grid[::stride, ::stride].reshape(-1, 3)
    key_chunks: list[torch.Tensor] = []
    colour_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    frame_chunks: list[torch.Tensor] = []

    print(
        f"stage=fuse views={view_count} size={width} stride={stride} "
        f"voxel={args.voxel_size}",
        flush=True,
    )
    for index in range(view_count):
        image, depth, pose = load_view(index, args.pred_dir, args.depth_dir, pose_values, device)
        valid = depth_edge_mask(depth, args.edge_depth_ratio)[::stride, ::stride].reshape(-1)
        sampled_depth = depth[::stride, ::stride].reshape(-1)
        sampled_colour = image[::stride, ::stride].reshape(-1, 3)
        rotation, centre = pose[:3, :3], pose[:3, 3]
        world = centre[None] + sampled_depth[:, None] * (sampled_rays @ rotation.T)
        key = packed_keys(world[valid], args.voxel_size, args.pack_span)
        colour = sampled_colour[valid]
        central = sampled_rays[valid, 2].clamp_min(0).pow(args.central_power)
        view_index = index // args.frame_count
        weight = central * (args.front_weight if view_index == 0 else 1.0)

        # One observation per voxel per rendered view prevents dense regions in
        # a single image from masquerading as multi-view/loop support.
        unique, inverse = torch.unique(key, return_inverse=True)
        sums = torch.zeros((len(unique), 3), dtype=torch.float32, device=device)
        weights = torch.zeros(len(unique), dtype=torch.float32, device=device)
        sums.index_add_(0, inverse, colour * weight[:, None])
        weights.index_add_(0, inverse, weight)
        key_chunks.append(unique)
        colour_chunks.append(sums / weights.clamp_min(1e-8)[:, None])
        weight_chunks.append(weights)
        frame_chunks.append(
            torch.full((len(unique),), index % args.frame_count, dtype=torch.int32, device=device)
        )
        if (index + 1) % 12 == 0:
            print(f"[collect {index + 1:03d}/{view_count}]", flush=True)

    keys = torch.cat(key_chunks)
    colours = torch.cat(colour_chunks)
    weights = torch.cat(weight_chunks)
    frames = torch.cat(frame_chunks)
    del key_chunks, colour_chunks, weight_chunks, frame_chunks

    unique, inverse = torch.unique(keys, return_inverse=True)
    sums = torch.zeros((len(unique), 3), dtype=torch.float32, device=device)
    total_weight = torch.zeros(len(unique), dtype=torch.float32, device=device)
    observations = torch.zeros(len(unique), dtype=torch.int32, device=device)
    minimum_frame = torch.full((len(unique),), args.frame_count, dtype=torch.int32, device=device)
    maximum_frame = torch.full((len(unique),), -1, dtype=torch.int32, device=device)
    sums.index_add_(0, inverse, colours * weights[:, None])
    total_weight.index_add_(0, inverse, weights)
    observations.index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.int32))
    minimum_frame.scatter_reduce_(0, inverse, frames, reduce="amin", include_self=True)
    maximum_frame.scatter_reduce_(0, inverse, frames, reduce="amax", include_self=True)
    support = (
        (observations >= args.min_observations)
        & ((maximum_frame - minimum_frame) >= args.min_frame_span)
    )
    fused_keys = unique[support]
    fused_colours = sums[support] / total_weight[support].clamp_min(1e-8)[:, None]
    order = torch.argsort(fused_keys)
    fused_keys = fused_keys[order].contiguous()
    fused_colours = fused_colours[order].contiguous()
    print(
        f"global_voxels={len(unique)} loop_supported_voxels={len(fused_keys)}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=args.sidecar_mode)
    flat_rays = ray_grid.reshape(-1, 3)
    recoloured_pixels = 0
    all_pixels = 0
    for index in range(view_count):
        image, depth, pose = load_view(index, args.pred_dir, args.depth_dir, pose_values, device)
        rotation, centre = pose[:3, :3], pose[:3, 3]
        flat_depth = depth.reshape(-1)
        world = centre[None] + flat_depth[:, None] * (flat_rays @ rotation.T)
        key = packed_keys(world, args.voxel_size, args.pack_span)
        if len(fused_keys) == 0:
            hit = torch.zeros_like(key, dtype=torch.bool)
            location = torch.zeros_like(key)
        else:
            location = torch.searchsorted(fused_keys, key)
            location = location.clamp(max=len(fused_keys) - 1)
            hit = (
                (fused_keys[location] == key)
                & depth_edge_mask(depth, args.edge_depth_ratio).reshape(-1)
            )
        result = image.reshape(-1, 3).clone()
        result[hit] = (
            (1.0 - args.blend) * result[hit]
            + args.blend * fused_colours[location[hit]]
        )
        output = result.reshape(width, width, 3).clamp(0, 255).to(torch.uint8).cpu().numpy()
        suffix = "_color_loop.png" if args.sidecar_mode else ".png"
        destination = args.output_dir / f"{index:05d}{suffix}"
        if not cv2.imwrite(str(destination), output, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
            raise RuntimeError(f"cannot write {destination}")
        if args.sidecar_mode:
            mask = (
                hit.reshape(width, width).to(torch.uint8).mul(255).cpu().numpy()
            )
            mask_destination = args.output_dir / f"{index:05d}_color_loop_mask.png"
            if not cv2.imwrite(
                str(mask_destination), mask, [cv2.IMWRITE_PNG_COMPRESSION, 2]
            ):
                raise RuntimeError(f"cannot write {mask_destination}")
        recoloured_pixels += int(hit.sum())
        all_pixels += hit.numel()
        if (index + 1) % 12 == 0:
            print(
                f"[recolour {index + 1:03d}/{view_count}] "
                f"coverage={recoloured_pixels / all_pixels:.4f}",
                flush=True,
            )
    print(f"complete coverage={recoloured_pixels / all_pixels:.6f}", flush=True)


if __name__ == "__main__":
    main()
