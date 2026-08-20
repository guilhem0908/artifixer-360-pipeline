#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Inject same-centre real observations into world-locked target renders.

Each target frustum shares its centre with one source video frame.  Their
overlap is therefore a pure rotation and can be transferred exactly without a
depth estimate.  These real pixels are the highest-confidence conditioning
available to ArtiFixer and prevent a novel direction from redefining surfaces
already observed by the input video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def intrinsics(transforms: dict, frame: dict) -> tuple[int, int, float, float, float, float]:
    def value(name: str):
        return frame[name] if name in frame else transforms[name]

    return (
        int(value("w")),
        int(value("h")),
        float(value("fl_x")),
        float(value("fl_y")),
        float(value("cx")),
        float(value("cy")),
    )


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def source_path(image_root: Path, file_path: str) -> Path:
    path = Path(file_path)
    return path if path.is_absolute() else image_root / path


def rotation_reprojection(
    target_frame: dict,
    source_frame: dict,
    target_k: tuple[int, int, float, float, float, float],
    source_k: tuple[int, int, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_w, target_h, target_fx, target_fy, target_cx, target_cy = target_k
    source_w, source_h, source_fx, source_fy, source_cx, source_cy = source_k
    target_pose = np.asarray(target_frame["transform_matrix"], dtype=np.float64)
    source_pose = np.asarray(source_frame["transform_matrix"], dtype=np.float64)
    if target_pose.shape != (4, 4) or source_pose.shape != (4, 4):
        raise ValueError("camera transforms must be 4x4")
    centre_error = float(np.linalg.norm(target_pose[:3, 3] - source_pose[:3, 3]))
    if centre_error > 1e-6:
        raise ValueError(f"target/source centres differ by {centre_error:.9g}")

    yy, xx = np.mgrid[0:target_h, 0:target_w].astype(np.float64)
    target_rays = np.stack(
        (
            (xx + 0.5 - target_cx) / target_fx,
            -(yy + 0.5 - target_cy) / target_fy,
            -np.ones_like(xx),
        ),
        axis=-1,
    )
    relative = source_pose[:3, :3].T @ target_pose[:3, :3]
    source_rays = target_rays @ relative.T
    forward = -source_rays[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        map_x = source_fx * source_rays[..., 0] / forward + source_cx - 0.5
        map_y = -source_fy * source_rays[..., 1] / forward + source_cy - 0.5
    valid = (
        (forward > 0)
        & (map_x >= 0)
        & (map_x <= source_w - 1)
        & (map_y >= 0)
        & (map_y <= source_h - 1)
    )
    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def prepare(
    *,
    transforms_path: Path,
    rig_manifest_path: Path,
    image_root: Path,
    render_dir: Path,
    opacity_dir: Path,
    output_render_dir: Path,
    output_opacity_dir: Path,
    output_real_dir: Path | None,
    output_real_mask_dir: Path | None,
    anchor_start: int,
    feather_pixels: float,
    frame_start: int = 0,
    frame_stop: int | None = None,
) -> dict:
    transforms = json.loads(transforms_path.read_text())
    rig = json.loads(rig_manifest_path.read_text())
    frames = transforms["frames"]
    frame_count = int(rig["frames_per_view"])
    target_count = frame_count * len(rig["view_order"])
    frame_stop = frame_count if frame_stop is None else frame_stop
    if not 0 <= frame_start < frame_stop <= frame_count:
        raise ValueError("invalid sparse target frame interval")
    if anchor_start < target_count or anchor_start + frame_count > len(frames):
        raise ValueError("anchor range does not follow the full target trajectory")
    output_render_dir.mkdir(parents=True, exist_ok=True)
    output_opacity_dir.mkdir(parents=True, exist_ok=True)
    if (output_real_dir is None) != (output_real_mask_dir is None):
        raise ValueError("output_real_dir and output_real_mask_dir must be provided together")
    if output_real_dir is not None:
        output_real_dir.mkdir(parents=True, exist_ok=True)
        assert output_real_mask_dir is not None
        output_real_mask_dir.mkdir(parents=True, exist_ok=True)

    coverages = []
    selected_target_indices = [
        view_index * frame_count + timestamp
        for view_index in range(len(rig["view_order"]))
        for timestamp in range(frame_start, frame_stop)
    ]
    for target_index in selected_target_indices:
        timestamp = target_index % frame_count
        source_index = anchor_start + timestamp
        target_frame = frames[target_index]
        source_frame = frames[source_index]
        target_k = intrinsics(transforms, target_frame)
        source_k = intrinsics(transforms, source_frame)
        render = read_image(render_dir / f"{target_index:05d}.png")
        opacity = read_image(opacity_dir / f"{target_index:05d}.png", cv2.IMREAD_GRAYSCALE)
        if render.shape[:2] != (target_k[1], target_k[0]) or opacity.shape != render.shape[:2]:
            raise ValueError(f"target {target_index} render/intrinsics shape mismatch")
        real = read_image(source_path(image_root, str(source_frame["file_path"])))
        if real.shape[:2] != (source_k[1], source_k[0]):
            raise ValueError(f"source {source_index} image/intrinsics shape mismatch")

        map_x, map_y, valid = rotation_reprojection(target_frame, source_frame, target_k, source_k)
        warped = cv2.remap(
            real,
            map_x,
            map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
        )
        if feather_pixels > 0:
            distance = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 3)
            alpha = np.clip(distance / feather_pixels, 0.0, 1.0)
        else:
            alpha = valid.astype(np.float32)
        conditioned = (
            render.astype(np.float32) * (1.0 - alpha[..., None])
            + warped.astype(np.float32) * alpha[..., None]
        )
        conditioned_opacity = np.maximum(opacity.astype(np.float32), 255.0 * alpha)
        cv2.imwrite(str(output_render_dir / f"{target_index:05d}.png"), np.clip(conditioned, 0, 255).astype(np.uint8))
        cv2.imwrite(
            str(output_opacity_dir / f"{target_index:05d}.png"),
            np.clip(conditioned_opacity, 0, 255).astype(np.uint8),
        )
        if output_real_dir is not None:
            assert output_real_mask_dir is not None
            # Keep an unblended, auditable LOCKED layer separate from the
            # conditioning image. The hard valid mask is the authority used
            # by proposal verification and exact final reinsertion.
            real_layer = np.where(valid[..., None], warped, 0).astype(np.uint8)
            real_mask = valid.astype(np.uint8) * 255
            if not cv2.imwrite(str(output_real_dir / f"{target_index:05d}.png"), real_layer):
                raise RuntimeError(f"cannot write real anchor {target_index:05d}")
            if not cv2.imwrite(
                str(output_real_mask_dir / f"{target_index:05d}.png"), real_mask
            ):
                raise RuntimeError(f"cannot write real mask {target_index:05d}")
        coverages.append(float(valid.mean()))
        print(
            f"[anchor {target_index + 1:04d}/{target_count:04d}] coverage={coverages[-1]:.4f}",
            flush=True,
        )

    summary = {
        "schema_version": 1,
        "method": "same_centre_real_rotation_reprojection",
        "semantic_rules": False,
        "target_count": len(selected_target_indices),
        "total_rig_target_count": target_count,
        "target_frame_range": [frame_start, frame_stop],
        "target_layout": (
            "complete_view_major"
            if len(selected_target_indices) == target_count
            else "sparse_requested_view_major"
        ),
        "anchor_count": frame_count,
        "mean_coverage": float(np.mean(coverages)),
        "min_coverage": float(np.min(coverages)),
        "max_coverage": float(np.max(coverages)),
        "feather_pixels": feather_pixels,
        "locked_layers_written": output_real_dir is not None,
        "locked_pixel_policy": "hard same-centre rotation reprojection; no feather",
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--rig-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--opacity-dir", type=Path, required=True)
    parser.add_argument("--output-render-dir", type=Path, required=True)
    parser.add_argument("--output-opacity-dir", type=Path, required=True)
    parser.add_argument("--output-real-dir", type=Path)
    parser.add_argument("--output-real-mask-dir", type=Path)
    parser.add_argument("--anchor-start", type=int, required=True)
    parser.add_argument("--feather-pixels", type=float, default=8.0)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    if args.feather_pixels < 0:
        raise ValueError("--feather-pixels must be non-negative")
    summary = prepare(
        transforms_path=args.transforms,
        rig_manifest_path=args.rig_manifest,
        image_root=args.image_root,
        render_dir=args.render_dir,
        opacity_dir=args.opacity_dir,
        output_render_dir=args.output_render_dir,
        output_opacity_dir=args.output_opacity_dir,
        output_real_dir=args.output_real_dir,
        output_real_mask_dir=args.output_real_mask_dir,
        anchor_start=args.anchor_start,
        feather_pixels=args.feather_pixels,
        frame_start=args.frame_start,
        frame_stop=args.frame_stop,
    )
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
