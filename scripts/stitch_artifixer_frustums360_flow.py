#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Stitch ArtiFixer frustums with geometry alignment and hard ownership.

Unlike feather or multiband blending, this aligns one side of each ownership
boundary by dense optical flow and then performs a hard cut.  It therefore does
not average two incompatible reconstructions at the seam.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from scripts.stitch_artifixer_frustums360 import FrustumProjector, Manifest, read_images
except ModuleNotFoundError:
    from stitch_artifixer_frustums360 import FrustumProjector, Manifest, read_images


def boundary_pairs(labels: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    pairs: dict[tuple[int, int], np.ndarray] = {}
    neighbors = ((labels, np.roll(labels, -1, axis=1)), (labels[:-1], labels[1:]))
    for first, second in neighbors:
        changed = first != second
        ys, xs = np.nonzero(changed)
        for y, x in zip(ys.tolist(), xs.tolist()):
            a, b = sorted((int(first[y, x]), int(second[y, x])))
            key = (a, b)
            if key not in pairs:
                pairs[key] = np.zeros_like(labels, dtype=np.uint8)
            pairs[key][y, x] = 255
            if y + 1 < labels.shape[0]:
                pairs[key][y + 1, x] = 255
    return pairs


def dense_pair_flow(
    reference: np.ndarray,
    moving: np.ndarray,
    valid: np.ndarray,
    *,
    downscale: int,
    max_flow: float,
) -> np.ndarray:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    mov_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    common = np.where(valid, ((ref_gray.astype(np.float32) + mov_gray) * 0.5), 127).astype(np.uint8)
    ref_gray = np.where(valid, ref_gray, common).astype(np.uint8)
    mov_gray = np.where(valid, mov_gray, common).astype(np.uint8)
    small_size = (max(32, reference.shape[1] // downscale), max(16, reference.shape[0] // downscale))
    ref_small = cv2.resize(ref_gray, small_size, interpolation=cv2.INTER_AREA)
    mov_small = cv2.resize(mov_gray, small_size, interpolation=cv2.INTER_AREA)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    flow = dis.calc(ref_small, mov_small, None)
    flow = cv2.resize(flow, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= reference.shape[1] / small_size[0]
    flow[..., 1] *= reference.shape[0] / small_size[1]
    flow = cv2.GaussianBlur(flow, (0, 0), 2.0)
    magnitude = np.linalg.norm(flow, axis=-1)
    scale = np.minimum(1.0, max_flow / np.maximum(magnitude, 1e-6))
    return flow * scale[..., None]


def remap_with_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    y, x = np.mgrid[: image.shape[0], : image.shape[1]].astype(np.float32)
    return cv2.remap(
        image,
        x + flow[..., 0].astype(np.float32),
        y + flow[..., 1].astype(np.float32),
        cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_WRAP,
    )


def align_views(
    projector: FrustumProjector,
    images: dict[str, np.ndarray],
    *,
    flow_band: float,
    downscale: int,
    max_flow: float,
    minimum_improvement: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    order = projector.manifest.view_order
    warped = {view: projector.warp(images[view], view) for view in order}
    labels = projector.owner_labels
    pairs = boundary_pairs(labels)
    flow_sum = {view: np.zeros((*labels.shape, 2), np.float32) for view in order}
    weight_sum = {view: np.zeros(labels.shape, np.float32) for view in order}
    metrics: list[dict[str, object]] = []

    for (parent_index, child_index), boundary in sorted(pairs.items()):
        parent, child = order[parent_index], order[child_index]
        valid = projector.maps[parent].valid & projector.maps[child].valid
        distance = cv2.distanceTransform(np.where(boundary == 0, 255, 0).astype(np.uint8), cv2.DIST_L2, 3)
        influence = np.clip(1.0 - distance / flow_band, 0.0, 1.0)
        influence *= (labels == child_index) & valid
        evaluation = (distance <= max(4.0, flow_band * 0.15)) & valid
        if np.count_nonzero(evaluation) < 64 or np.count_nonzero(influence) < 64:
            continue

        flow = dense_pair_flow(
            warped[parent], warped[child], valid, downscale=downscale, max_flow=max_flow
        )
        aligned_child = remap_with_flow(warped[child], flow)
        before = float(
            np.abs(warped[parent].astype(np.float32) - warped[child].astype(np.float32))[evaluation].mean()
        )
        after = float(
            np.abs(warped[parent].astype(np.float32) - aligned_child.astype(np.float32))[evaluation].mean()
        )
        accepted = after < before * (1.0 - minimum_improvement)
        metrics.append(
            {"parent": parent, "child": child, "before_mae": before, "after_mae": after, "accepted": accepted}
        )
        print(
            f"pair={parent}->{child} before={before:.3f} after={after:.3f} accepted={accepted}",
            flush=True,
        )
        if not accepted:
            continue
        flow_sum[child] += flow * influence[..., None]
        weight_sum[child] += influence

    corrected: dict[str, np.ndarray] = {}
    for view in order:
        weight = weight_sum[view]
        displacement = flow_sum[view] / np.maximum(weight[..., None], 1.0)
        corrected[view] = remap_with_flow(warped[view], displacement)
    return corrected, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--flow-band", type=float, default=96.0)
    parser.add_argument("--flow-downscale", type=int, default=4)
    parser.add_argument("--max-flow", type=float, default=48.0)
    parser.add_argument("--minimum-improvement", type=float, default=0.03)
    args = parser.parse_args()

    manifest = Manifest.load(args.manifest)
    projector = FrustumProjector(manifest, args.width, args.height, ownership_mode="angular")
    images = read_images(args.input_dir, manifest, args.frame)
    corrected, metrics = align_views(
        projector,
        images,
        flow_band=args.flow_band,
        downscale=args.flow_downscale,
        max_flow=args.max_flow,
        minimum_improvement=args.minimum_improvement,
    )
    panorama = np.zeros((args.height, args.width, 3), np.uint8)
    for index, view in enumerate(manifest.view_order):
        panorama[projector.owner_labels == index] = corrected[view][projector.owner_labels == index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), panorama, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
        raise RuntimeError(f"cannot write {args.output}")
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps({"frame": args.frame, "pairs": metrics}, indent=2) + "\n")
    print(f"OUTPUT={args.output}", flush=True)


if __name__ == "__main__":
    main()
