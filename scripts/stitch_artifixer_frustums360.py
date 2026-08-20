#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Stitch overlapping perspective frustums into a full ERP panorama video."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np


OPENGL_CV_CHANGE = np.diag((1.0, -1.0, -1.0))


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


@dataclass(frozen=True)
class Manifest:
    view_order: tuple[str, ...]
    frames_per_view: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotations: dict[str, np.ndarray]

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        raw = json.loads(path.read_text())
        order = tuple(str(value) for value in raw["view_order"])
        if len(order) < 2 or len(set(order)) != len(order):
            raise ValueError("view_order must contain unique view identifiers")
        rotations: dict[str, np.ndarray] = {}
        for view in order:
            matrix = np.asarray(raw["local_rotations"][view], dtype=np.float64)
            if matrix.shape == (4, 4):
                matrix = matrix[:3, :3]
            if matrix.shape != (3, 3):
                raise ValueError(f"rotation {view} must be 3x3 or 4x4")
            matrix = OPENGL_CV_CHANGE @ matrix @ OPENGL_CV_CHANGE
            if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
                raise ValueError(f"rotation {view} is not orthonormal")
            rotations[view] = matrix
        size = raw["size"]
        width, height = ((int(size[0]), int(size[1])) if isinstance(size, list) else (int(size), int(size)))
        return cls(
            order,
            int(raw["frames_per_view"]),
            width,
            height,
            float(raw["fx"]),
            float(raw["fy"]),
            float(raw["cx"]),
            float(raw["cy"]),
            rotations,
        )


@dataclass(frozen=True)
class Projection:
    map_x: np.ndarray
    map_y: np.ndarray
    valid: np.ndarray
    weight: np.ndarray
    axis_score: np.ndarray


class FrustumProjector:
    def __init__(
        self,
        manifest: Manifest,
        output_width: int,
        output_height: int,
        feather_source_px: float = 96.0,
        ownership_mode: str = "angular",
    ):
        if output_width != 2 * output_height:
            raise ValueError("ERP output must have an exact 2:1 aspect ratio")
        if ownership_mode not in {"angular", "margin"}:
            raise ValueError(f"unsupported ownership_mode={ownership_mode!r}")
        self.manifest = manifest
        self.width = output_width
        self.height = output_height
        self.feather_source_px = feather_source_px
        self.ownership_mode = ownership_mode
        self.maps = self._build_maps()
        score_field = "axis_score" if ownership_mode == "angular" else "weight"
        scores = np.stack([getattr(self.maps[view], score_field) for view in manifest.view_order])
        owner = np.argmax(scores, axis=0)
        self.owner_labels = owner.astype(np.uint8)
        self.ownership_masks = {
            view: np.where(owner == index, 255, 0).astype(np.uint8)
            for index, view in enumerate(manifest.view_order)
        }
        owner_counts = np.bincount(owner.ravel(), minlength=len(manifest.view_order))
        if np.any(owner_counts == 0):
            empty = [manifest.view_order[index] for index in np.flatnonzero(owner_counts == 0)]
            raise RuntimeError(f"spherical ownership leaves views empty: {empty}")
        print(
            "ownership=" + ownership_mode + " "
            + " ".join(
                f"{view}:{100.0 * owner_counts[index] / owner.size:.2f}%"
                for index, view in enumerate(manifest.view_order)
            ),
            flush=True,
        )
        self.overlap_pairs = self._overlap_pairs()

    def owner_labels_for(self, views: Sequence[str]) -> np.ndarray:
        """Return global manifest labels using only the available views."""
        available = set(views)
        active = tuple(view for view in self.manifest.view_order if view in available)
        if not active:
            raise ValueError("at least one available view is required")
        global_indices = np.asarray(
            [self.manifest.view_order.index(view) for view in active], dtype=np.uint8
        )
        scores = np.stack([self.maps[view].axis_score for view in active])
        uncovered = ~np.any(np.isfinite(scores), axis=0)
        if np.any(uncovered):
            raise RuntimeError(
                f"available frustums leave {np.count_nonzero(uncovered)} ERP pixels uncovered"
            )
        return global_indices[np.argmax(scores, axis=0)]

    def ownership_masks_for(self, views: Sequence[str]) -> dict[str, np.ndarray]:
        labels = self.owner_labels_for(views)
        return {
            view: np.where(labels == index, 255, 0).astype(np.uint8)
            for index, view in enumerate(self.manifest.view_order)
        }

    def _build_maps(self) -> dict[str, Projection]:
        longitude = ((np.arange(self.width, dtype=np.float32) + 0.5) / self.width) * (2 * np.pi) - np.pi
        latitude = np.pi / 2 - ((np.arange(self.height, dtype=np.float32) + 0.5) / self.height) * np.pi
        lon, lat = np.meshgrid(longitude, latitude)
        cos_lat = np.cos(lat)
        base = (cos_lat * np.sin(lon), -np.sin(lat), cos_lat * np.cos(lon))
        result: dict[str, Projection] = {}
        m = self.manifest
        for view in m.view_order:
            rotation = m.rotations[view]
            x = rotation[0, 0] * base[0] + rotation[1, 0] * base[1] + rotation[2, 0] * base[2]
            y = rotation[0, 1] * base[0] + rotation[1, 1] * base[1] + rotation[2, 1] * base[2]
            z = rotation[0, 2] * base[0] + rotation[1, 2] * base[1] + rotation[2, 2] * base[2]
            with np.errstate(divide="ignore", invalid="ignore"):
                map_x = m.fx * x / z + m.cx
                map_y = m.fy * y / z + m.cy
            valid = (
                (z > 0)
                & (map_x >= 0) & (map_x <= m.width - 1)
                & (map_y >= 0) & (map_y <= m.height - 1)
            )
            margin = np.minimum.reduce((map_x, m.width - 1 - map_x, map_y, m.height - 1 - map_y))
            weight = np.where(valid, np.maximum(smoothstep((margin + 0.5) / self.feather_source_px), 1e-5), 0)
            result[view] = Projection(
                np.where(valid, map_x, 0).astype(np.float32),
                np.where(valid, map_y, 0).astype(np.float32),
                valid,
                weight.astype(np.float32),
                np.where(valid, z, -np.inf).astype(np.float32),
            )
        coverage = np.sum([projection.valid for projection in result.values()], axis=0)
        if np.any(coverage == 0):
            raise RuntimeError(f"frustums leave {np.count_nonzero(coverage == 0)} ERP pixels uncovered")
        if np.any(coverage < 2):
            raise RuntimeError(f"overlap contract failed for {np.count_nonzero(coverage < 2)} ERP pixels")
        print(
            f"coverage_min={int(coverage.min())} coverage_mean={float(coverage.mean()):.4f} "
            f"coverage_max={int(coverage.max())}", flush=True,
        )
        return result

    def _overlap_pairs(self) -> tuple[tuple[str, str], ...]:
        pairs = []
        minimum = max(64, self.width * self.height // 10000)
        for first_index, first in enumerate(self.manifest.view_order):
            for second in self.manifest.view_order[first_index + 1:]:
                if int(np.count_nonzero(self.maps[first].valid & self.maps[second].valid)) >= minimum:
                    pairs.append((first, second))
        return tuple(pairs)

    def warp(self, image: np.ndarray, view: str) -> np.ndarray:
        m = self.manifest
        if image.shape[:2] != (m.height, m.width):
            raise ValueError(f"{view} is {image.shape[1]}x{image.shape[0]}, expected {m.width}x{m.height}")
        projection = self.maps[view]
        return cv2.remap(image, projection.map_x, projection.map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

    def stitch(
        self,
        images: Mapping[str, np.ndarray],
        gains: np.ndarray,
        bands: int,
        ownership_masks: Mapping[str, np.ndarray] | None = None,
        blend_mode: str = "multiband",
    ) -> np.ndarray:
        active = tuple(view for view in self.manifest.view_order if view in images)
        if not active:
            raise ValueError("cannot stitch without an available view")
        masks = self.ownership_masks_for(active) if ownership_masks is None else ownership_masks
        mask_coverage = np.any(np.stack([masks[view] != 0 for view in active]), axis=0)
        if np.any(~mask_coverage):
            raise RuntimeError(
                f"available ownership masks leave {np.count_nonzero(~mask_coverage)} ERP pixels uncovered"
            )
        if blend_mode == "hard":
            result = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for index, view in enumerate(self.manifest.view_order):
                if view not in images:
                    continue
                corrected = np.clip(
                    self.warp(images[view], view).astype(np.float32) * gains[index][None, None, :], 0, 255
                ).astype(np.uint8)
                owned = masks[view] != 0
                result[owned] = corrected[owned]
            return result
        if blend_mode == "multiband" and hasattr(cv2, "detail_MultiBandBlender"):
            blender = cv2.detail_MultiBandBlender(0, bands)
            blender.prepare((0, 0, self.width, self.height))
            for index, view in enumerate(self.manifest.view_order):
                if view not in images:
                    continue
                corrected = np.clip(
                    self.warp(images[view], view).astype(np.float32) * gains[index][None, None, :], 0, 255
                ).astype(np.int16)
                blender.feed(corrected, masks[view], (0, 0))
            result, coverage = blender.blend(None, None)
            if result is not None and coverage is not None and np.all(coverage != 0):
                return np.clip(result, 0, 255).astype(np.uint8)
        numerator = np.zeros((self.height, self.width, 3), np.float32)
        denominator = np.zeros((self.height, self.width), np.float32)
        for index, view in enumerate(self.manifest.view_order):
            if view not in images:
                continue
            warped = self.warp(images[view], view).astype(np.float32)
            weight = self.maps[view].weight
            numerator += warped * gains[index][None, None, :] * weight[..., None]
            denominator += weight
        if np.any(denominator <= 0):
            raise RuntimeError(
                f"available feather weights leave {np.count_nonzero(denominator <= 0)} ERP pixels uncovered"
            )
        return np.clip(numerator / denominator[..., None], 0, 255).astype(np.uint8)


def read_images(
    input_dir: Path,
    manifest: Manifest,
    frame: int,
    *,
    allow_missing: bool = False,
) -> dict[str, np.ndarray]:
    images = {}
    for view_index, view in enumerate(manifest.view_order):
        index = view_index * manifest.frames_per_view + frame
        path = input_dir / f"{index:05d}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            if allow_missing:
                continue
            raise FileNotFoundError(f"missing {view} frame {frame}: {path}")
        images[view] = image
    if not images:
        raise FileNotFoundError(f"all inputs are missing for frame {frame}")
    return images


def solve_gains(projector: FrustumProjector, images: Mapping[str, np.ndarray], max_gain: float) -> np.ndarray:
    order = projector.manifest.view_order
    active = tuple(view for view in order if view in images)
    warped = {view: projector.warp(images[view], view).astype(np.float32) for view in active}
    rows, values = [], []
    for first, second in projector.overlap_pairs:
        if first not in images or second not in images:
            continue
        overlap = projector.maps[first].valid & projector.maps[second].valid
        a, b = warped[first][overlap], warped[second][overlap]
        usable = np.all((a > 10) & (a < 245) & (b > 10) & (b < 245), axis=1)
        if np.count_nonzero(usable) < 64:
            continue
        row = np.zeros(len(order), np.float64)
        row[order.index(first)] = 1
        row[order.index(second)] = -1
        rows.append(row)
        values.append(np.median(np.log(b[usable] + 1) - np.log(a[usable] + 1), axis=0))
    anchor = np.zeros(len(order)); anchor[order.index(active[0])] = 8
    rows.append(anchor); values.append(np.zeros(3))
    for index in range(len(order)):
        regularizer = np.zeros(len(order)); regularizer[index] = 0.08
        rows.append(regularizer); values.append(np.zeros(3))
    log_gain = np.linalg.lstsq(np.stack(rows), np.stack(values), rcond=None)[0]
    limit = math.log(max_gain)
    return np.exp(np.clip(log_gain, -limit, limit)).astype(np.float32)


def smooth_gains(raw: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    log_gain = np.log(np.maximum(raw.astype(np.float64), 1e-6))
    median = np.empty_like(log_gain)
    for index in range(len(raw)):
        median[index] = np.median(log_gain[max(0, index - 2):min(len(raw), index + 3)], axis=0)
    forward = median.copy()
    for index in range(1, len(raw)):
        forward[index] = alpha * forward[index - 1] + (1 - alpha) * median[index]
    result = forward.copy()
    for index in range(len(raw) - 2, -1, -1):
        result[index] = alpha * result[index + 1] + (1 - alpha) * forward[index]
    return np.exp(result).astype(np.float32)


def horizontal_graphcut_labels(
    projector: FrustumProjector,
    images: Mapping[str, np.ndarray],
    gains: np.ndarray,
) -> np.ndarray:
    """Move only equatorial seams through low-disagreement overlap regions."""
    order = projector.manifest.view_order
    required = tuple(f"H{yaw:03d}" for yaw in (0, 60, 120, 180, 240, 300))
    if not all(view in order for view in required):
        raise ValueError(f"horizontal graph-cut requires views {required}")
    warped = {
        view: np.clip(
            projector.warp(images[view], view).astype(np.float32)
            * gains[order.index(view)][None, None, :],
            0,
            255,
        ).astype(np.float32)
        for view in required
    }
    labels = projector.owner_labels_for(tuple(images))
    ring = [order.index(view) for view in required]
    for first_index, second_index in zip(ring, ring[1:] + ring[:1]):
        first, second = order[first_index], order[second_index]
        masks = [
            np.where(projector.maps[view].valid, 255, 0).astype(np.uint8)
            for view in (first, second)
        ]
        try:
            result = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD").find(
                [warped[first], warped[second]], [(0, 0), (0, 0)], masks
            )
        except cv2.error as error:
            print(f"warning: graph-cut failed for {first}-{second}: {error}", flush=True)
            continue
        first_mask, second_mask = (item.get() > 0 for item in result)
        pair_region = (labels == first_index) | (labels == second_index)
        labels[pair_region & first_mask & ~second_mask] = first_index
        labels[pair_region & second_mask & ~first_mask] = second_index
    return labels


def all_graphcut_labels(
    projector: FrustumProjector,
    images: Mapping[str, np.ndarray],
    gains: np.ndarray,
) -> np.ndarray:
    """Find low-disagreement seams jointly across every overlapping frustum."""
    order = projector.manifest.view_order
    active = tuple(view for view in order if view in images)
    active_indices = np.asarray([order.index(view) for view in active], dtype=np.uint8)
    fallback = projector.owner_labels_for(active)
    warped = []
    masks = []
    for view in active:
        index = order.index(view)
        warped.append(
            np.clip(
                projector.warp(images[view], view).astype(np.float32)
                * gains[index][None, None, :],
                0,
                255,
            ).astype(np.float32)
        )
        masks.append(np.where(projector.maps[view].valid, 255, 0).astype(np.uint8))

    try:
        cut_masks = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD").find(
            warped, [(0, 0)] * len(active), masks
        )
    except cv2.error as error:
        print(f"warning: all-view graph-cut failed: {error}", flush=True)
        return fallback

    retained = np.stack([item.get() > 0 for item in cut_masks])
    scores = np.stack([projector.maps[view].axis_score for view in active])
    candidates = np.where(retained, scores, -np.inf)
    labels = active_indices[np.argmax(candidates, axis=0)]
    uncovered = ~np.any(retained, axis=0)
    labels[uncovered] = fallback[uncovered]
    return labels


def repair_unavailable_labels(
    projector: FrustumProjector,
    labels: np.ndarray,
    available_views: Sequence[str],
) -> np.ndarray:
    """Remove labels introduced by temporal smoothing when a view is unavailable."""
    active_indices = np.asarray(
        [projector.manifest.view_order.index(view) for view in available_views],
        dtype=np.uint8,
    )
    result = labels.copy()
    invalid = ~np.isin(result, active_indices)
    if np.any(invalid):
        fallback = projector.owner_labels_for(available_views)
        result[invalid] = fallback[invalid]
    return result


def temporal_majority_labels(labels: np.ndarray, radius: int = 1) -> np.ndarray:
    if labels.ndim != 3:
        raise ValueError("labels must have shape [frames, height, width]")
    if radius <= 0 or len(labels) <= 1:
        return labels.copy()
    view_count = int(labels.max()) + 1
    result = np.empty_like(labels)
    for frame in range(len(labels)):
        window = labels[max(0, frame - radius):min(len(labels), frame + radius + 1)]
        counts = np.stack([np.count_nonzero(window == view, axis=0) for view in range(view_count)])
        result[frame] = np.argmax(counts, axis=0).astype(labels.dtype)
    return result


def labels_to_masks(labels: np.ndarray, view_order: Sequence[str], width: int, height: int) -> dict[str, np.ndarray]:
    if labels.shape != (height, width):
        labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)
    return {
        view: np.where(labels == index, 255, 0).astype(np.uint8)
        for index, view in enumerate(view_order)
    }


def encode_video(frames_dir: Path, output: Path, fps: float, crf: int, frame_count: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is unavailable")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "%05d.png"), "-frames:v", str(frame_count),
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], check=True)


def load_frame_indices(path: Path | None, frame_count: int) -> list[int]:
    """Return source timestamps to stitch, preserving the requested order."""
    if path is None:
        return list(range(frame_count))
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("frame_indices", raw.get("qc_selection", {}).get("frame_indices"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("frame-indices JSON must be a non-empty list or contain frame_indices")
    indices = [int(value) for value in raw]
    if len(set(indices)) != len(indices):
        raise ValueError("frame indices must be unique")
    invalid = [value for value in indices if value < 0 or value >= frame_count]
    if invalid:
        raise ValueError(f"frame indices outside [0, {frame_count}): {invalid}")
    return indices


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-video", type=Path)
    parser.add_argument(
        "--frame-indices",
        type=Path,
        help="Optional JSON list of source timestamps to stitch (QC mode).",
    )
    parser.add_argument("--output-width", type=int, default=4096)
    parser.add_argument("--output-height", type=int, default=2048)
    parser.add_argument("--qc-width", type=int, default=1024)
    parser.add_argument("--seam-width", type=int, default=512)
    parser.add_argument(
        "--seam-mode",
        choices=("central", "horizontal-graphcut", "all-graphcut"),
        default="central",
    )
    parser.add_argument("--multiband-bands", type=int, default=6)
    parser.add_argument("--blend-mode", choices=("multiband", "hard", "feather"), default="hard")
    parser.add_argument("--disable-gains", action="store_true")
    parser.add_argument("--feather-source-px", type=float, default=96.0)
    parser.add_argument("--max-gain", type=float, default=1.05)
    parser.add_argument(
        "--ownership-mode",
        choices=("angular", "margin"),
        default="angular",
        help="Assign ERP pixels by nearest optical axis (default) or legacy distance-to-face-edge margin.",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--crf", type=int, default=17)
    parser.add_argument(
        "--allow-missing-inputs",
        action="store_true",
        help=(
            "Skip absent or unreadable frustum images only when the remaining views still "
            "cover every ERP ray. Missing views are excluded from gain solving and graph-cut."
        ),
    )
    args = parser.parse_args(argv)
    manifest = Manifest.load(args.manifest)
    frame_indices = load_frame_indices(args.frame_indices, manifest.frames_per_view)
    missing_inputs: dict[int, list[str]] = {}
    for frame in frame_indices:
        missing = []
        for view_index, view in enumerate(manifest.view_order):
            path = args.input_dir / f"{view_index * manifest.frames_per_view + frame:05d}.png"
            image = (
                cv2.imread(str(path), cv2.IMREAD_COLOR)
                if path.is_file() and path.stat().st_size > 0
                else None
            )
            if image is None or image.shape[:2] != (manifest.height, manifest.width):
                missing.append(view)
        if missing:
            missing_inputs[frame] = missing
    if missing_inputs and not args.allow_missing_inputs:
        first_frame = next(iter(missing_inputs))
        raise FileNotFoundError(
            f"missing inputs for {len(missing_inputs)} frame(s); first frame "
            f"{first_frame}: {missing_inputs[first_frame]}"
        )
    if missing_inputs:
        print(
            f"missing_inputs=allowed frames={len(missing_inputs)} "
            f"images={sum(len(value) for value in missing_inputs.values())}",
            flush=True,
        )

    def load_images(frame: int) -> dict[str, np.ndarray]:
        return read_images(
            args.input_dir,
            manifest,
            frame,
            allow_missing=args.allow_missing_inputs,
        )

    qc = FrustumProjector(
        manifest,
        args.qc_width,
        args.qc_width // 2,
        feather_source_px=24,
        ownership_mode=args.ownership_mode,
    )
    if args.disable_gains:
        gains = np.ones((len(frame_indices), len(manifest.view_order), 3), dtype=np.float32)
        print("gains=disabled", flush=True)
    else:
        raw_gains = []
        for output_frame, frame in enumerate(frame_indices):
            images = load_images(frame)
            qc.owner_labels_for(tuple(images))
            raw_gains.append(solve_gains(qc, images, args.max_gain))
            print(f"[gain {output_frame + 1:03d}/{len(frame_indices):03d}] source={frame:03d}", flush=True)
        gains = smooth_gains(np.stack(raw_gains))
    seam_labels = None
    if args.seam_mode in {"horizontal-graphcut", "all-graphcut"}:
        if args.seam_width <= 0 or args.seam_width % 2:
            raise ValueError("--seam-width must be a positive even integer")
        seam_projector = FrustumProjector(
            manifest,
            args.seam_width,
            args.seam_width // 2,
            feather_source_px=12,
            ownership_mode=args.ownership_mode,
        )
        labels = []
        seam_available_views = []
        for output_frame, frame in enumerate(frame_indices):
            seam_function = (
                horizontal_graphcut_labels
                if args.seam_mode == "horizontal-graphcut"
                else all_graphcut_labels
            )
            images = load_images(frame)
            seam_available_views.append(tuple(images))
            labels.append(seam_function(seam_projector, images, gains[output_frame]))
            print(f"[seam {output_frame + 1:03d}/{len(frame_indices):03d}] source={frame:03d}", flush=True)
        seam_labels = temporal_majority_labels(np.stack(labels), radius=1)
        for output_frame, available_views in enumerate(seam_available_views):
            seam_labels[output_frame] = repair_unavailable_labels(
                seam_projector,
                seam_labels[output_frame],
                available_views,
            )
    projector = FrustumProjector(
        manifest,
        args.output_width,
        args.output_height,
        feather_source_px=args.feather_source_px,
        ownership_mode=args.ownership_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    for output_frame, frame in enumerate(frame_indices):
        images = load_images(frame)
        masks = (
            labels_to_masks(
                repair_unavailable_labels(
                    seam_projector,
                    seam_labels[output_frame],
                    tuple(images),
                ),
                manifest.view_order,
                args.output_width,
                args.output_height,
            )
            if seam_labels is not None
            else None
        )
        panorama = projector.stitch(
            images,
            gains[output_frame],
            args.multiband_bands,
            masks,
            blend_mode=args.blend_mode,
        )
        destination = frames_dir / f"{output_frame:05d}.png"
        if not cv2.imwrite(str(destination), panorama, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
            raise RuntimeError(f"cannot write {destination}")
        print(
            f"[stitch {output_frame + 1:03d}/{len(frame_indices):03d}] "
            f"source={frame:03d} {destination.name}",
            flush=True,
        )
    metadata = {
        "view_order": list(manifest.view_order),
        "frames_per_view": manifest.frames_per_view,
        "stitched_frame_count": len(frame_indices),
        "source_frame_indices": frame_indices,
        "source_size": [manifest.width, manifest.height],
        "output_size": [args.output_width, args.output_height],
        "overlap_pairs": [list(pair) for pair in projector.overlap_pairs],
        "seam_mode": args.seam_mode,
        "seam_width": args.seam_width if seam_labels is not None else None,
        "blend_mode": args.blend_mode,
        "exposure_gains": not args.disable_gains,
        "feather_source_px": args.feather_source_px if args.blend_mode == "feather" else None,
        "allow_missing_inputs": args.allow_missing_inputs,
        "missing_input_count": sum(len(value) for value in missing_inputs.values()),
        "missing_inputs": {str(frame): views for frame, views in missing_inputs.items()},
        "minimum_available_views": min(
            len(manifest.view_order) - len(missing_inputs.get(frame, ()))
            for frame in frame_indices
        ),
    }
    (args.output_dir / "panorama_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.output_video is not None:
        encode_video(frames_dir, args.output_video, args.fps, args.crf, len(frame_indices))
        print(f"video={args.output_video}", flush=True)


if __name__ == "__main__":
    main()
