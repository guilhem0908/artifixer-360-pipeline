#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Stitch six synchronized pinhole views into full equirectangular panoramas.

Input images are six contiguous frame blocks in F/R/B/L/U/D order.  Camera
intrinsics and the local OpenGL face rotations are read from the manifest made
by ``generate_cubemap_trajectories.py``.  Projection uses pixel-centre rays;
no panorama pixel is extrapolated or invented.

The default blender uses deterministic geometric ownership masks and OpenCV's
multi-band blender.  A portable spherical feather blender remains available as
a fallback when the optional OpenCV stitching/detail module is unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


FACE_ORDER = ("F", "R", "B", "L", "U", "D")
FACE_NAMES = {
    "F": "front",
    "R": "right",
    "B": "back",
    "L": "left",
    "U": "up",
    "D": "down",
}
EDGE_PAIRS = (
    ("F", "R"), ("R", "B"), ("B", "L"), ("L", "F"),
    ("F", "U"), ("R", "U"), ("B", "U"), ("L", "U"),
    ("F", "D"), ("R", "D"), ("B", "D"), ("L", "D"),
)

# Camera-to-base rotations for OpenCV rays (+X right, +Y down, +Z forward).
DEFAULT_CAMERA_TO_BASE_CV = {
    "F": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64),
    "R": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], np.float64),
    "B": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], np.float64),
    "L": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], np.float64),
    "U": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64),
    "D": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64),
}
OPENGL_CV_CHANGE = np.diag((1.0, -1.0, -1.0))


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _face(value: object) -> str:
    aliases = {
        "F": "F", "FRONT": "F", "R": "R", "RIGHT": "R",
        "B": "B", "BACK": "B", "L": "L", "LEFT": "L",
        "U": "U", "UP": "U", "D": "D", "DOWN": "D",
    }
    key = str(value).strip().upper()
    if key not in aliases:
        raise ValueError(f"unknown cubemap face {value!r}")
    return aliases[key]


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if self.width <= 1 or self.height <= 1:
            raise ValueError(f"invalid source size {self.width}x{self.height}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("intrinsics must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("fx and fy must be positive")

    @property
    def hfov_degrees(self) -> float:
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.fx)))

    @property
    def vfov_degrees(self) -> float:
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.fy)))


@dataclass(frozen=True)
class CubemapManifest:
    intrinsics: Intrinsics
    frames_per_face: int
    rotations: dict[str, np.ndarray]
    source_path: Path
    raw: dict[str, Any]
    rotation_conversion: str

    @classmethod
    def load(cls, path: Path) -> "CubemapManifest":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read manifest {path}: {error}") from error
        if not isinstance(raw, dict):
            raise ValueError("manifest root must be an object")

        order = tuple(_face(value) for value in raw.get("face_order", FACE_ORDER))
        if order != FACE_ORDER:
            raise ValueError(f"face_order must be {FACE_ORDER}, got {order}")

        nested = raw.get("intrinsics")
        intr = nested if isinstance(nested, Mapping) else raw
        size = raw.get("size", intr.get("size"))
        if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
            width, height = int(size[0]), int(size[1])
        elif size is not None:
            width = height = int(size)
        else:
            width = int(intr.get("width", intr.get("w", raw.get("width", raw.get("w", 0)))))
            height = int(intr.get("height", intr.get("h", raw.get("height", raw.get("h", 0)))))

        def number(*keys: str) -> float:
            for key in keys:
                if key in intr:
                    return float(intr[key])
                if key in raw:
                    return float(raw[key])
            raise ValueError(f"manifest is missing {'/'.join(keys)}")

        camera = Intrinsics(
            width=width,
            height=height,
            fx=number("fx", "fl_x"),
            fy=number("fy", "fl_y"),
            cx=number("cx"),
            cy=number("cy"),
        )
        camera.validate()
        frames = int(raw.get("frames_per_face", 154))
        if frames <= 0:
            raise ValueError("frames_per_face must be positive")

        matrices = raw.get("local_rotations")
        if matrices is None:
            rotations = {face: matrix.copy() for face, matrix in DEFAULT_CAMERA_TO_BASE_CV.items()}
            conversion = "canonical_opencv_fallback"
        else:
            if not isinstance(matrices, Mapping):
                if not isinstance(matrices, Sequence) or len(matrices) != 6:
                    raise ValueError("local_rotations must be a face mapping or six matrices")
                matrices = dict(zip(FACE_ORDER, matrices))
            convention = str(raw.get("rotation_convention", "")).lower()
            pose_convention = str(raw.get("pose_convention", "")).lower()
            is_opengl = (
                "opengl" in convention
                or "c2w_postmultiply" in convention
                or (not convention and "opengl" in pose_convention)
            )
            rotations = {}
            for face in FACE_ORDER:
                source = matrices.get(face, matrices.get(FACE_NAMES[face]))
                if source is None:
                    raise ValueError(f"local_rotations is missing face {face}")
                matrix = np.asarray(source, dtype=np.float64)
                if matrix.shape == (4, 4):
                    matrix = matrix[:3, :3]
                if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                    raise ValueError(f"rotation {face} must be a finite 3x3 or 4x4 matrix")
                if is_opengl:
                    matrix = OPENGL_CV_CHANGE @ matrix @ OPENGL_CV_CHANGE
                elif "base_to_camera" in convention:
                    matrix = matrix.T
                error = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
                determinant = float(np.linalg.det(matrix))
                if error > 1e-5 or abs(determinant - 1.0) > 1e-5:
                    raise ValueError(
                        f"rotation {face} is not orthonormal (error={error}, det={determinant})"
                    )
                rotations[face] = matrix
            conversion = "C @ Q_opengl @ C" if is_opengl else (convention or "camera_to_base_opencv")

        return cls(camera, frames, rotations, path, raw, conversion)


@dataclass(frozen=True)
class ProjectionMap:
    map_x: np.ndarray
    map_y: np.ndarray
    valid: np.ndarray
    weight: np.ndarray


class CubemapProjector:
    """Exact pinhole-to-equirectangular geometry for one output resolution."""

    def __init__(
        self,
        intrinsics: Intrinsics,
        rotations: Mapping[str, np.ndarray],
        width: int = 4096,
        height: int = 2048,
        feather_source_px: float | None = None,
        blend_mode: str = "auto",
        multiband_bands: int = 5,
    ) -> None:
        if width <= 0 or height <= 0 or width != 2 * height:
            raise ValueError("equirectangular output must have a positive exact 2:1 size")
        self.intrinsics = intrinsics
        self.rotations = {face: np.asarray(rotations[face], np.float64) for face in FACE_ORDER}
        self.width = int(width)
        self.height = int(height)
        seam_margins = (
            intrinsics.cx - intrinsics.fx,
            (intrinsics.width - 1.0) - (intrinsics.cx + intrinsics.fx),
            intrinsics.cy - intrinsics.fy,
            (intrinsics.height - 1.0) - (intrinsics.cy + intrinsics.fy),
        )
        automatic = 2.0 * max(2.0, min(seam_margins))
        self.feather_source_px = float(automatic if feather_source_px is None else feather_source_px)
        if self.feather_source_px <= 0.0:
            raise ValueError("feather_source_px must be positive")
        if blend_mode not in {"auto", "multiband", "feather", "hard"}:
            raise ValueError("blend_mode must be auto, multiband, feather, or hard")
        if multiband_bands <= 0:
            raise ValueError("multiband_bands must be positive")
        self.requested_blend_mode = blend_mode
        self.multiband_bands_requested = int(multiband_bands)
        half_overlap_degrees = max(
            0.0, min(intrinsics.hfov_degrees, intrinsics.vfov_degrees) / 2.0 - 45.0
        )
        ownership_to_fov_edge_px = max(1.0, self.width * half_overlap_degrees / 360.0)
        safe_bands = max(1, int(math.floor(math.log2(ownership_to_fov_edge_px))))
        self.multiband_bands = min(self.multiband_bands_requested, safe_bands)
        self._multiband_enabled = (
            blend_mode not in {"feather", "hard"} and hasattr(cv2, "detail_MultiBandBlender")
        )
        self.last_blend_mode = (
            "hard" if blend_mode == "hard"
            else ("multiband" if self._multiband_enabled else "feather")
        )
        self._reported_multiband_fallback = False
        self.maps = self._build_maps()
        self.ownership_masks = self._build_ownership_masks()

    def _build_maps(self) -> dict[str, ProjectionMap]:
        longitude = ((np.arange(self.width, dtype=np.float32) + 0.5) / self.width) * (2.0 * np.pi) - np.pi
        latitude = (np.pi / 2.0) - ((np.arange(self.height, dtype=np.float32) + 0.5) / self.height) * np.pi
        lon, lat = np.meshgrid(longitude, latitude)
        cos_lat = np.cos(lat)
        direction = (
            cos_lat * np.sin(lon),
            -np.sin(lat),
            cos_lat * np.cos(lon),
        )
        result: dict[str, ProjectionMap] = {}
        intr = self.intrinsics
        for face in FACE_ORDER:
            rotation = self.rotations[face]
            # p_camera = R_camera_to_base.T @ p_base
            x = rotation[0, 0] * direction[0] + rotation[1, 0] * direction[1] + rotation[2, 0] * direction[2]
            y = rotation[0, 1] * direction[0] + rotation[1, 1] * direction[1] + rotation[2, 1] * direction[2]
            z = rotation[0, 2] * direction[0] + rotation[1, 2] * direction[1] + rotation[2, 2] * direction[2]
            with np.errstate(divide="ignore", invalid="ignore"):
                map_x = intr.fx * (x / z) + intr.cx
                map_y = intr.fy * (y / z) + intr.cy
            valid = (
                (z > 0.0)
                # Match gs360nav's strict cubemap raster convention: a ray
                # landing exactly on the outer face boundary (coordinate W/H)
                # belongs to that face and samples its replicated edge pixel.
                & (map_x >= 0.0) & (map_x <= intr.width)
                & (map_y >= 0.0) & (map_y <= intr.height)
            )
            margin = np.minimum.reduce((
                map_x, (intr.width - 1.0) - map_x,
                map_y, (intr.height - 1.0) - map_y,
            ))
            weight = _smoothstep((margin + 0.5) / self.feather_source_px)
            weight = np.where(valid, np.maximum(weight, 1e-5), 0.0).astype(np.float32)
            result[face] = ProjectionMap(
                np.where(valid, np.clip(map_x, 0.0, intr.width - 1.0), 0.0).astype(np.float32),
                np.where(valid, np.clip(map_y, 0.0, intr.height - 1.0), 0.0).astype(np.float32),
                valid,
                weight,
            )
        coverage = np.sum([item.valid for item in result.values()], axis=0)
        if np.any(coverage == 0):
            raise RuntimeError(f"cubemap leaves {int(np.count_nonzero(coverage == 0))} panorama pixels uncovered")
        return result

    def _build_ownership_masks(self) -> dict[str, np.ndarray]:
        """Assign every panorama pixel to its most central valid face."""
        scores = np.stack([self.maps[face].weight for face in FACE_ORDER], axis=0)
        owner = np.argmax(scores, axis=0)
        masks = {
            face: np.where(owner == index, 255, 0).astype(np.uint8)
            for index, face in enumerate(FACE_ORDER)
        }
        coverage = np.sum([mask > 0 for mask in masks.values()], axis=0)
        if not np.all(coverage == 1):
            raise RuntimeError("geometric ownership must assign every panorama pixel exactly once")
        return masks

    def warp(self, image: np.ndarray, face: str) -> np.ndarray:
        if image.shape[:2] != (self.intrinsics.height, self.intrinsics.width):
            raise ValueError(
                f"{face} image is {image.shape[1]}x{image.shape[0]}, expected "
                f"{self.intrinsics.width}x{self.intrinsics.height}"
            )
        mapping = self.maps[face]
        return cv2.remap(
            image, mapping.map_x, mapping.map_y, cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def pixel_to_base_direction(self, face: str, u: float, v: float) -> np.ndarray:
        camera = np.array(
            [(u - self.intrinsics.cx) / self.intrinsics.fx,
             (v - self.intrinsics.cy) / self.intrinsics.fy, 1.0],
            dtype=np.float64,
        )
        base = self.rotations[face] @ camera
        return base / np.linalg.norm(base)

    def base_direction_to_pixel(self, face: str, direction: np.ndarray) -> tuple[float, float]:
        camera = self.rotations[face].T @ np.asarray(direction, np.float64)
        if camera[2] <= 0.0:
            raise ValueError(f"direction is behind face {face}")
        return (
            self.intrinsics.fx * camera[0] / camera[2] + self.intrinsics.cx,
            self.intrinsics.fy * camera[1] / camera[2] + self.intrinsics.cy,
        )

    def _stitch_feather(self, images: Mapping[str, np.ndarray], gains: np.ndarray) -> np.ndarray:
        numerator = np.zeros((self.height, self.width, 3), np.float32)
        denominator = np.zeros((self.height, self.width), np.float32)
        for face_index, face in enumerate(FACE_ORDER):
            warped = self.warp(images[face], face).astype(np.float32)
            weight = self.maps[face].weight
            numerator += warped * gains[face_index][None, None, :] * weight[..., None]
            denominator += weight
        if np.any(denominator <= 0.0):
            raise RuntimeError("feather blend has uncovered output pixels")
        return np.clip(numerator / denominator[..., None], 0.0, 255.0).astype(np.uint8)

    def _stitch_multiband(self, images: Mapping[str, np.ndarray], gains: np.ndarray) -> np.ndarray:
        blender = cv2.detail_MultiBandBlender(0, self.multiband_bands)
        blender.prepare((0, 0, self.width, self.height))
        for face_index, face in enumerate(FACE_ORDER):
            warped = self.warp(images[face], face).astype(np.float32)
            corrected = np.clip(
                warped * gains[face_index][None, None, :], 0.0, 255.0
            ).astype(np.int16)
            blender.feed(corrected, self.ownership_masks[face], (0, 0))
        blended, coverage = blender.blend(None, None)
        if blended is None or coverage is None or np.any(coverage == 0):
            raise RuntimeError("multi-band blend has uncovered output pixels")
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _stitch_hard(self, images: Mapping[str, np.ndarray], gains: np.ndarray) -> np.ndarray:
        """Strict dominant-axis cubemap reprojection, matching gs360nav opencv_fast.

        Every output ray is sampled from exactly one canonical 90-degree face.
        There is no feathering, multi-band averaging, or seam blur.
        """
        panorama = np.zeros((self.height, self.width, 3), np.uint8)
        for face_index, face in enumerate(FACE_ORDER):
            warped = self.warp(images[face], face).astype(np.float32)
            corrected = np.clip(warped * gains[face_index][None, None, :], 0.0, 255.0).astype(np.uint8)
            mask = self.ownership_masks[face] > 0
            panorama[mask] = corrected[mask]
        return panorama

    def stitch(self, images: Mapping[str, np.ndarray], gains: np.ndarray) -> np.ndarray:
        if self.requested_blend_mode == "hard":
            self.last_blend_mode = "hard"
            return self._stitch_hard(images, gains)
        if self._multiband_enabled:
            try:
                panorama = self._stitch_multiband(images, gains)
                self.last_blend_mode = "multiband"
                return panorama
            except (AttributeError, TypeError, cv2.error, RuntimeError) as error:
                self._multiband_enabled = False
                self.last_blend_mode = "feather"
                if not self._reported_multiband_fallback:
                    print(f"warning: multi-band blend unavailable, using feather: {error}", file=sys.stderr)
                    self._reported_multiband_fallback = True
        self.last_blend_mode = "feather"
        return self._stitch_feather(images, gains)


def format_input_path(directory: Path, pattern: str, index: int, face: str, frame: int) -> Path:
    try:
        name = pattern.format(index=index, face=face, frame=frame)
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid input pattern {pattern!r}: {error}") from error
    return directory / name


def read_face_images(
    directory: Path, pattern: str, frame_index: int, frame_count: int,
) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for face_index, face in enumerate(FACE_ORDER):
        source_index = face_index * frame_count + frame_index
        path = format_input_path(directory, pattern, source_index, face, frame_index)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"missing {face} frame {frame_index}: {path}")
        images[face] = image
    return images


def warped_views(
    projector: CubemapProjector, images: Mapping[str, np.ndarray], gains: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if gains is None:
        gains = np.ones((6, 3), np.float32)
    for index, face in enumerate(FACE_ORDER):
        image = projector.warp(images[face], face).astype(np.float32)
        result[face] = np.clip(image * gains[index][None, None, :], 0.0, 255.0)
    return result


def solve_global_gains(
    views: Mapping[str, np.ndarray], maps: Mapping[str, ProjectionMap], max_gain: float = 1.12,
) -> np.ndarray:
    """Solve one exposure gain per face/channel on the twelve-edge graph."""
    if max_gain <= 1.0:
        raise ValueError("max_gain must be greater than one")
    rows: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for first, second in EDGE_PAIRS:
        overlap = maps[first].valid & maps[second].valid
        a = views[first][overlap]
        b = views[second][overlap]
        usable = np.all((a > 10.0) & (a < 245.0) & (b > 10.0) & (b < 245.0), axis=1)
        if int(np.count_nonzero(usable)) < 64:
            continue
        a = a[usable]
        b = b[usable]
        difference = np.median(np.log(b + 1.0) - np.log(a + 1.0), axis=0)
        row = np.zeros(6, np.float64)
        row[FACE_ORDER.index(first)] = 1.0
        row[FACE_ORDER.index(second)] = -1.0
        rows.append(row)
        values.append(difference.astype(np.float64))

    # Front is the global exposure anchor; weak regularization avoids drift on
    # textureless edges without overpowering measured overlap relationships.
    anchor = np.zeros(6, np.float64)
    anchor[0] = 8.0
    rows.append(anchor)
    values.append(np.zeros(3, np.float64))
    for index in range(6):
        regularizer = np.zeros(6, np.float64)
        regularizer[index] = 0.08
        rows.append(regularizer)
        values.append(np.zeros(3, np.float64))
    matrix = np.stack(rows)
    rhs = np.stack(values)
    log_gain = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    limit = math.log(max_gain)
    return np.exp(np.clip(log_gain, -limit, limit)).astype(np.float32)


def temporal_smooth_gains(
    raw: np.ndarray, frame_indices: Sequence[int], median_window: int = 5, alpha: float = 0.65,
) -> np.ndarray:
    """Median then zero-phase exponential smoothing in log-gain space."""
    if raw.ndim != 3 or raw.shape[1:] != (6, 3):
        raise ValueError("raw gains must have shape [frames, 6, 3]")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("gain smoothing alpha must be in [0, 1)")
    if len(raw) <= 1:
        return raw.copy()
    log_gain = np.log(np.maximum(raw.astype(np.float64), 1e-6))
    radius = max(0, int(median_window) // 2)
    median = np.empty_like(log_gain)
    for index in range(len(raw)):
        lo, hi = max(0, index - radius), min(len(raw), index + radius + 1)
        median[index] = np.median(log_gain[lo:hi], axis=0)
    forward = median.copy()
    for index in range(1, len(raw)):
        gap = max(1, int(frame_indices[index]) - int(frame_indices[index - 1]))
        keep = alpha ** gap
        forward[index] = keep * forward[index - 1] + (1.0 - keep) * median[index]
    backward = forward.copy()
    for index in range(len(raw) - 2, -1, -1):
        gap = max(1, int(frame_indices[index + 1]) - int(frame_indices[index]))
        keep = alpha ** gap
        backward[index] = keep * backward[index + 1] + (1.0 - keep) * forward[index]
    return np.exp(backward).astype(np.float32)


def _feature_displacement(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Return p95 LK displacement and forward/back track fraction in an overlap.

    Phase correlation on two images multiplied by the same hard overlap mask is
    biased toward zero by the mask boundary itself.  Sparse pyramidal LK avoids
    that failure: corners are detected inside a lightly eroded overlap, tracked
    in both directions, and rejected when the forward/back error is too large.
    """
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    if int(np.count_nonzero(mask_u8)) < 64:
        return 0.0, 0.0
    safe_mask = cv2.erode(mask_u8, np.ones((5, 5), np.uint8), iterations=1)
    if int(np.count_nonzero(safe_mask)) < 64:
        safe_mask = mask_u8

    first = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    second = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(
        first,
        maxCorners=256,
        qualityLevel=0.01,
        minDistance=7.0,
        mask=safe_mask,
        blockSize=5,
        useHarrisDetector=False,
    )
    if points is None or len(points) < 8:
        return 0.0, 0.0

    source = points.reshape(-1, 2)
    best_displacement = np.empty(0, dtype=np.float32)
    accepted = 0
    # A single coarse pyramid can jump from a narrow overlap to a similar
    # feature elsewhere in the panorama.  Evaluate local, medium and coarse
    # searches independently and retain the one with the most validated tracks.
    for max_level in (0, 2, 4):
        lk = {
            "winSize": (7, 7),
            "maxLevel": max_level,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            "minEigThreshold": 1e-4,
        }
        forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(first, second, points, None, **lk)
        if forward is None or status_forward is None:
            continue
        backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(second, first, forward, None, **lk)
        if backward is None or status_backward is None:
            continue

        target = forward.reshape(-1, 2)
        returned = backward.reshape(-1, 2)
        finite = np.all(np.isfinite(target), axis=1) & np.all(np.isfinite(returned), axis=1)
        valid = (status_forward.reshape(-1) > 0) & (status_backward.reshape(-1) > 0) & finite
        target_x = np.rint(np.where(finite, target[:, 0], -1.0)).astype(np.int64)
        target_y = np.rint(np.where(finite, target[:, 1], -1.0)).astype(np.int64)
        inside = (
            (target_x >= 0) & (target_x < safe_mask.shape[1])
            & (target_y >= 0) & (target_y < safe_mask.shape[0])
        )
        in_overlap = np.zeros(len(points), dtype=bool)
        in_overlap[inside] = safe_mask[target_y[inside], target_x[inside]] > 0
        forward_backward_error = np.linalg.norm(returned - source, axis=1)
        valid &= in_overlap & (forward_backward_error <= 1.5)
        candidate_count = int(np.count_nonzero(valid))
        if candidate_count > accepted:
            accepted = candidate_count
            best_displacement = np.linalg.norm(target[valid] - source[valid], axis=1)

    track_fraction = accepted / len(points)
    minimum_reliable = max(8, int(math.ceil(0.10 * len(points))))
    if accepted < minimum_reliable:
        # A textured overlap whose features cannot be tracked consistently is
        # itself strong evidence of structural disagreement.  Textureless
        # overlaps already returned above before reaching this branch.
        return float(0.25 * min(first.shape)), float(track_fraction)
    return float(np.percentile(best_displacement, 95.0)), float(track_fraction)


def pair_metrics(
    views: Mapping[str, np.ndarray], maps: Mapping[str, ProjectionMap], first: str, second: str,
) -> dict[str, float]:
    overlap = maps[first].valid & maps[second].valid
    a, b = views[first], views[second]
    if int(np.count_nonzero(overlap)) < 64:
        return {"mad_rgb": 0.0, "structural_displacement_px": 0.0,
                "track_fraction": 0.0, "gradient_discontinuity_ratio": 0.0}
    difference = np.abs(a[overlap].astype(np.float32) - b[overlap].astype(np.float32))
    mad = float(np.median(difference))
    gray_a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_a = cv2.magnitude(cv2.Sobel(gray_a, cv2.CV_32F, 1, 0), cv2.Sobel(gray_a, cv2.CV_32F, 0, 1))
    grad_b = cv2.magnitude(cv2.Sobel(gray_b, cv2.CV_32F, 1, 0), cv2.Sobel(gray_b, cv2.CV_32F, 0, 1))
    gradient_ratio = float(
        np.median(np.abs(grad_a[overlap] - grad_b[overlap]))
        / (np.median(0.5 * (grad_a[overlap] + grad_b[overlap])) + 1.0)
    )
    displacement, response = _feature_displacement(a, b, overlap)
    return {
        "mad_rgb": round(mad, 6),
        "structural_displacement_px": round(displacement, 6),
        "track_fraction": round(response, 6),
        "gradient_discontinuity_ratio": round(gradient_ratio, 6),
    }


def assess_pair(
    current: dict[str, float], reference: dict[str, float] | None,
    mad_limit: float, displacement_limit: float, gradient_limit: float,
) -> tuple[list[str], dict[str, float]]:
    thresholds = {
        "mad_rgb": max(mad_limit, 1.5 * reference["mad_rgb"] + 2.0) if reference else mad_limit,
        "structural_displacement_px": displacement_limit,
        "gradient_discontinuity_ratio": (
            max(gradient_limit, 1.5 * reference["gradient_discontinuity_ratio"])
            if reference else gradient_limit
        ),
    }
    reasons = [name for name, limit in thresholds.items() if current[name] > limit]
    return reasons, {name: round(value, 6) for name, value in thresholds.items()}


def parse_frame_indices(text: str | None, frame_count: int) -> list[int]:
    if text is None:
        return list(range(frame_count))
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            values.extend(range(start, stop + 1))
        else:
            values.append(int(token))
    values = list(dict.fromkeys(values))
    if not values or any(value < 0 or value >= frame_count for value in values):
        raise ValueError(f"frame indices must be inside [0, {frame_count - 1}]")
    return values


def encode_video(frames_dir: Path, output: Path, fps: float, crf: int, frame_count: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was requested but is not available; omit --output-video and encode on host")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"),
        "-frames:v", str(frame_count), "-c:v", "libx264", "-preset", "slow",
        "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", "--input_dir", type=Path, required=True)
    parser.add_argument("--reference-dir", "--reference_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True)
    parser.add_argument("--output-video", "--output_video", type=Path)
    parser.add_argument("--no-video", "--no_video", action="store_true")
    parser.add_argument("--frame-count", "--frame_count", type=int)
    parser.add_argument("--frame-indices", "--frame_indices")
    parser.add_argument("--input-pattern", default="{index:05d}.png")
    parser.add_argument("--reference-pattern", default="{index:05d}.png")
    parser.add_argument("--output-width", type=int, default=4096)
    parser.add_argument("--output-height", type=int, default=2048)
    parser.add_argument("--qc-width", type=int, default=1024)
    parser.add_argument("--feather-source-px", type=float)
    parser.add_argument("--blend-mode", choices=("auto", "multiband", "feather", "hard"), default="auto")
    parser.add_argument(
        "--disable-gains", action="store_true",
        help="Keep all six faces at unit gain; useful for a single radiance field rendered with shared exposure.",
    )
    parser.add_argument("--multiband-bands", type=int, default=5)
    parser.add_argument("--max-gain", type=float, default=1.12)
    parser.add_argument("--gain-median-window", type=int, default=5)
    parser.add_argument("--gain-smoothing-alpha", type=float, default=0.65)
    parser.add_argument("--qc-mad-limit", type=float, default=18.0)
    parser.add_argument("--qc-displacement-limit", type=float, default=16.0)
    parser.add_argument("--qc-gradient-limit", type=float, default=1.5)
    parser.add_argument("--qc-failure-fraction", type=float, default=0.10)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--crf", type=int, default=17)
    parser.add_argument("--smoke", action="store_true", help="stitch frames 0/middle/last at at most 1024x512")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = CubemapManifest.load(args.manifest)
    frame_count = manifest.frames_per_face if args.frame_count is None else int(args.frame_count)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if args.frame_count is not None and frame_count != manifest.frames_per_face:
        raise ValueError(
            f"--frame-count {frame_count} disagrees with manifest frames_per_face={manifest.frames_per_face}"
        )
    frame_indices = parse_frame_indices(args.frame_indices, frame_count)
    output_width, output_height = args.output_width, args.output_height
    if args.smoke:
        if args.frame_indices is None:
            frame_indices = sorted({0, frame_count // 2, frame_count - 1})
        if (output_width, output_height) == (4096, 2048):
            output_width, output_height = 1024, 512
    if output_width != 2 * output_height:
        raise ValueError("--output-width must equal twice --output-height")
    if args.qc_width <= 0 or args.qc_width % 2:
        raise ValueError("--qc-width must be a positive even integer")
    if args.multiband_bands <= 0:
        raise ValueError("--multiband-bands must be positive")
    qc_width = min(int(args.qc_width), output_width)
    qc_height = qc_width // 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"projection=equirectangular size={output_width}x{output_height} "
        f"qc={qc_width}x{qc_height} frames={len(frame_indices)}",
        flush=True,
    )
    print(f"rotation_conversion={manifest.rotation_conversion}", flush=True)

    qc_projector = CubemapProjector(
        manifest.intrinsics, manifest.rotations, qc_width, qc_height,
        args.feather_source_px, blend_mode="feather",
    )
    raw_gains: list[np.ndarray] = []
    for progress, frame_index in enumerate(frame_indices, 1):
        images = read_face_images(args.input_dir, args.input_pattern, frame_index, frame_count)
        raw_views = warped_views(qc_projector, images)
        raw_gains.append(
            np.ones((6, 3), np.float32)
            if args.disable_gains else solve_global_gains(raw_views, qc_projector.maps, args.max_gain)
        )
        print(f"[gain {progress:03d}/{len(frame_indices):03d}] frame={frame_index:05d}", flush=True)
    raw_gain_array = np.stack(raw_gains)
    gains = temporal_smooth_gains(
        raw_gain_array, frame_indices, args.gain_median_window, args.gain_smoothing_alpha
    )

    projector = CubemapProjector(
        manifest.intrinsics, manifest.rotations, output_width, output_height,
        args.feather_source_px, blend_mode=args.blend_mode,
        multiband_bands=args.multiband_bands,
    )
    print(
        f"blend_requested={args.blend_mode} blend_selected={projector.last_blend_mode} "
        f"multiband_bands={projector.multiband_bands} "
        f"multiband_bands_requested={args.multiband_bands}",
        flush=True,
    )
    qc_frames: list[dict[str, Any]] = []
    frame_metadata: list[dict[str, Any]] = []
    bad_counts = {f"{a}-{b}": 0 for a, b in EDGE_PAIRS}
    for progress, (gain_index, frame_index) in enumerate(zip(range(len(frame_indices)), frame_indices), 1):
        images = read_face_images(args.input_dir, args.input_pattern, frame_index, frame_count)
        current_views = warped_views(qc_projector, images, gains[gain_index])
        reference_views: dict[str, np.ndarray] | None = None
        if args.reference_dir is not None:
            reference_images = read_face_images(
                args.reference_dir, args.reference_pattern, frame_index, frame_count
            )
            reference_raw = warped_views(qc_projector, reference_images)
            reference_gain = solve_global_gains(reference_raw, qc_projector.maps, args.max_gain)
            reference_views = warped_views(qc_projector, reference_images, reference_gain)

        pairs: dict[str, Any] = {}
        for first, second in EDGE_PAIRS:
            name = f"{first}-{second}"
            current = pair_metrics(current_views, qc_projector.maps, first, second)
            reference = (
                pair_metrics(reference_views, qc_projector.maps, first, second)
                if reference_views is not None else None
            )
            reasons, thresholds = assess_pair(
                current, reference, args.qc_mad_limit,
                args.qc_displacement_limit, args.qc_gradient_limit,
            )
            if reasons:
                bad_counts[name] += 1
            pairs[name] = {
                "metrics": current,
                "reference_metrics": reference,
                "thresholds": thresholds,
                "failed": bool(reasons),
                "reasons": reasons,
            }

        panorama = projector.stitch(images, gains[gain_index])
        destination = frames_dir / f"{frame_index:05d}.png"
        if not cv2.imwrite(str(destination), panorama, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
            raise RuntimeError(f"cannot write {destination}")
        qc_frames.append({"frame_index": frame_index, "pairs": pairs})
        frame_metadata.append({
            "frame_index": frame_index,
            "output": str(destination.relative_to(args.output_dir)),
            "raw_gains_bgr": {
                face: [round(float(x), 7) for x in raw_gain_array[gain_index, index]]
                for index, face in enumerate(FACE_ORDER)
            },
            "smoothed_gains_bgr": {
                face: [round(float(x), 7) for x in gains[gain_index, index]]
                for index, face in enumerate(FACE_ORDER)
            },
        })
        print(f"[stitch {progress:03d}/{len(frame_indices):03d}] {destination.name}", flush=True)

    adjacency: dict[str, Any] = {}
    needs_a3d = False
    for name, count in bad_counts.items():
        fraction = count / len(frame_indices)
        failed = fraction > args.qc_failure_fraction
        needs_a3d = needs_a3d or failed
        adjacency[name] = {
            "bad_frames": count,
            "assessed_frames": len(frame_indices),
            "failure_fraction": round(fraction, 7),
            "needs_a3d": failed,
        }
    qc = {
        "schema_version": 1,
        "metric_resolution": [qc_width, qc_height],
        "metric_method": {
            "color": "overlap median absolute BGR difference",
            "structure": "p95 sparse pyramidal-LK displacement with forward/back validation",
            "gradient": "median gradient disagreement / median local gradient",
        },
        "reference_dir": str(args.reference_dir) if args.reference_dir is not None else None,
        "failure_rule": f"any adjacency with failure_fraction > {args.qc_failure_fraction}",
        "adjacencies": adjacency,
        "frames": qc_frames,
        "needs_a3d": needs_a3d,
    }
    (args.output_dir / "qc.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "needs_a3d.env").write_text(
        f"NEEDS_A3D={'true' if needs_a3d else 'false'}\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": 1,
        "projection": "full_equirectangular_pixel_centres",
        "coordinate_system": "+X right, +Y down, +Z forward; longitude [-pi,pi), latitude (+pi/2,-pi/2)",
        "view_order": list(FACE_ORDER),
        "source_size": [manifest.intrinsics.width, manifest.intrinsics.height],
        "intrinsics": {
            "fx": manifest.intrinsics.fx, "fy": manifest.intrinsics.fy,
            "cx": manifest.intrinsics.cx, "cy": manifest.intrinsics.cy,
        },
        "source_fov_degrees": [manifest.intrinsics.hfov_degrees, manifest.intrinsics.vfov_degrees],
        "rotation_conversion": manifest.rotation_conversion,
        "camera_to_base_opencv": {face: manifest.rotations[face].tolist() for face in FACE_ORDER},
        "output_size": [output_width, output_height],
        "fps": args.fps,
        "blend": (
            "global temporal RGB gains + deterministic geometric-ownership "
            f"{projector.last_blend_mode}"
        ),
        "blend_mode_requested": args.blend_mode,
        "blend_mode_used": projector.last_blend_mode,
        "multiband_bands": (
            projector.multiband_bands if projector.last_blend_mode == "multiband" else None
        ),
        "multiband_bands_requested": args.multiband_bands,
        "feather_source_px": projector.feather_source_px,
        "frames": frame_metadata,
    }
    (args.output_dir / "panorama_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    complete = frame_indices == list(range(frame_count))
    if args.output_video is not None and not args.no_video:
        if not complete:
            raise ValueError("--output-video requires the complete sequential frame set")
        encode_video(frames_dir, args.output_video, args.fps, args.crf, frame_count)
        print(f"video={args.output_video}", flush=True)
    print(f"needs_a3d={'true' if needs_a3d else 'false'}", flush=True)
    return {"metadata": metadata, "qc": qc}


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
