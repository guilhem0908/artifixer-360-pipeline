#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Generic spatial and temporal quality metrics for ERP panorama sequences."""

from __future__ import annotations

import cv2
import numpy as np


def erp_seam_metrics(image: np.ndarray) -> dict[str, float]:
    """Measure the discontinuity between the first and last ERP columns."""

    if image.ndim != 3 or image.shape[1] < 4:
        raise ValueError("ERP seam requires an HxWx3 image")
    values = image.astype(np.float32) / 255.0
    seam_delta = np.mean(np.abs(values[:, 0] - values[:, -1]), axis=1)
    local_delta = np.concatenate(
        (
            np.mean(np.abs(values[:, 1] - values[:, 0]), axis=1),
            np.mean(np.abs(values[:, -1] - values[:, -2]), axis=1),
        )
    )
    seam_mae = float(np.mean(seam_delta))
    local_mae = float(np.mean(local_delta))
    return {
        "mae": seam_mae,
        "p95": float(np.percentile(seam_delta, 95)),
        "adjacent_column_mae": local_mae,
        "normalized_ratio": (
            0.0
            if seam_mae <= 1e-8 and local_mae <= 1e-8
            else seam_mae / max(local_mae, 1e-6)
        ),
    }


def _edge_map(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.clip(np.sqrt(dx**2 + dy**2) / 4.0, 0.0, 1.0)


def flow_transition_metrics(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    output_width: int = 256,
    fb_tolerance_pixels: float = 2.0,
) -> dict[str, float]:
    """Compute a deterministic circular-ERP optical-flow/edge diagnostic."""

    if previous.shape != current.shape:
        raise ValueError("temporal frames differ in shape")
    if np.array_equal(previous, current):
        return {
            "valid_fraction": 1.0,
            "warp_mae": 0.0,
            "warp_p95": 0.0,
            "edge_flicker": 0.0,
            "unwarped_mae": 0.0,
        }

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    height = max(32, int(round(previous.shape[0] * output_width / previous.shape[1])))
    size = (output_width, height)
    first = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
    second = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
    first_gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
    pad = max(16, output_width // 8)
    first_padded = cv2.copyMakeBorder(first_gray, 0, 0, pad, pad, cv2.BORDER_WRAP)
    second_padded = cv2.copyMakeBorder(second_gray, 0, 0, pad, pad, cv2.BORDER_WRAP)
    parameters = dict(
        pyr_scale=0.5,
        levels=4,
        winsize=21,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    backward_all = cv2.calcOpticalFlowFarneback(
        second_padded, first_padded, None, **parameters
    )
    forward_all = cv2.calcOpticalFlowFarneback(
        first_padded, second_padded, None, **parameters
    )
    backward = backward_all[:, pad : pad + output_width]
    grid_x, grid_y = np.meshgrid(
        np.arange(output_width, dtype=np.float32) + pad,
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + backward[:, :, 0]
    map_y = grid_y + backward[:, :, 1]
    warped_first = cv2.remap(
        cv2.copyMakeBorder(first, 0, 0, pad, pad, cv2.BORDER_WRAP),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    sampled_forward = cv2.remap(
        forward_all,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    fb_error = np.linalg.norm(backward + sampled_forward, axis=2)
    valid = (
        (map_x >= 0)
        & (map_x <= output_width + 2 * pad - 1)
        & (map_y >= 0)
        & (map_y <= height - 1)
        & (fb_error <= fb_tolerance_pixels)
    )
    unwarped_mae = float(
        np.mean(np.abs(first.astype(np.float32) - second) / 255.0)
    )
    if not np.any(valid):
        return {
            "valid_fraction": 0.0,
            "warp_mae": 1.0,
            "warp_p95": 1.0,
            "edge_flicker": 1.0,
            "unwarped_mae": unwarped_mae,
        }

    delta = np.mean(
        np.abs(warped_first.astype(np.float32) - second.astype(np.float32))
        / 255.0,
        axis=2,
    )
    warped_edges = _edge_map(warped_first)
    current_edges = _edge_map(second)
    return {
        "valid_fraction": float(np.mean(valid)),
        "warp_mae": float(np.mean(delta[valid])),
        "warp_p95": float(np.percentile(delta[valid], 95)),
        "edge_flicker": float(
            np.mean(np.abs(warped_edges[valid] - current_edges[valid]))
        ),
        "unwarped_mae": unwarped_mae,
    }
