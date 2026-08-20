#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Fail closed when synchronized edits erase reliable structural detail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_gray(path: Path, size: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def load_opacity(path: Path, size: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return gx, gy, np.sqrt(gx * gx + gy * gy)


def evaluate(
    *,
    manifest_path: Path,
    window_manifest_path: Path,
    input_dir: Path,
    prediction_dir: Path,
    opacity_dir: Path,
    output_path: Path,
    size: int,
    opacity_threshold: float,
    minimum_input_gradient: float,
    minimum_edge_coverage: float,
    minimum_strength_ratio: float,
    minimum_orientation_cosine: float,
    minimum_per_view_strength_ratio: float | None = None,
) -> dict:
    rig = json.loads(manifest_path.read_text())
    windows = json.loads(window_manifest_path.read_text())
    view_count = len(rig["view_order"])
    frames_per_view = int(rig["frames_per_view"])
    frame_indices = [int(value) for value in windows["published_source_indices"]]
    ratios = []
    orientations = []
    selected_pixels = 0
    total_pixels = 0
    per_view = []
    for view_index, view_name in enumerate(rig["view_order"]):
        view_ratios = []
        view_orientations = []
        view_selected = 0
        for frame_index in frame_indices:
            global_index = view_index * frames_per_view + frame_index
            name = f"{global_index:05d}.png"
            source = load_gray(input_dir / name, size)
            prediction = load_gray(prediction_dir / name, size)
            opacity = load_opacity(opacity_dir / name, size)
            source_gx, source_gy, source_magnitude = gradients(source)
            prediction_gx, prediction_gy, prediction_magnitude = gradients(prediction)
            mask = (
                (opacity >= opacity_threshold)
                & (source >= 0.08)
                & (source <= 0.92)
                & (source_magnitude >= minimum_input_gradient)
            )
            if not np.any(mask):
                continue
            strength = prediction_magnitude[mask] / np.maximum(source_magnitude[mask], 1e-6)
            cosine = (
                source_gx[mask] * prediction_gx[mask]
                + source_gy[mask] * prediction_gy[mask]
            ) / np.maximum(source_magnitude[mask] * prediction_magnitude[mask], 1e-6)
            strength = np.clip(strength, 0.0, 2.0)
            cosine = np.clip(cosine, -1.0, 1.0)
            ratios.append(strength)
            orientations.append(cosine)
            view_ratios.append(strength)
            view_orientations.append(cosine)
            count = int(mask.sum())
            selected_pixels += count
            view_selected += count
            total_pixels += mask.size
        if view_selected:
            per_view.append(
                {
                    "view": str(view_name),
                    "selected_pixels": view_selected,
                    "strength_ratio_median": float(np.median(np.concatenate(view_ratios))),
                    "orientation_cosine_median": float(np.median(np.concatenate(view_orientations))),
                }
            )
    if not ratios or total_pixels == 0:
        raise ValueError("no high-confidence structural edges were found")
    all_ratios = np.concatenate(ratios)
    all_orientations = np.concatenate(orientations)
    metrics = {
        "edge_coverage": selected_pixels / total_pixels,
        "strength_ratio_median": float(np.median(all_ratios)),
        "strength_ratio_p10": float(np.quantile(all_ratios, 0.10)),
        "strength_ratio_per_view_min": min(
            view["strength_ratio_median"] for view in per_view
        ),
        "orientation_cosine_median": float(np.median(all_orientations)),
        "selected_pixels": selected_pixels,
    }
    checks = {
        "edge_coverage": metrics["edge_coverage"] >= minimum_edge_coverage,
        "edge_strength_preserved": metrics["strength_ratio_median"] >= minimum_strength_ratio,
        "edge_orientation_preserved": metrics["orientation_cosine_median"] >= minimum_orientation_cosine,
    }
    if minimum_per_view_strength_ratio is not None:
        checks["per_view_edge_strength_preserved"] = (
            metrics["strength_ratio_per_view_min"] >= minimum_per_view_strength_ratio
        )
    report = {
        "schema_version": 1,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "frame_indices": frame_indices,
        "thresholds": {
            "opacity_threshold": opacity_threshold,
            "minimum_input_gradient": minimum_input_gradient,
            "minimum_edge_coverage": minimum_edge_coverage,
            "minimum_strength_ratio": minimum_strength_ratio,
            "minimum_per_view_strength_ratio": minimum_per_view_strength_ratio,
            "minimum_orientation_cosine": minimum_orientation_cosine,
        },
        "checks": checks,
        "metrics": metrics,
        "per_view": per_view,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--opacity-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--opacity-threshold", type=float, default=0.90)
    parser.add_argument("--minimum-input-gradient", type=float, default=0.03)
    parser.add_argument("--minimum-edge-coverage", type=float, default=0.005)
    parser.add_argument("--minimum-strength-ratio", type=float, default=0.55)
    parser.add_argument("--minimum-per-view-strength-ratio", type=float)
    parser.add_argument("--minimum-orientation-cosine", type=float, default=0.75)
    args = parser.parse_args()
    report = evaluate(
        manifest_path=args.manifest,
        window_manifest_path=args.window_manifest,
        input_dir=args.input_dir,
        prediction_dir=args.prediction_dir,
        opacity_dir=args.opacity_dir,
        output_path=args.output,
        size=args.size,
        opacity_threshold=args.opacity_threshold,
        minimum_input_gradient=args.minimum_input_gradient,
        minimum_edge_coverage=args.minimum_edge_coverage,
        minimum_strength_ratio=args.minimum_strength_ratio,
        minimum_orientation_cosine=args.minimum_orientation_cosine,
        minimum_per_view_strength_ratio=args.minimum_per_view_strength_ratio,
    )
    print(json.dumps({"verdict": report["verdict"], "checks": report["checks"], "metrics": report["metrics"]}))
    raise SystemExit(0 if report["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
