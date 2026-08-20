#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Keep reliable 3D renders and apply ArtiFixer only where opacity is uncertain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def composite(render: np.ndarray, prediction: np.ndarray, opacity: np.ndarray, low: float, high: float, blur: int) -> tuple[np.ndarray, np.ndarray]:
    confidence = opacity.astype(np.float32) / 255.0
    alpha = np.clip((high - confidence) / (high - low), 0.0, 1.0)
    if blur > 1:
        alpha = cv2.GaussianBlur(alpha, (blur, blur), 0)
    locked = render.astype(np.float32) * (1.0 - alpha[..., None]) + prediction.astype(np.float32) * alpha[..., None]
    return np.clip(np.rint(locked), 0, 255).astype(np.uint8), alpha


def apply_lock(render_dir: Path, opacity_dir: Path, prediction_dir: Path, output_dir: Path, low: float, high: float, blur: int) -> dict[str, object]:
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("require 0 <= low-opacity < high-opacity <= 1")
    if blur < 1 or blur % 2 == 0:
        raise ValueError("mask-blur must be a positive odd integer")
    if output_dir.exists():
        raise FileExistsError(output_dir)

    predictions = sorted(prediction_dir.glob("*.png"))
    if not predictions:
        raise ValueError(f"no PNG predictions in {prediction_dir}")
    output_dir.mkdir(parents=True)
    high_before: list[float] = []
    high_after: list[float] = []
    alpha_means: list[float] = []
    records = []
    for prediction_path in predictions:
        render_path = render_dir / prediction_path.name
        opacity_path = opacity_dir / prediction_path.name
        render = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
        prediction = cv2.imread(str(prediction_path), cv2.IMREAD_COLOR)
        opacity = cv2.imread(str(opacity_path), cv2.IMREAD_GRAYSCALE)
        if render is None or prediction is None or opacity is None:
            raise FileNotFoundError((render_path, prediction_path, opacity_path))
        if render.shape != prediction.shape or render.shape[:2] != opacity.shape:
            raise ValueError(f"shape mismatch for {prediction_path.name}")
        locked, alpha = composite(render, prediction, opacity, low, high, blur)
        output_path = output_dir / prediction_path.name
        if not cv2.imwrite(str(output_path), locked):
            raise RuntimeError(f"cannot write {output_path}")
        mask = opacity.astype(np.float32) / 255.0 >= 0.95
        before = np.mean(np.abs(prediction.astype(np.float32) - render.astype(np.float32)), axis=2) / 255.0
        after = np.mean(np.abs(locked.astype(np.float32) - render.astype(np.float32)), axis=2) / 255.0
        if np.any(mask):
            high_before.extend(before[mask].tolist())
            high_after.extend(after[mask].tolist())
        alpha_means.append(float(np.mean(alpha)))
        records.append({"name": prediction_path.name, "sha256": sha256(output_path)})

    if not high_after:
        raise ValueError("no pixels satisfy the high-opacity audit mask")
    report = {
        "schema": "artifixer.opacity_locked_predictions_v1",
        "verdict": "PASS",
        "frame_count": len(records),
        "low_opacity": low,
        "high_opacity": high,
        "mask_blur": blur,
        "mean_alpha": float(np.mean(alpha_means)),
        "high_opacity_change_before_mean": float(np.mean(high_before)),
        "high_opacity_change_after_mean": float(np.mean(high_after)),
        "high_opacity_change_after_p95": float(np.percentile(high_after, 95.0)),
        "frames": records,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--opacity-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--low-opacity", type=float, default=0.78)
    parser.add_argument("--high-opacity", type=float, default=0.98)
    parser.add_argument("--mask-blur", type=int, default=15)
    args = parser.parse_args()
    report = apply_lock(
        args.render_dir, args.opacity_dir, args.prediction_dir, args.output_dir,
        args.low_opacity, args.high_opacity, args.mask_blur,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, sort_keys=True))


if __name__ == "__main__":
    main()
