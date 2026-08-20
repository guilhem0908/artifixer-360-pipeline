#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate same-centre angular consistency across spherical-snake turns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def image_index(root: Path) -> dict[int, Path]:
    direct = sorted(root.glob("[0-9][0-9][0-9][0-9][0-9].png"))
    candidates = direct or sorted(root.glob("**/pred/[0-9][0-9][0-9][0-9][0-9].png"))
    result: dict[int, Path] = {}
    for path in candidates:
        index = int(path.stem)
        if index in result:
            raise ValueError(f"duplicate frame {index:05d}: {result[index]} and {path}")
        result[index] = path
    if not result:
        raise FileNotFoundError(f"no numbered PNG frames below {root}")
    return result


def source_coordinates_for_target(
    height: int,
    width: int,
    *,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map target pixels to a source image for a pure shared-centre rotation."""
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    target_rays = np.stack(
        ((u - cx) / fl_x, -(v - cy) / fl_y, -np.ones_like(u)),
        axis=-1,
    )
    world_rays = target_rays @ np.asarray(target_c2w, dtype=np.float64)[:3, :3].T
    source_rays = world_rays @ np.asarray(source_c2w, dtype=np.float64)[:3, :3]
    denominator = -source_rays[..., 2]
    map_x = cx + fl_x * source_rays[..., 0] / np.maximum(denominator, 1e-12)
    map_y = cy - fl_y * source_rays[..., 1] / np.maximum(denominator, 1e-12)
    valid = (
        (denominator > 1e-8)
        & (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )
    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def rotational_overlap_metrics(
    source_image: np.ndarray,
    target_image: np.ndarray,
    *,
    intrinsics: dict[str, float],
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
) -> dict[str, float]:
    if source_image.shape != target_image.shape:
        raise ValueError("source and target images must have identical shapes")
    height, width = target_image.shape[:2]
    map_x, map_y, valid = source_coordinates_for_target(
        height,
        width,
        fl_x=float(intrinsics["fl_x"]),
        fl_y=float(intrinsics["fl_y"]),
        cx=float(intrinsics["cx"]),
        cy=float(intrinsics["cy"]),
        source_c2w=source_c2w,
        target_c2w=target_c2w,
    )
    warped = cv2.remap(
        source_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    errors = np.mean(np.abs(warped - target_image), axis=2)[valid]
    if errors.size == 0:
        raise ValueError("the two views have no valid rotational overlap")
    return {
        "coverage": float(np.mean(valid)),
        "mae": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95.0)),
    }


def evaluate_directory(
    image_dir: Path,
    *,
    trajectory: dict,
    snake_manifest: dict,
) -> dict[str, object]:
    poses = [np.asarray(frame["transform_matrix"], dtype=np.float64) for frame in trajectory["frames"]]
    intrinsics = {
        name: float(trajectory[name])
        for name in ("fl_x", "fl_y", "cx", "cy")
    }
    images_by_index = image_index(image_dir)
    all_turns = []
    for turn in snake_manifest["turns"]:
        previous_lane = snake_manifest["lanes"][turn["from_lane"]]
        next_lane = snake_manifest["lanes"][turn["to_lane"]]
        indices = [previous_lane["stop"] - 1]
        indices.extend(range(turn["start"], turn["stop"]))
        indices.append(next_lane["start"])
        images = {}
        for index in indices:
            path = images_by_index.get(index)
            if path is None:
                raise FileNotFoundError(f"missing frame {index:05d} below {image_dir}")
            images[index] = load_rgb(path)
        adjacent = []
        for first, second in zip(indices, indices[1:]):
            metrics = rotational_overlap_metrics(
                images[first],
                images[second],
                intrinsics=intrinsics,
                source_c2w=poses[first],
                target_c2w=poses[second],
            )
            adjacent.append({"source": first, "target": second, **metrics})
        direct = rotational_overlap_metrics(
            images[indices[0]],
            images[indices[-1]],
            intrinsics=intrinsics,
            source_c2w=poses[indices[0]],
            target_c2w=poses[indices[-1]],
        )
        all_turns.append(
            {
                "from_lane": turn["from_lane"],
                "to_lane": turn["to_lane"],
                "indices": indices,
                "adjacent": adjacent,
                "direct": {"source": indices[0], "target": indices[-1], **direct},
            }
        )
    return {"directory": str(image_dir.resolve()), "turns": all_turns}


def summarize(evaluation: dict[str, object]) -> dict[str, float]:
    adjacent = [pair for turn in evaluation["turns"] for pair in turn["adjacent"]]
    direct = [turn["direct"] for turn in evaluation["turns"]]
    return {
        "adjacent_mae_mean": float(np.mean([pair["mae"] for pair in adjacent])),
        "adjacent_p95_max": float(np.max([pair["p95"] for pair in adjacent])),
        "direct_mae_mean": float(np.mean([pair["mae"] for pair in direct])),
        "direct_p95_max": float(np.max([pair["p95"] for pair in direct])),
        "direct_coverage_min": float(np.min([pair["coverage"] for pair in direct])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--snake-manifest", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mae", type=float, default=0.10)
    parser.add_argument("--max-p95", type=float, default=0.30)
    parser.add_argument("--min-direct-coverage", type=float, default=0.50)
    args = parser.parse_args()

    trajectory = json.loads(args.trajectory.read_text())
    snake_manifest = json.loads(args.snake_manifest.read_text())
    render = evaluate_directory(
        args.render_dir,
        trajectory=trajectory,
        snake_manifest=snake_manifest,
    )
    prediction = evaluate_directory(
        args.prediction_dir,
        trajectory=trajectory,
        snake_manifest=snake_manifest,
    )
    render_summary = summarize(render)
    prediction_summary = summarize(prediction)
    checks = {
        "adjacent_mae": prediction_summary["adjacent_mae_mean"] <= args.max_mae,
        "adjacent_p95": prediction_summary["adjacent_p95_max"] <= args.max_p95,
        "direct_mae": prediction_summary["direct_mae_mean"] <= args.max_mae,
        "direct_p95": prediction_summary["direct_p95_max"] <= args.max_p95,
        "direct_coverage": prediction_summary["direct_coverage_min"] >= args.min_direct_coverage,
    }
    report = {
        "schema": "artifixer.spherical_snake_turn_qc_v1",
        "thresholds": {
            "max_mae": args.max_mae,
            "max_p95": args.max_p95,
            "min_direct_coverage": args.min_direct_coverage,
        },
        "render": render,
        "render_summary": render_summary,
        "prediction": prediction,
        "prediction_summary": prediction_summary,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"verdict": report["verdict"], **prediction_summary}, sort_keys=True))
    raise SystemExit(0 if report["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
