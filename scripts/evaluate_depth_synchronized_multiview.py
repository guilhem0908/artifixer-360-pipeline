#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate depth-valid overlap consistency for a synchronized multi-view run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Make `python scripts/<name>.py` work on a plain clone: the repository root must
# be importable for the `scripts.*` / `model_eval.*` imports below.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model_eval.synchronized_multiview import (  # noqa: E402
    build_depth_aware_latent_reprojection_graph,
    validate_reprojection_graph,
    warp_video_latents,
)


def load_rgb_video(
    directory: Path,
    *,
    view_count: int,
    frames_per_view: int,
    frame_indices: list[int],
    size: int,
) -> torch.Tensor:
    videos = []
    for view_index in range(view_count):
        frames = []
        for frame_index in frame_indices:
            path = directory / f"{view_index * frames_per_view + frame_index:05d}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
            frames.append(torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1))
        videos.append(torch.stack(frames, dim=1))
    return torch.stack(videos)


def load_depth_video(
    directory: Path,
    *,
    view_count: int,
    frames_per_view: int,
    frame_indices: list[int],
    size: int,
) -> torch.Tensor:
    videos = []
    for view_index in range(view_count):
        frames = []
        for frame_index in frame_indices:
            path = directory / f"{view_index * frames_per_view + frame_index:05d}.npy"
            if not path.is_file():
                raise FileNotFoundError(path)
            value = np.load(path, allow_pickle=False)
            if value.ndim == 3 and value.shape[-1] == 1:
                value = value[..., 0]
            if value.ndim != 2 or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"invalid radial depth: {path}")
            tensor = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))[None, None]
            frames.append(F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[0, 0])
        videos.append(torch.stack(frames))
    return torch.stack(videos)


def consistency_metrics(
    videos: torch.Tensor,
    graph,
    view_names: list[str],
) -> dict:
    edges = []
    weighted_sum = 0.0
    total_pixels = 0
    for edge in graph:
        warped = warp_video_latents(videos[edge.source], edge.grid)
        mask = edge.valid.to(torch.bool)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(videos.shape[2], -1, -1)
        difference = (videos[edge.target] - warped).abs().mean(dim=0)
        values = difference[mask]
        if not values.numel():
            raise ValueError(f"empty evaluated edge {edge.source}->{edge.target}")
        mean = float(values.mean().item())
        p95 = float(torch.quantile(values, 0.95).item())
        count = int(values.numel())
        weighted_sum += mean * count
        total_pixels += count
        edges.append(
            {
                "source": view_names[edge.source],
                "target": view_names[edge.target],
                "coverage": float(mask.float().mean().item()),
                "mae": mean,
                "p95": p95,
            }
        )
    return {
        "mae": weighted_sum / max(total_pixels, 1),
        "p95_max": max(edge["p95"] for edge in edges),
        "p95_q95": float(np.quantile([edge["p95"] for edge in edges], 0.95)),
        "coverage_min": min(edge["coverage"] for edge in edges),
        "edges": edges,
    }


def evaluate(
    *,
    manifest_path: Path,
    window_manifest_path: Path,
    input_dir: Path,
    prediction_dir: Path,
    depth_dir: Path,
    output_path: Path,
    qc_size: int,
    depth_relative_tolerance: float,
    max_prediction_mae: float,
    max_prediction_p95: float,
    max_regression_ratio: float,
    regression_slack: float,
    minimum_coverage: float,
    max_prediction_p95_q95: float | None = None,
) -> dict:
    rig = json.loads(manifest_path.read_text())
    windows = json.loads(window_manifest_path.read_text())
    view_names = [str(value) for value in rig["view_order"]]
    frame_count = int(rig["frames_per_view"])
    frame_indices = [int(value) for value in windows["published_source_indices"]]
    rotations = [rig["local_rotations"][name] for name in view_names]
    depths = load_depth_video(
        depth_dir,
        view_count=len(view_names),
        frames_per_view=frame_count,
        frame_indices=frame_indices,
        size=qc_size,
    )
    graph = build_depth_aware_latent_reprojection_graph(
        rotations,
        depths,
        qc_size,
        qc_size,
        float(rig["fov_degrees"]),
        device=torch.device("cpu"),
        dtype=torch.float32,
        vae_temporal_scale=1,
        depth_relative_tolerance=depth_relative_tolerance,
        minimum_depth_overlap_fraction=minimum_coverage,
        spatial_valid_fraction=0.5,
        temporal_valid_fraction=0.5,
    )
    graph_summary = validate_reprojection_graph(
        graph,
        view_names,
        required_bidirectional_pairs=[("H000", "H300")],
    )
    inputs = load_rgb_video(
        input_dir,
        view_count=len(view_names),
        frames_per_view=frame_count,
        frame_indices=frame_indices,
        size=qc_size,
    )
    predictions = load_rgb_video(
        prediction_dir,
        view_count=len(view_names),
        frames_per_view=frame_count,
        frame_indices=frame_indices,
        size=qc_size,
    )
    input_metrics = consistency_metrics(inputs, graph, view_names)
    prediction_metrics = consistency_metrics(predictions, graph, view_names)
    checks = {
        "prediction_mae": prediction_metrics["mae"] <= max_prediction_mae,
        "prediction_p95": prediction_metrics["p95_max"] <= max_prediction_p95,
        "coverage": prediction_metrics["coverage_min"] >= minimum_coverage,
        "no_overlap_regression": prediction_metrics["mae"]
        <= input_metrics["mae"] * max_regression_ratio + regression_slack,
        "horizontal_loop_present": True,
    }
    if max_prediction_p95_q95 is not None:
        checks["prediction_p95_q95"] = (
            prediction_metrics["p95_q95"] <= max_prediction_p95_q95
        )
    report = {
        "schema_version": 1,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "frame_indices": frame_indices,
        "view_order": view_names,
        "graph": graph_summary,
        "thresholds": {
            "max_prediction_mae": max_prediction_mae,
            "max_prediction_p95": max_prediction_p95,
            "max_prediction_p95_q95": max_prediction_p95_q95,
            "max_regression_ratio": max_regression_ratio,
            "regression_slack": regression_slack,
            "minimum_coverage": minimum_coverage,
            "depth_relative_tolerance": depth_relative_tolerance,
        },
        "checks": checks,
        "input": input_metrics,
        "prediction": prediction_metrics,
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
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qc-size", type=int, default=256)
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.08)
    parser.add_argument("--max-prediction-mae", type=float, default=0.10)
    parser.add_argument("--max-prediction-p95", type=float, default=0.30)
    parser.add_argument("--max-prediction-p95-q95", type=float)
    parser.add_argument("--max-regression-ratio", type=float, default=1.15)
    parser.add_argument("--regression-slack", type=float, default=0.005)
    parser.add_argument("--minimum-coverage", type=float, default=0.002)
    args = parser.parse_args()
    if args.qc_size <= 0:
        raise SystemExit("--qc-size must be positive")
    report = evaluate(
        manifest_path=args.manifest,
        window_manifest_path=args.window_manifest,
        input_dir=args.input_dir,
        prediction_dir=args.prediction_dir,
        depth_dir=args.depth_dir,
        output_path=args.output,
        qc_size=args.qc_size,
        depth_relative_tolerance=args.depth_relative_tolerance,
        max_prediction_mae=args.max_prediction_mae,
        max_prediction_p95=args.max_prediction_p95,
        max_regression_ratio=args.max_regression_ratio,
        regression_slack=args.regression_slack,
        minimum_coverage=args.minimum_coverage,
        max_prediction_p95_q95=args.max_prediction_p95_q95,
    )
    print(json.dumps({"verdict": report["verdict"], "checks": report["checks"]}, sort_keys=True))
    raise SystemExit(0 if report["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
