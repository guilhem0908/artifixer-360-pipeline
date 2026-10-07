#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed temporal coherence gate for numbered ERP panorama frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Make `python scripts/<name>.py` work on a plain clone: the repository root must
# be importable for the `scripts.*` / `model_eval.*` imports below.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.panorama_metrics import flow_transition_metrics  # noqa: E402


def numbered_frames(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise ValueError(f"no PNG frames in {directory}")
    expected = [f"{index:05d}.png" for index in range(len(paths))]
    if [path.name for path in paths] != expected:
        raise ValueError(f"frames in {directory} are not one contiguous zero-based sequence")
    return paths


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def sequence_metrics(paths: list[Path], *, output_width: int) -> dict[str, object]:
    transitions = []
    previous = read_rgb(paths[0])
    for index, path in enumerate(paths[1:], start=1):
        current = read_rgb(path)
        metrics = flow_transition_metrics(previous, current, output_width=output_width)
        transitions.append({"from_index": index - 1, "to_index": index, **metrics})
        previous = current
    if not transitions:
        raise ValueError("temporal QC requires at least two frames")
    return {
        "transition_count": len(transitions),
        "valid_fraction_min": min(item["valid_fraction"] for item in transitions),
        "warp_mae_median": float(np.median([item["warp_mae"] for item in transitions])),
        "warp_mae_max": max(item["warp_mae"] for item in transitions),
        "warp_p95_max": max(item["warp_p95"] for item in transitions),
        "edge_flicker_median": float(np.median([item["edge_flicker"] for item in transitions])),
        "edge_flicker_max": max(item["edge_flicker"] for item in transitions),
        "transitions": transitions,
    }


def evaluate(
    *,
    input_dir: Path,
    prediction_dir: Path,
    output: Path,
    output_width: int = 256,
    minimum_valid_fraction: float = 0.40,
    maximum_warp_mae: float = 0.10,
    maximum_warp_p95: float = 0.30,
    maximum_edge_flicker: float = 0.12,
    maximum_regression_ratio: float = 1.25,
    regression_slack: float = 0.01,
    required_boundaries: tuple[int, ...] = (),
) -> dict[str, object]:
    input_paths = numbered_frames(input_dir)
    prediction_paths = numbered_frames(prediction_dir)
    if len(input_paths) != len(prediction_paths):
        raise ValueError("input and prediction frame counts differ")
    baseline = sequence_metrics(input_paths, output_width=output_width)
    prediction = sequence_metrics(prediction_paths, output_width=output_width)
    checks = {
        "valid_warp_coverage": prediction["valid_fraction_min"] >= minimum_valid_fraction,
        "absolute_warp_mae": prediction["warp_mae_max"] <= maximum_warp_mae,
        "absolute_warp_p95": prediction["warp_p95_max"] <= maximum_warp_p95,
        "absolute_edge_flicker": prediction["edge_flicker_max"] <= maximum_edge_flicker,
        "warp_mae_non_regression": prediction["warp_mae_median"]
        <= baseline["warp_mae_median"] * maximum_regression_ratio + regression_slack,
        "edge_flicker_non_regression": prediction["edge_flicker_median"]
        <= baseline["edge_flicker_median"] * maximum_regression_ratio + regression_slack,
    }
    prediction_by_stop = {item["to_index"]: item for item in prediction["transitions"]}
    boundary_reports = []
    for boundary in required_boundaries:
        if boundary not in prediction_by_stop:
            raise ValueError(f"required boundary {boundary} is not an evaluated transition")
        metrics = prediction_by_stop[boundary]
        gates = {
            "warp_mae": metrics["warp_mae"] <= maximum_warp_mae,
            "warp_p95": metrics["warp_p95"] <= maximum_warp_p95,
            "edge_flicker": metrics["edge_flicker"] <= maximum_edge_flicker,
            "valid_fraction": metrics["valid_fraction"] >= minimum_valid_fraction,
        }
        checks[f"required_boundary_{boundary}"] = all(gates.values())
        boundary_reports.append({"to_index": boundary, "metrics": metrics, "gates": gates})
    report: dict[str, object] = {
        "schema_version": 1,
        "method": "circular_erp_bidirectional_flow_consistency",
        "frame_count": len(input_paths),
        "thresholds": {
            "output_width": output_width,
            "minimum_valid_fraction": minimum_valid_fraction,
            "maximum_warp_mae": maximum_warp_mae,
            "maximum_warp_p95": maximum_warp_p95,
            "maximum_edge_flicker": maximum_edge_flicker,
            "maximum_regression_ratio": maximum_regression_ratio,
            "regression_slack": regression_slack,
        },
        "checks": checks,
        "input": baseline,
        "prediction": prediction,
        "required_boundaries": boundary_reports,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-width", type=int, default=256)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.40)
    parser.add_argument("--maximum-warp-mae", type=float, default=0.10)
    parser.add_argument("--maximum-warp-p95", type=float, default=0.30)
    parser.add_argument("--maximum-edge-flicker", type=float, default=0.12)
    parser.add_argument("--maximum-regression-ratio", type=float, default=1.25)
    parser.add_argument("--regression-slack", type=float, default=0.01)
    parser.add_argument("--required-boundary", type=int, action="append", default=[])
    args = parser.parse_args()
    try:
        report = evaluate(
            input_dir=args.input_dir,
            prediction_dir=args.prediction_dir,
            output=args.output,
            output_width=args.output_width,
            minimum_valid_fraction=args.minimum_valid_fraction,
            maximum_warp_mae=args.maximum_warp_mae,
            maximum_warp_p95=args.maximum_warp_p95,
            maximum_edge_flicker=args.maximum_edge_flicker,
            maximum_regression_ratio=args.maximum_regression_ratio,
            regression_slack=args.regression_slack,
            required_boundaries=tuple(args.required_boundary),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps({"verdict": report["verdict"], "checks": report["checks"]}, sort_keys=True))
    raise SystemExit(0 if report["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
