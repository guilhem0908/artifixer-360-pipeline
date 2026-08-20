import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.evaluate_depth_synchronized_multiview import evaluate


def _rotation_y(degrees: float) -> list[list[float]]:
    radians = np.deg2rad(degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]


def test_evaluator_passes_identical_depth_consistent_pair(tmp_path: Path) -> None:
    manifest = tmp_path / "rig.json"
    windows = tmp_path / "windows.json"
    input_dir = tmp_path / "input"
    prediction_dir = tmp_path / "prediction"
    depth_dir = tmp_path / "depth"
    for directory in (input_dir, prediction_dir, depth_dir):
        directory.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "view_order": ["H000", "H300"],
                "frames_per_view": 2,
                "fov_degrees": 110.0,
                "local_rotations": {
                    "H000": _rotation_y(0.0),
                    "H300": _rotation_y(-60.0),
                },
            }
        )
    )
    windows.write_text(json.dumps({"published_source_indices": [0, 1]}))
    for index in range(4):
        # A constant image is exactly invariant under the perspective overlap warp.
        image = np.full((32, 32, 3), 96, dtype=np.uint8)
        Image.fromarray(image).save(input_dir / f"{index:05d}.png")
        Image.fromarray(image).save(prediction_dir / f"{index:05d}.png")
        np.save(depth_dir / f"{index:05d}.npy", np.ones((32, 32, 1), dtype=np.float32))

    report = evaluate(
        manifest_path=manifest,
        window_manifest_path=windows,
        input_dir=input_dir,
        prediction_dir=prediction_dir,
        depth_dir=depth_dir,
        output_path=tmp_path / "report.json",
        qc_size=32,
        depth_relative_tolerance=0.08,
        max_prediction_mae=0.01,
        max_prediction_p95=0.01,
        max_regression_ratio=1.0,
        regression_slack=0.0,
        minimum_coverage=0.002,
    )

    assert report["verdict"] == "PASS"
    assert report["checks"]["horizontal_loop_present"] is True
    assert report["prediction"]["mae"] == pytest.approx(0.0, abs=1e-7)
    assert report["prediction"]["p95_q95"] == pytest.approx(0.0, abs=1e-7)
