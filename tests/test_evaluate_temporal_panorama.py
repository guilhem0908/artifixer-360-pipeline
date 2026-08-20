from pathlib import Path

import cv2
import numpy as np

from scripts.evaluate_temporal_panorama import evaluate


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def moving_sequence(directory: Path, *, flicker: bool = False) -> None:
    rng = np.random.default_rng(7)
    for index in range(5):
        image = np.zeros((64, 128, 3), dtype=np.uint8)
        image[:, :, 0] = np.linspace(20, 180, 128, dtype=np.uint8)[None]
        image[20:44, 15 + index : 45 + index] = [220, 180, 80]
        if flicker and index % 2:
            image = rng.integers(0, 256, image.shape, dtype=np.uint8)
        write_rgb(directory / f"{index:05d}.png", image)


def test_identical_temporal_sequence_passes(tmp_path: Path):
    input_dir = tmp_path / "input"
    prediction_dir = tmp_path / "prediction"
    moving_sequence(input_dir)
    moving_sequence(prediction_dir)
    report = evaluate(
        input_dir=input_dir,
        prediction_dir=prediction_dir,
        output=tmp_path / "report.json",
        output_width=64,
    )
    assert report["verdict"] == "PASS"
    assert all(report["checks"].values())


def test_framewise_flicker_fails_absolute_or_regression_gate(tmp_path: Path):
    input_dir = tmp_path / "input"
    prediction_dir = tmp_path / "prediction"
    moving_sequence(input_dir)
    moving_sequence(prediction_dir, flicker=True)
    report = evaluate(
        input_dir=input_dir,
        prediction_dir=prediction_dir,
        output=tmp_path / "report.json",
        output_width=64,
    )
    assert report["verdict"] == "FAIL"
    assert not all(report["checks"].values())


def test_strict_relative_gate_requires_real_median_improvement(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    moving_sequence(baseline_dir)
    moving_sequence(candidate_dir)
    report = evaluate(
        input_dir=baseline_dir,
        prediction_dir=candidate_dir,
        output=tmp_path / "strict_report.json",
        output_width=64,
        maximum_regression_ratio=0.98,
        regression_slack=0.0,
    )
    assert report["verdict"] == "FAIL"
    assert report["checks"]["warp_mae_non_regression"] is False
    assert report["checks"]["edge_flicker_non_regression"] is False
