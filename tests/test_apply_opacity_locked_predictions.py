from pathlib import Path

import cv2
import numpy as np

from scripts.apply_opacity_locked_predictions import apply_lock


def write(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), value)


def test_lock_preserves_observed_pixels_and_repairs_uncertain_pixels(tmp_path: Path) -> None:
    render_dir = tmp_path / "render"
    opacity_dir = tmp_path / "opacity"
    prediction_dir = tmp_path / "prediction"
    output_dir = tmp_path / "locked"
    render = np.full((16, 16, 3), 20, np.uint8)
    prediction = np.full((16, 16, 3), 220, np.uint8)
    opacity = np.full((16, 16), 255, np.uint8)
    opacity[:, :6] = 0
    write(render_dir / "00003.png", render)
    write(prediction_dir / "00003.png", prediction)
    write(opacity_dir / "00003.png", opacity)

    report = apply_lock(render_dir, opacity_dir, prediction_dir, output_dir, 0.78, 0.98, 1)
    locked = cv2.imread(str(output_dir / "00003.png"))
    assert report["verdict"] == "PASS"
    assert report["frame_count"] == 1
    assert np.array_equal(locked[:, 8:], render[:, 8:])
    assert np.array_equal(locked[:, :6], prediction[:, :6])
    assert report["high_opacity_change_after_mean"] == 0.0


def test_lock_rejects_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    try:
        apply_lock(tmp_path, tmp_path, tmp_path, output, 0.78, 0.98, 15)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output must be rejected")
