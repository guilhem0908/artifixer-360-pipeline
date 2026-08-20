import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.evaluate_high_confidence_detail import evaluate


def _write_fixture(root: Path, *, blurred: bool) -> tuple[Path, Path, Path, Path, Path]:
    manifest = root / "manifest.json"
    windows = root / "windows.json"
    inputs = root / "inputs"
    predictions = root / "predictions"
    opacity = root / "opacity"
    for directory in (inputs, predictions, opacity):
        directory.mkdir()
    manifest.write_text(json.dumps({"view_order": ["A"], "frames_per_view": 1}))
    windows.write_text(json.dumps({"published_source_indices": [0]}))
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, 32:] = 180
    prediction = cv2.GaussianBlur(image, (15, 15), 0) if blurred else image.copy()
    Image.fromarray(image).save(inputs / "00000.png")
    Image.fromarray(prediction).save(predictions / "00000.png")
    Image.fromarray(np.full((64, 64), 255, dtype=np.uint8)).save(opacity / "00000.png")
    return manifest, windows, inputs, predictions, opacity


def test_detail_gate_accepts_preserved_edge_and_rejects_blur(tmp_path: Path) -> None:
    for blurred, expected in ((False, "PASS"), (True, "FAIL")):
        root = tmp_path / str(blurred)
        root.mkdir()
        manifest, windows, inputs, predictions, opacity = _write_fixture(root, blurred=blurred)
        report = evaluate(
            manifest_path=manifest,
            window_manifest_path=windows,
            input_dir=inputs,
            prediction_dir=predictions,
            opacity_dir=opacity,
            output_path=root / "report.json",
            size=64,
            opacity_threshold=0.9,
            minimum_input_gradient=0.03,
            minimum_edge_coverage=0.0001,
            minimum_strength_ratio=0.55,
            minimum_orientation_cosine=0.75,
            minimum_per_view_strength_ratio=0.55,
        )
        assert report["verdict"] == expected
        assert "per_view_edge_strength_preserved" in report["checks"]
