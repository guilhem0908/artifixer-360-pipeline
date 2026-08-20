import numpy as np

from scripts.panorama_metrics import erp_seam_metrics, flow_transition_metrics


def test_erp_seam_is_zero_for_wrapped_columns() -> None:
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    image[:, 0] = [30, 60, 90]
    image[:, -1] = image[:, 0]
    metrics = erp_seam_metrics(image)
    assert metrics["mae"] == 0.0
    assert metrics["p95"] == 0.0


def test_identical_frames_have_perfect_temporal_metrics() -> None:
    image = np.arange(32 * 64 * 3, dtype=np.uint8).reshape(32, 64, 3)
    metrics = flow_transition_metrics(image, image)
    assert metrics == {
        "valid_fraction": 1.0,
        "warp_mae": 0.0,
        "warp_p95": 0.0,
        "edge_flicker": 0.0,
        "unwarped_mae": 0.0,
    }


def test_temporal_metrics_reject_shape_mismatch() -> None:
    first = np.zeros((16, 32, 3), dtype=np.uint8)
    second = np.zeros((16, 31, 3), dtype=np.uint8)
    try:
        flow_transition_metrics(first, second)
    except ValueError as error:
        assert "differ in shape" in str(error)
    else:
        raise AssertionError("shape mismatch must fail")
