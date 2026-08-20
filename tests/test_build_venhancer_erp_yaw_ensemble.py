from __future__ import annotations

import numpy as np

from scripts.build_venhancer_erp_yaw_ensemble import (
    apply_seam_guard,
    direct_weight,
    evaluate_seams,
    fuse_frame,
)


def test_direct_weight_uses_shifted_yaw_at_seam_and_native_yaw_at_center() -> None:
    weight = direct_weight(1024).reshape(-1)
    assert weight[0] == 0.0
    assert weight[-1] == 0.0
    assert weight[512] == 1.0
    assert weight[256] == 1.0
    assert weight[768] == 1.0
    assert np.all((weight >= 0.0) & (weight <= 1.0))


def test_fuse_frame_selects_the_boundary_safe_owner() -> None:
    direct = np.full((4, 1024, 3), 40, dtype=np.uint8)
    shifted = np.full((4, 1024, 3), 220, dtype=np.uint8)
    fused = fuse_frame(direct, shifted)
    assert np.all(fused[:, 0] == 220)
    assert np.all(fused[:, -1] == 220)
    assert np.all(fused[:, 512] == 40)


def test_seam_gate_detects_catastrophic_boundary_regression() -> None:
    baseline = np.full((16, 32, 3), 100, dtype=np.uint8)
    good = baseline.copy()
    bad = baseline.copy()
    bad[:, 0] = 255
    bad[:, -1] = 0
    assert evaluate_seams([baseline], [good])["verdict"] == "PASS"
    report = evaluate_seams([baseline], [bad])
    assert report["verdict"] == "FAIL"
    assert report["checks"]["absolute_mae_all_frames"] is False


def test_seam_guard_preserves_exact_wrap_columns_and_leaves_center_enhanced() -> None:
    source = np.full((4, 64, 3), 25, dtype=np.uint8)
    enhanced = np.full((4, 64, 3), 225, dtype=np.uint8)
    guarded = apply_seam_guard(enhanced, source, 8)
    assert np.array_equal(guarded[:, 0], source[:, 0])
    assert np.array_equal(guarded[:, -1], source[:, -1])
    assert np.array_equal(guarded[:, 32], enhanced[:, 32])
    assert np.all(guarded[:, 4] > source[:, 4])
    assert np.all(guarded[:, 4] < enhanced[:, 4])
