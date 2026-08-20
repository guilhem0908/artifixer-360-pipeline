from __future__ import annotations

import numpy as np

from scripts.reorient_erp_to_source_camera import (
    camera_to_reference_cv_rotation,
    erp_remap,
)


def test_identity_pose_preserves_erp_pixel_centres() -> None:
    pose = np.eye(4)
    rotation = camera_to_reference_cv_rotation(pose, pose)
    map_x, map_y = erp_remap(16, 8, rotation)
    expected_x, expected_y = np.meshgrid(np.arange(16), np.arange(8))
    np.testing.assert_allclose(map_x, expected_x, atol=1e-5)
    np.testing.assert_allclose(map_y, expected_y, atol=1e-5)


def test_output_centre_samples_current_camera_forward_in_reference_erp() -> None:
    reference = np.eye(4)
    angle = np.deg2rad(90.0)
    current = np.eye(4)
    current[:3, :3] = np.array(
        [[np.cos(angle), 0.0, -np.sin(angle)], [0.0, 1.0, 0.0], [np.sin(angle), 0.0, np.cos(angle)]]
    )
    rotation = camera_to_reference_cv_rotation(reference, current)
    map_x, _ = erp_remap(360, 180, rotation)
    # Positive OpenGL yaw-right places the current forward axis at +90 degrees
    # in the reference ERP, one quarter turn to the right of its centre.
    assert abs(float(map_x[90, 180]) - 269.5) < 0.6
