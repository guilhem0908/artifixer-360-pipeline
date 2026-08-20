import unittest

import cv2
import numpy as np

from scripts.evaluate_spherical_snake_turn import (
    rotational_overlap_metrics,
    source_coordinates_for_target,
)
from scripts.generate_nerfstudio14_trajectories import local_rotation


class SphericalSnakeTurnEvaluationTest(unittest.TestCase):
    def test_identity_rotation_has_zero_error(self):
        rng = np.random.default_rng(7)
        image = rng.random((48, 64, 3), dtype=np.float32)
        pose = np.eye(4)
        result = rotational_overlap_metrics(
            image,
            image.copy(),
            intrinsics={"fl_x": 40.0, "fl_y": 40.0, "cx": 31.5, "cy": 23.5},
            source_c2w=pose,
            target_c2w=pose,
        )
        self.assertAlmostEqual(result["mae"], 0.0, places=7)
        self.assertEqual(result["coverage"], 1.0)

    def test_known_rotation_remap_recovers_synthetic_target(self):
        height, width = 96, 128
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        source = np.stack(
            (u / width, v / height, (u + v) / (width + height)), axis=-1
        ).astype(np.float32)
        source_pose = np.eye(4)
        target_pose = local_rotation(10.0, 0.0)
        intrinsics = {"fl_x": 80.0, "fl_y": 80.0, "cx": 63.5, "cy": 47.5}
        map_x, map_y, valid = source_coordinates_for_target(
            height,
            width,
            source_c2w=source_pose,
            target_c2w=target_pose,
            **intrinsics,
        )
        target = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR)
        result = rotational_overlap_metrics(
            source,
            target,
            intrinsics=intrinsics,
            source_c2w=source_pose,
            target_c2w=target_pose,
        )
        self.assertGreater(np.mean(valid), 0.75)
        self.assertAlmostEqual(result["mae"], 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
