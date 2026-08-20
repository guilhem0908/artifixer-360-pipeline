import unittest

import numpy as np

from scripts.generate_nerfstudio14_trajectories import VIEW_SPECS
from scripts.generate_world_locked14_trajectories import generate


class WorldLocked14TrajectoryTest(unittest.TestCase):
    def test_all_views_are_world_locked_and_preserve_centres(self):
        frames = []
        for index in range(5):
            angle = 0.35 * index
            pose = np.eye(4)
            pose[:3, :3] = np.array(
                [
                    [np.cos(angle), 0, -np.sin(angle)],
                    [0, 1, 0],
                    [np.sin(angle), 0, np.cos(angle)],
                ]
            )
            pose[:3, 3] = (index, 2 * index, -index)
            frames.append({"transform_matrix": pose.tolist()})
        trajectories, combined, manifest = generate({"frames": frames})

        self.assertTrue(manifest["world_locked"])
        self.assertEqual(manifest["reference_policy"], "first_frame")
        self.assertFalse(manifest["semantic_rules"])
        self.assertEqual(len(combined["frames"]), 5 * len(VIEW_SPECS))
        expected_centres = np.asarray([frame["transform_matrix"] for frame in frames])[:, :3, 3]
        for name, _, _, _ in VIEW_SPECS:
            poses = np.asarray([frame["transform_matrix"] for frame in trajectories[name]["frames"]])
            np.testing.assert_allclose(poses[:, :3, 3], expected_centres)
            np.testing.assert_allclose(
                poses[:, :3, :3], np.repeat(poses[0:1, :3, :3], len(poses), axis=0)
            )

    def test_all_views_share_the_same_centres_at_each_time(self):
        frames = []
        for index in range(3):
            pose = np.eye(4)
            pose[:3, 3] = (index, -index, index / 2)
            frames.append({"transform_matrix": pose.tolist()})
        trajectories, _, _ = generate({"frames": frames})
        for frame_index in range(3):
            centres = [
                np.asarray(trajectories[name]["frames"][frame_index]["transform_matrix"])[:3, 3]
                for name, _, _, _ in VIEW_SPECS
            ]
            np.testing.assert_allclose(centres, np.repeat([centres[0]], len(centres), axis=0))


if __name__ == "__main__":
    unittest.main()
