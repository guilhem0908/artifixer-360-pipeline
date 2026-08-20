import unittest

import numpy as np

from scripts.generate_spherical_snake_inference_manifest import (
    build_manifest,
    translation_break_positions,
)
from scripts.generate_spherical_snake_trajectory import generate, rotation_distance_degrees


def source_with_moving_rotations(count: int = 40) -> dict:
    frames = []
    for index in range(count):
        angle = index * 0.08
        pose = np.eye(4)
        pose[:3, :3] = np.array(
            [
                [np.cos(angle), 0.0, -np.sin(angle)],
                [0.0, 1.0, 0.0],
                [np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        pose[:3, 3] = (index, index / 2.0, -index / 3.0)
        frames.append({"file_path": f"images/{index:05d}.png", "transform_matrix": pose.tolist()})
    return {"frames": frames}


class SphericalSnakeTrajectoryTest(unittest.TestCase):
    def test_two_lane_pilot_is_one_65_frame_smooth_world_locked_stream(self):
        trajectory, manifest = generate(
            source_with_moving_rotations(),
            reference_frame=0,
            start_frame=0,
            frame_count=30,
            orientations=[(0.0, 0.0), (30.0, 0.0)],
            max_turn_degrees=5.0,
        )
        self.assertEqual(len(trajectory["frames"]), 65)
        self.assertEqual(manifest["target_count"], 65)
        self.assertEqual(manifest["lanes"][0]["direction"], "forward")
        self.assertEqual(manifest["lanes"][1]["direction"], "reverse")
        self.assertEqual(manifest["turns"][0]["intermediate_count"], 5)
        self.assertTrue(all("file_path" not in frame for frame in trajectory["frames"]))

        poses = np.asarray([frame["transform_matrix"] for frame in trajectory["frames"]])
        source_centres = np.asarray(
            [frame["transform_matrix"] for frame in source_with_moving_rotations()["frames"]]
        )[:, :3, 3]
        np.testing.assert_allclose(poses[:30, :3, 3], source_centres[:30])
        np.testing.assert_allclose(poses[30:35, :3, 3], np.repeat(poses[29:30, :3, 3], 5, axis=0))
        np.testing.assert_allclose(poses[35:, :3, 3], source_centres[:30][::-1])
        np.testing.assert_allclose(poses[:30, :3, :3], np.repeat(poses[0:1, :3, :3], 30, axis=0))
        np.testing.assert_allclose(poses[35:, :3, :3], np.repeat(poses[35:36, :3, :3], 30, axis=0))
        angular_steps = [
            rotation_distance_degrees(first[:3, :3], second[:3, :3])
            for first, second in zip(poses, poses[1:])
        ]
        self.assertLessEqual(max(angular_steps), 5.0 + 1e-7)

    def test_source_segment_and_reference_are_validated(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            generate(
                source_with_moving_rotations(10),
                reference_frame=0,
                start_frame=5,
                frame_count=6,
                orientations=[(0.0, 0.0)],
                max_turn_degrees=5.0,
            )
        with self.assertRaisesRegex(ValueError, "reference_frame"):
            generate(
                source_with_moving_rotations(10),
                reference_frame=10,
                start_frame=0,
                frame_count=5,
                orientations=[(0.0, 0.0)],
                max_turn_degrees=5.0,
            )


class SphericalSnakeInferenceManifestTest(unittest.TestCase):
    def test_translation_break_detection_ignores_stationary_turns(self):
        centres = [0.0, 1.0, 2.0, 2.0, 3.0, 13.0, 14.0]
        frames = []
        for centre in centres:
            pose = np.eye(4)
            pose[0, 3] = centre
            frames.append({"transform_matrix": pose.tolist()})
        breaks, metrics = translation_break_positions({"frames": frames}, ratio=5.0)
        self.assertEqual(breaks, [5])
        self.assertEqual(metrics["median_positive_step"], 1.0)
        self.assertEqual(metrics["threshold"], 5.0)

    def test_single_65_target_pilot_is_one_phase_aligned_item(self):
        manifest = build_manifest(
            list(range(65)),
            list(range(65, 142)),
            context_views=12,
            core_size=65,
            halo_size=0,
        )
        self.assertEqual(len(manifest["items"]), 1)
        item = manifest["items"][0]
        self.assertEqual(item["target_indices"], list(range(65)))
        self.assertEqual(item["output_indices"], list(range(65)))
        self.assertEqual(item["global_frame_ids"], list(range(65)))
        self.assertEqual(len(item["neighbor_indices"]), 12)

    def test_long_stream_has_overlap_and_exact_core_coverage(self):
        manifest = build_manifest(
            list(range(310)),
            list(range(310, 387)),
            context_views=12,
            core_size=49,
            halo_size=12,
        )
        self.assertGreater(len(manifest["items"]), 1)
        self.assertLessEqual(
            max(len(item["target_indices"]) for item in manifest["items"]),
            77,
        )
        outputs = [
            value
            for item in manifest["items"]
            for value in item["output_indices"]
            if value != -1
        ]
        self.assertEqual(outputs, list(range(310)))
        self.assertTrue(
            all(item["global_frame_ids"][0] % 4 == 0 for item in manifest["items"])
        )

    def test_explicit_break_resets_phase_and_no_window_crosses_discontinuity(self):
        manifest = build_manifest(
            list(range(120)),
            list(range(120, 197)),
            context_views=12,
            core_size=49,
            halo_size=12,
            break_positions=[57],
        )
        self.assertEqual(manifest["blocks"], [{"start": 0, "stop": 57}, {"start": 57, "stop": 120}])
        for item in manifest["items"]:
            targets = item["target_indices"]
            self.assertFalse(min(targets) < 57 <= max(targets))
            self.assertEqual(item["global_frame_ids"][0] % 4, 0)
        outputs = [
            value
            for item in manifest["items"]
            for value in item["output_indices"]
            if value != -1
        ]
        self.assertEqual(outputs, list(range(120)))


if __name__ == "__main__":
    unittest.main()
