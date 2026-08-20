import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.generate_cubemap_trajectories import (
    FACE_ORDER,
    LOCAL_ROTATIONS,
    cubemap_intrinsics,
    generate_trajectories,
    write_cubemap_trajectories,
)


def pose(rotation: np.ndarray, center: tuple[float, float, float]) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = center
    return matrix.tolist()


class GenerateCubemapTrajectoriesTests(unittest.TestCase):
    def test_default_intrinsics_are_square_centered_and_fov_100(self) -> None:
        intrinsics = cubemap_intrinsics(1216, 100.0)

        expected_focal = 608.0 / math.tan(math.radians(50.0))
        self.assertEqual(intrinsics["w"], 1216)
        self.assertEqual(intrinsics["h"], 1216)
        self.assertAlmostEqual(intrinsics["fl_x"], expected_focal)
        self.assertEqual(intrinsics["fl_x"], intrinsics["fl_y"])
        self.assertEqual(intrinsics["cx"], 608.0)
        self.assertEqual(intrinsics["cy"], 608.0)

    def test_canonical_rotations_are_rigid_and_point_at_six_axes(self) -> None:
        expected_forward = {
            "F": (0.0, 0.0, -1.0),
            "R": (1.0, 0.0, 0.0),
            "B": (0.0, 0.0, 1.0),
            "L": (-1.0, 0.0, 0.0),
            "U": (0.0, 1.0, 0.0),
            "D": (0.0, -1.0, 0.0),
        }

        self.assertEqual(FACE_ORDER, ("F", "R", "B", "L", "U", "D"))
        for face in FACE_ORDER:
            rotation = LOCAL_ROTATIONS[face][:3, :3]
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3))
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
            np.testing.assert_allclose(-rotation[:, 2], expected_forward[face])

    def test_faces_keep_source_centers_and_combined_blocks_keep_frame_order(self) -> None:
        source_rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        centers = [(1.0, 2.0, 3.0), (-4.0, 5.0, 6.0), (7.0, 8.0, -9.0)]
        source = {
            "frames": [
                {"file_path": f"images/{index}.png", "transform_matrix": pose(source_rotation, center)}
                for index, center in enumerate(centers)
            ]
        }

        trajectories, combined = generate_trajectories(source)

        self.assertEqual(len(combined["frames"]), len(FACE_ORDER) * len(centers))
        for face_index, face in enumerate(FACE_ORDER):
            face_frames = trajectories[face]["frames"]
            combined_block = combined["frames"][
                face_index * len(centers) : (face_index + 1) * len(centers)
            ]
            self.assertEqual(combined_block, face_frames)
            for frame_index, frame in enumerate(face_frames):
                matrix = np.asarray(frame["transform_matrix"])
                np.testing.assert_allclose(matrix[:3, 3], centers[frame_index])
                np.testing.assert_allclose(
                    matrix[:3, :3], source_rotation @ LOCAL_ROTATIONS[face][:3, :3]
                )
                self.assertNotIn("file_path", frame)

    def test_applied_transform_is_materialized_before_face_rotation(self) -> None:
        applied_transform = np.eye(4, dtype=np.float64)
        applied_transform[:3, 3] = (10.0, 20.0, 30.0)
        source_pose = np.eye(4, dtype=np.float64)
        source_pose[:3, 3] = (11.0, 22.0, 33.0)
        source = {
            "applied_transform": applied_transform[:3].tolist(),
            "frames": [{"transform_matrix": source_pose.tolist()}],
        }

        trajectories, _ = generate_trajectories(source)

        for face in FACE_ORDER:
            face_pose = np.asarray(trajectories[face]["frames"][0]["transform_matrix"])
            np.testing.assert_allclose(face_pose[:3, 3], (1.0, 2.0, 3.0))

    def test_writer_creates_individual_combined_and_manifest_json(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        source_path = root / "source.json"
        output_dir = root / "cubemap"
        source_path.write_text(
            json.dumps(
                {
                    "frames": [
                        {"transform_matrix": pose(np.eye(3), (float(index), 0.0, 0.0))}
                        for index in range(2)
                    ]
                }
            )
        )

        manifest = write_cubemap_trajectories(source_path, output_dir)

        self.assertEqual(
            {path.name for path in output_dir.iterdir()},
            {"F.json", "R.json", "B.json", "L.json", "U.json", "D.json", "combined.json", "manifest.json"},
        )
        self.assertEqual(manifest["face_order"], list(FACE_ORDER))
        self.assertEqual(manifest["rotation_convention"], "opengl_face_c2w_postmultiply")
        self.assertEqual(manifest["frames_per_face"], 2)
        self.assertEqual(manifest["total_frames"], 12)
        self.assertEqual(manifest["fov"], 100.0)
        self.assertEqual(manifest["fov_degrees"], 100.0)
        self.assertEqual(
            [(block["face"], block["start"], block["stop"]) for block in manifest["blocks"]],
            [(face, index * 2, index * 2 + 2) for index, face in enumerate(FACE_ORDER)],
        )
        written_manifest = json.loads((output_dir / "manifest.json").read_text())
        self.assertEqual(written_manifest, manifest)
        self.assertEqual(len(json.loads((output_dir / "combined.json").read_text())["frames"]), 12)


if __name__ == "__main__":
    unittest.main()
