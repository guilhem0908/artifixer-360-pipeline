import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.extract_spherical_snake_lanes import extract, prediction_index
from scripts.generate_nerfstudio14_trajectories import VIEW_SPECS, generate as generate_rig
from scripts.generate_spherical_snake_trajectory import generate as generate_snake


class ExtractSphericalSnakeLanesTest(unittest.TestCase):
    def test_reorders_alternating_lanes_into_view_major_source_time(self):
        frames = []
        for index in range(3):
            pose = np.eye(4)
            pose[0, 3] = index
            frames.append({"transform_matrix": pose.tolist()})
        source = {"frames": frames}
        orientations = [(yaw, pitch) for _name, yaw, pitch, _group in VIEW_SPECS]
        names = [name for name, _yaw, _pitch, _group in VIEW_SPECS]
        trajectory, snake = generate_snake(
            source,
            reference_frame=0,
            start_frame=0,
            frame_count=3,
            orientations=orientations,
            orientation_names=names,
            max_turn_degrees=10.0,
        )
        _, _, rig = generate_rig(source, size=64, fov_degrees=110)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pred = root / "batch_0000" / "pred"
            pred.mkdir(parents=True)
            prediction_paths = {}
            for index in range(len(trajectory["frames"])):
                path = pred / f"{index:05d}.png"
                path.write_bytes(str(index).encode())
                prediction_paths[index] = path
            output = root / "view_major"
            result = extract(prediction_paths, snake, rig, output)
            self.assertEqual(result["output_count"], len(VIEW_SPECS) * 3)
            for view_index, (name, _, _, _) in enumerate(VIEW_SPECS):
                lane = next(item for item in snake["lanes"] if item["name"] == name)
                entries = snake["frames"][lane["start"] : lane["stop"]]
                by_source = {
                    item["source_index"]: item["target_index"] for item in entries
                }
                for source_index in range(3):
                    destination = output / f"{view_index * 3 + source_index:05d}.png"
                    self.assertEqual(destination.read_text(), str(by_source[source_index]))

    def test_recursive_prediction_index_rejects_duplicates(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for batch in ("batch_0000", "batch_0001"):
                pred = root / batch / "pred"
                pred.mkdir(parents=True)
                (pred / "00007.png").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                prediction_index(root)


if __name__ == "__main__":
    unittest.main()
