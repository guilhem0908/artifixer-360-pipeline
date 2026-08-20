import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.materialize_spherical_snake_predictions import materialize


class MaterializeSphericalSnakePredictionsTest(unittest.TestCase):
    def test_materializes_sharded_outputs_in_one_contiguous_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for batch, indices in (("batch_0000", (0, 2)), ("batch_0001", (1, 3))):
                pred = root / "input" / batch / "pred"
                pred.mkdir(parents=True)
                for index in indices:
                    (pred / f"{index:05d}.png").write_bytes(str(index).encode())
            result = materialize(root / "input", root / "output", 4)
            self.assertEqual(result["count"], 4)
            self.assertEqual(
                [(root / "output" / f"{index:05d}.png").read_text() for index in range(4)],
                ["0", "1", "2", "3"],
            )

    def test_rejects_missing_prediction(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pred = root / "input" / "batch_0000" / "pred"
            pred.mkdir(parents=True)
            (pred / "00000.png").write_bytes(b"0")
            with self.assertRaisesRegex(ValueError, "missing"):
                materialize(root / "input", root / "output", 2)

    def test_hardlink_mode_materializes_container_safe_regular_files(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pred = root / "input" / "batch_0000" / "pred"
            pred.mkdir(parents=True)
            source = pred / "00000.png"
            source.write_bytes(b"frame")
            result = materialize(root / "input", root / "output", 1, symlink=False)
            destination = root / "output" / "00000.png"
            self.assertEqual(result["link_mode"], "hardlink")
            self.assertTrue(destination.is_file())
            self.assertFalse(destination.is_symlink())
            self.assertEqual(source.stat().st_ino, destination.stat().st_ino)


if __name__ == "__main__":
    unittest.main()
