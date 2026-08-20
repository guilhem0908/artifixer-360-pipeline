# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the ArtiFixer 360 research pipeline by Guilhem Carmouze, 2026.

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from model_eval.datasets import reconstructed_colmap_eval
from model_eval.datasets.reconstructed_colmap_eval import (
    ReconstructedColmapEvalDataset,
    resize_frames_to_spatial_shape,
)
from model_training.data.utils import NeighborSelectionMode, equirectangular_camera_rays_from_w2cs


def write_image(path: Path, size: tuple[int, int], mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size).save(path)


class ReconstructedColmapEvalDatasetTests(unittest.TestCase):
    def test_explicit_manifest_preserves_joint_order_and_output_ids(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for index in range(4):
            write_image(root / f"image_root/images/frame_{index:05d}.png", (32, 32))
        for index in (0, 1):
            write_image(root / f"renders/{index:05d}.png", (32, 32))
            write_image(root / f"opacity/{index:05d}.png", (32, 32), mode="L")
        transform = torch.eye(4).tolist()
        (root / "transforms.json").write_text(
            json.dumps(
                {
                    "w": 32,
                    "h": 32,
                    "fl_x": 20.0,
                    "fl_y": 20.0,
                    "cx": 16.0,
                    "cy": 16.0,
                    "frames": [
                        {"file_path": f"images/frame_{index:05d}.png", "transform_matrix": transform}
                        for index in range(4)
                    ],
                }
            )
        )
        (root / "selected_indices.json").write_text("[2, 3]")
        (root / "caption.h5").write_bytes(b"placeholder")
        (root / "split.json").write_text(
            json.dumps(
                {
                    "test": {
                        "scene": {
                            "transforms_path": "transforms.json",
                            "image_root": "image_root",
                            "render_dir": "renders",
                            "opacity_dir": "opacity",
                            "selected_indices_path": "selected_indices.json",
                            "prompt_path": "caption.h5",
                            "camera_scale": 1.0,
                            "has_gt": False,
                        }
                    }
                }
            )
        )
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "items": [
                        {
                            "target_indices": [1, 0],
                            "neighbor_indices": [2, 3],
                            "output_indices": [101, -1],
                            "global_frame_ids": [0, 1],
                            "item_noise_seed": 37,
                        }
                    ],
                }
            )
        )

        def fake_compute_camera_rays(*, frame_indices, image_shape, **kwargs):
            height, width = image_shape
            count = len(frame_indices)
            return {
                "camera_rays": torch.zeros(count, height, width, 6),
                "w2cs": torch.eye(4).repeat(count, 1, 1),
                "Ks": torch.eye(3).repeat(count, 1, 1),
                "neighbor_w2cs": torch.eye(4).repeat(2, 1, 1),
                "neighbor_Ks": torch.eye(3).repeat(2, 1, 1),
            }

        with (
            patch.object(
                reconstructed_colmap_eval,
                "load_encoded_prompt",
                return_value=(torch.zeros(1, 1, dtype=torch.bfloat16), ""),
            ),
            patch.object(reconstructed_colmap_eval, "compute_camera_rays", side_effect=fake_compute_camera_rays),
        ):
            dataset = ReconstructedColmapEvalDataset(
                split="test",
                split_path=root / "split.json",
                num_views=2,
                neighbor_selection_mode=NeighborSelectionMode.EVENLY_SPACED,
                max_test_frames=81,
                inference_manifest_path=root / "manifest.json",
            )
            item = dataset[0]

        self.assertEqual(item["frame_indices"].tolist(), [1, 0])
        self.assertEqual(item["neighbor_indices"].tolist(), [2, 3])
        self.assertEqual(item["output_indices"].tolist(), [101, -1])
        self.assertEqual(item["global_frame_ids"].tolist(), [0, 1])
        self.assertEqual(item["item_noise_seed"], 37)

    def test_explicit_causal_manifest_allows_missing_global_noise_ids(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for index in range(3):
            write_image(root / f"image_root/images/frame_{index:05d}.png", (32, 32))
        write_image(root / "renders/00000.png", (32, 32))
        write_image(root / "opacity/00000.png", (32, 32), mode="L")
        transform = torch.eye(4).tolist()
        (root / "transforms.json").write_text(json.dumps({
            "w": 32, "h": 32, "fl_x": 20.0, "fl_y": 20.0, "cx": 16.0, "cy": 16.0,
            "frames": [
                {"file_path": f"images/frame_{index:05d}.png", "transform_matrix": transform}
                for index in range(3)
            ],
        }))
        (root / "selected_indices.json").write_text("[1, 2]")
        (root / "caption.h5").write_bytes(b"placeholder")
        (root / "split.json").write_text(json.dumps({"test": {"scene": {
            "transforms_path": "transforms.json", "image_root": "image_root",
            "render_dir": "renders", "opacity_dir": "opacity",
            "selected_indices_path": "selected_indices.json", "prompt_path": "caption.h5",
            "camera_scale": 1.0, "has_gt": False,
        }}}))
        (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "items": [{
            "target_indices": [0], "neighbor_indices": [1, 2], "output_indices": [7],
            "item_noise_seed": 123,
        }]}))

        def fake_compute_camera_rays(*, frame_indices, image_shape, **kwargs):
            height, width = image_shape
            count = len(frame_indices)
            return {
                "camera_rays": torch.zeros(count, height, width, 6),
                "w2cs": torch.eye(4).repeat(count, 1, 1),
                "Ks": torch.eye(3).repeat(count, 1, 1),
                "neighbor_w2cs": torch.eye(4).repeat(2, 1, 1),
                "neighbor_Ks": torch.eye(3).repeat(2, 1, 1),
            }

        with (
            patch.object(reconstructed_colmap_eval, "load_encoded_prompt",
                         return_value=(torch.zeros(1, 1, dtype=torch.bfloat16), "")),
            patch.object(reconstructed_colmap_eval, "compute_camera_rays", side_effect=fake_compute_camera_rays),
        ):
            item = ReconstructedColmapEvalDataset(
                split="test", split_path=root / "split.json", num_views=2,
                neighbor_selection_mode=NeighborSelectionMode.EVENLY_SPACED,
                max_test_frames=81, inference_manifest_path=root / "manifest.json",
            )[0]
        self.assertNotIn("global_frame_ids", item)
        self.assertEqual(item["item_noise_seed"], 123)

    def test_camera_and_neighbor_tensors_use_render_resolution(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        image_root = root / "image_root"
        render_dir = root / "renders"
        opacity_dir = root / "opacity"

        write_image(image_root / "images/frame_00000.png", (32, 16))
        write_image(image_root / "images/frame_00001.png", (32, 16))
        write_image(render_dir / "00001.png", (64, 32))
        write_image(opacity_dir / "00001.png", (64, 32), mode="L")

        transform_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        (root / "transforms.json").write_text(
            json.dumps(
                {
                    "w": 64,
                    "h": 32,
                    "fl_x": 40.0,
                    "fl_y": 40.0,
                    "cx": 32.0,
                    "cy": 16.0,
                    "frames": [
                        {"file_path": "images/frame_00000.png", "transform_matrix": transform_matrix},
                        {"file_path": "images/frame_00001.png", "transform_matrix": transform_matrix},
                    ],
                }
            )
        )
        (root / "selected_indices.json").write_text("[0]")
        (root / "caption.h5").write_bytes(b"placeholder")
        (root / "split.json").write_text(
            json.dumps(
                {
                    "test": {
                        "scene": {
                            "transforms_path": "transforms.json",
                            "image_root": "image_root",
                            "render_dir": "renders",
                            "opacity_dir": "opacity",
                            "selected_indices_path": "selected_indices.json",
                            "prompt_path": "caption.h5",
                            "camera_scale": 1.0,
                        }
                    }
                }
            )
        )

        def fake_compute_camera_rays(*, image_shape: tuple[int, int], **kwargs):
            height, width = image_shape
            return {"camera_rays": torch.zeros(1, height, width, 6)}

        with (
            patch.object(
                reconstructed_colmap_eval,
                "load_encoded_prompt",
                return_value=(torch.zeros(1, 1, dtype=torch.bfloat16), ""),
            ),
            patch.object(reconstructed_colmap_eval, "compute_camera_rays", side_effect=fake_compute_camera_rays),
        ):
            dataset = ReconstructedColmapEvalDataset(
                split="test",
                split_path=root / "split.json",
                num_views=1,
                neighbor_selection_mode=NeighborSelectionMode.CONSECUTIVE,
                max_test_frames=1,
            )
            item = dataset[0]

        render_shape = tuple(item["rgb_rendered"].shape[-2:])
        self.assertEqual(render_shape, (32, 64))
        self.assertEqual(tuple(item["rgb_neighbors"].shape[-2:]), render_shape)
        self.assertEqual(tuple(item["camera_rays"].shape[-3:-1]), render_shape)


class ResizeFramesToSpatialShapeTests(unittest.TestCase):
    def test_resizes_neighbors_to_target_patch_grid(self) -> None:
        frames = torch.linspace(0.0, 1.0, 2 * 3 * 34 * 62).reshape(2, 3, 34, 62)

        resized = resize_frames_to_spatial_shape(frames, (48, 48))

        self.assertEqual(resized.shape, (2, 3, 48, 48))
        self.assertTrue(resized.is_contiguous())
        self.assertTrue(torch.isfinite(resized).all())
        self.assertGreaterEqual(float(resized.min()), 0.0)
        self.assertLessEqual(float(resized.max()), 1.0)

    def test_preserves_tensor_when_shape_already_matches(self) -> None:
        frames = torch.rand(2, 3, 48, 64)

        resized = resize_frames_to_spatial_shape(frames, (48, 64))

        self.assertIs(resized, frames)

    def test_rejects_non_positive_target_shape(self) -> None:
        frames = torch.rand(1, 3, 16, 16)

        with self.assertRaises(ValueError):
            resize_frames_to_spatial_shape(frames, (0, 16))


class EquirectangularCameraRayTests(unittest.TestCase):
    def test_identity_camera_produces_unit_spherical_directions(self) -> None:
        rays = equirectangular_camera_rays_from_w2cs(torch.eye(4).unsqueeze(0), 4, 8)

        self.assertEqual(tuple(rays.shape), (1, 4, 8, 6))
        moments = rays[..., :3]
        directions = rays[..., 3:]
        self.assertTrue(torch.allclose(moments, torch.zeros_like(moments)))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(directions, dim=-1), torch.ones(1, 4, 8)))
        self.assertLess(float(directions[0, 0, :, 1].max()), 0.0)
        self.assertGreater(float(directions[0, -1, :, 1].min()), 0.0)
        self.assertLess(float(directions[0, :, 0, 0].mean()), 0.0)
        self.assertGreater(float(directions[0, :, -1, 0].mean()), 0.0)

    def test_rejects_non_two_to_one_images(self) -> None:
        with self.assertRaises(ValueError):
            equirectangular_camera_rays_from_w2cs(torch.eye(4).unsqueeze(0), 8, 8)


if __name__ == "__main__":
    unittest.main()
