import json

import cv2
import numpy as np
import pytest

from scripts.generate_nerfstudio14_trajectories import generate
from scripts.stitch_artifixer_frustums360 import (
    FrustumProjector,
    Manifest,
    all_graphcut_labels,
    load_frame_indices,
    read_images,
    repair_unavailable_labels,
)


def test_nerfstudio14_frustums_cover_every_pixel_at_least_twice(tmp_path) -> None:
    source = {"frames": [{"transform_matrix": np.eye(4).tolist()}]}
    _, _, raw = generate(source, size=128, fov_degrees=110)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    manifest = Manifest.load(path)
    projector = FrustumProjector(manifest, 360, 180, feather_source_px=16)
    coverage = np.sum([item.valid for item in projector.maps.values()], axis=0)
    assert coverage.min() >= 2
    assert coverage.mean() > 3


def test_front_view_maps_to_panorama_centre(tmp_path) -> None:
    source = {"frames": [{"transform_matrix": np.eye(4).tolist()}]}
    _, _, raw = generate(source, size=128, fov_degrees=110)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    manifest = Manifest.load(path)
    projector = FrustumProjector(manifest, 360, 180, feather_source_px=16)
    mapping = projector.maps["H000"]
    assert abs(float(mapping.map_x[90, 180]) - 64) < 1
    assert abs(float(mapping.map_y[90, 180]) - 64) < 1


def test_hard_blend_uses_exact_central_owner(tmp_path) -> None:
    source = {"frames": [{"transform_matrix": np.eye(4).tolist()}]}
    _, _, raw = generate(source, size=128, fov_degrees=110)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    manifest = Manifest.load(path)
    projector = FrustumProjector(manifest, 360, 180, feather_source_px=16)
    images = {
        view: np.full((128, 128, 3), index, dtype=np.uint8)
        for index, view in enumerate(manifest.view_order)
    }
    gains = np.ones((len(manifest.view_order), 3), dtype=np.float32)
    panorama = projector.stitch(images, gains, bands=1, blend_mode="hard")
    assert np.array_equal(panorama[..., 0], projector.owner_labels)


def test_angular_ownership_uses_every_view(tmp_path) -> None:
    source = {"frames": [{"transform_matrix": np.eye(4).tolist()}]}
    _, _, raw = generate(source, size=128, fov_degrees=110)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    manifest = Manifest.load(path)
    projector = FrustumProjector(manifest, 720, 360, feather_source_px=16)

    counts = np.bincount(projector.owner_labels.ravel(), minlength=len(manifest.view_order))
    assert np.all(counts > 0)
    assert np.max(
        np.stack([projector.maps[view].axis_score for view in manifest.view_order]),
        axis=0,
    ).min() >= np.cos(np.deg2rad(55.0))


def test_load_frame_indices_preserves_qc_order(tmp_path) -> None:
    path = tmp_path / "frames.json"
    path.write_text(json.dumps({"frame_indices": [8, 2, 19]}))
    assert load_frame_indices(path, 20) == [8, 2, 19]
    assert load_frame_indices(None, 3) == [0, 1, 2]


def test_read_images_rejects_missing_by_default_and_can_skip_it(tmp_path) -> None:
    source = {"frames": [{"transform_matrix": np.eye(4).tolist()}]}
    _, _, raw = generate(source, size=32, fov_degrees=110)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    manifest = Manifest.load(path)
    for index in range(len(manifest.view_order) - 1):
        assert cv2.imwrite(
            str(tmp_path / f"{index:05d}.png"),
            np.zeros((32, 32, 3), np.uint8),
        )
    (tmp_path / f"{len(manifest.view_order) - 1:05d}.png").write_bytes(
        b"non-empty but truncated image"
    )

    with pytest.raises(FileNotFoundError, match="missing"):
        read_images(tmp_path, manifest, 0)
    images = read_images(tmp_path, manifest, 0, allow_missing=True)
    assert tuple(images) == manifest.view_order[:-1]


def test_all_graphcut_excludes_missing_view_and_keeps_global_labels(tmp_path) -> None:
    source = {"frames": [{"transform_matrix": np.eye(4).tolist()}]}
    _, _, raw = generate(source, size=32, fov_degrees=110)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    manifest = Manifest.load(path)
    projector = FrustumProjector(manifest, 64, 32, feather_source_px=4)
    missing = "D270"
    images = {
        view: np.full((32, 32, 3), index * 10, dtype=np.uint8)
        for index, view in enumerate(manifest.view_order)
        if view != missing
    }
    gains = np.ones((len(manifest.view_order), 3), dtype=np.float32)

    labels = all_graphcut_labels(projector, images, gains)
    active_indices = {manifest.view_order.index(view) for view in images}
    assert labels.shape == (32, 64)
    assert set(np.unique(labels)).issubset(active_indices)
    panorama = projector.stitch(images, gains, bands=1, blend_mode="hard")
    assert panorama.shape == (32, 64, 3)


def test_repair_unavailable_labels_removes_temporally_reintroduced_view(tmp_path) -> None:
    source = {"frames": [{"transform_matrix": np.eye(4).tolist()}]}
    _, _, raw = generate(source, size=32, fov_degrees=110)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    manifest = Manifest.load(path)
    projector = FrustumProjector(manifest, 64, 32, feather_source_px=4)
    missing = "D270"
    missing_index = manifest.view_order.index(missing)
    labels = projector.owner_labels.copy()
    labels[8:24, 16:48] = missing_index

    repaired = repair_unavailable_labels(
        projector,
        labels,
        [view for view in manifest.view_order if view != missing],
    )
    assert missing_index not in np.unique(repaired)
