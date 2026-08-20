#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the ArtiFixer 360 research pipeline by Guilhem Carmouze, 2026.

"""ArtiFixer3D utilities for distilling corrected views back into 3DGRUT."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation
from threedgrut.datasets.utils import CAMERA_MODEL_NAMES, Camera
from threedgrut.datasets.utils import Image as ColmapImage
from threedgrut.datasets.utils import read_colmap_extrinsics_binary, read_colmap_intrinsics_binary

import data_processing.threedgrut_training as threedgrut_training
from data_processing.camera_trajectories import (
    applied_transform_matrix,
    camera_intrinsics_for_frame,
    opengl_c2w_to_opencv_w2c,
    trajectory_frame_count,
    write_json,
)
from data_processing.render_3dgrut_colmap import (
    render_3dgrut_colmap,
    render_outputs_complete,
    selected_indices_for_render,
)

ARTIFIXER3D_EXPERIMENT = "artifixer3d"

# Canonical AF3D split pose contract:
#
# Every prepared reconstructed_colmap split stores transforms.json
# `transform_matrix` values as NeRFStudio/OpenGL camera-to-world poses. That is
# the convention produced by prepare_colmap_artifixer_inputs.py through
# opencv_w2c_to_opengl_c2w(). AF3D always materializes generated frames back to
# COLMAP/3DGRUT OpenCV world-to-camera by applying the OpenGL->OpenCV camera
# flip in opengl_c2w_to_opencv_w2c().
POSE_VALIDATION_ROTATION_TOLERANCE_DEGREES = 1e-3
POSE_VALIDATION_TRANSLATION_TOLERANCE = 1e-4


@dataclass(frozen=True)
class PreparedScene:
    # Key from split.json; reused unchanged in ArtiFixer3D+ inference metadata.
    scene_id: str
    # Prepared scene directory; used for default ArtiFixer3D output paths.
    scene_root: Path
    # Source transforms.json defining frame order, intrinsics, and generated-frame poses.
    transforms_path: Path
    # Prepared COLMAP root containing real anchor images and sparse/0.
    colmap_dir: Path
    # Caption embedding file carried into ArtiFixer3D+ inference metadata.
    prompt_path: Path
    # Metric scale expected by inference; copied from the prepared split.
    camera_scale: float
    # Preserve generated-only metadata as has_gt=False for ArtiFixer3D+ inference.
    has_gt: bool
    # Frame indices that keep real COLMAP images; the complement uses ArtiFixer PNGs.
    selected_indices: list[int]
    # Optional target filter passed through so ArtiFixer3D+ uses the same target frames.
    target_indices_path: Path | None
    # Prepared 3DGRUT checkpoint used as the default ArtiFixer3D resume point.
    reconstruction_checkpoint: Path | None
    # Total transforms frames; used to materialize every COLMAP entry and validate render completeness.
    frame_count: int


@dataclass(frozen=True)
class Artifixer3DPaths:
    """Generated paths for this ArtiFixer3D run; PreparedScene stores existing inputs."""

    # 3DGRUT training output root created by this run and passed as out_dir.
    run_root: Path
    # New COLMAP scene built from real anchors and ArtiFixer predictions, distinct from scene.colmap_dir.
    distillation_input_dir: Path
    # File where scene.selected_indices is written for 3DGRUT training and render validation.
    distillation_selected_indices_path: Path
    # Prediction RGB subdirectory inside distillation_input_dir referenced by image_path_override.
    override_image_dir: Path
    # Render output root created by this run and passed to render_3dgrut_colmap.
    render_output_root: Path
    # Concrete ours_STEP render directory used for reuse checks and the ArtiFixer3D+ split.
    render_checkpoint_dir: Path
    # reconstructed_colmap split consumed by the ArtiFixer3D+ inference pass.
    artifixer3d_plus_inference_split_path: Path


def resolve_split_path(split_root: Path, value: str) -> Path:
    """Resolve split metadata paths relative to the split file so splits stay relocatable."""
    path = Path(value)
    return path if path.is_absolute() else split_root / path


def resolve_metadata_path(split_root: Path, metadata: dict[str, object], field: str) -> Path:
    """Read a required path field from split metadata and fail before later phases guess."""
    assert field in metadata, f"Prepared scene is missing required field {field!r}"
    value = metadata[field]
    assert isinstance(value, str), f"Prepared scene field {field!r} must be a string path, got {value!r}"
    return resolve_split_path(split_root, value)


def resolve_metadata_number(metadata: dict[str, object], field: str) -> float:
    """Read required numeric metadata from prepared split metadata."""
    assert field in metadata, f"Prepared scene is missing required field {field!r}"
    value = metadata[field]
    assert isinstance(value, int | float) and not isinstance(
        value, bool
    ), f"Prepared scene field {field!r} must be a number, got {value!r}"
    return float(value)


def path_for_split(split_path: Path, path: Path) -> str:
    """Store paths relative to the split when possible so outputs move with the scene root."""
    split_root = split_path.parent.resolve()
    path = path.resolve()
    try:
        return path.relative_to(split_root).as_posix()
    except ValueError:
        return str(path)


def load_json_list(path: Path) -> list[int]:
    """Read selected-index JSON files and assert they contain integer frame ids."""
    values = json.loads(path.read_text())
    assert isinstance(values, list), f"{path} must contain a JSON list"
    assert all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    ), f"{path} must contain integer frame indices"
    return values


def read_frame_count(transforms_path: Path) -> int:
    """Derive frame count from transforms.json, the source of frame order for distillation."""
    transforms = json.loads(transforms_path.read_text())
    frames = transforms["frames"]
    assert isinstance(frames, list), f"{transforms_path} must contain a frames list"
    return len(frames)


def load_prepared_scene(scene_root: Path, split_path: Path | None, scene_id: str | None) -> PreparedScene:
    """Resolve and validate the prepared split into the contract ArtiFixer3D consumes."""
    scene_root = scene_root.resolve()
    split_path = (split_path or scene_root / "split.json").resolve()
    split_data = json.loads(split_path.read_text())["test"]
    assert isinstance(split_data, dict), f"{split_path} test split must be a mapping"

    if scene_id is None:
        assert len(split_data) == 1, f"--scene_id is required when {split_path} contains {len(split_data)} scenes"
        scene_id = next(iter(split_data))

    metadata = split_data[scene_id]
    assert isinstance(metadata, dict), f"Scene {scene_id!r} metadata must be a mapping"
    split_root = split_path.parent

    transforms_path = resolve_metadata_path(split_root, metadata, "transforms_path")
    colmap_dir = resolve_metadata_path(split_root, metadata, "image_root")
    selected_indices_path = resolve_metadata_path(split_root, metadata, "selected_indices_path")
    target_indices_path = (
        resolve_metadata_path(split_root, metadata, "target_indices_path")
        if "target_indices_path" in metadata
        else None
    )
    prompt_path = resolve_metadata_path(split_root, metadata, "prompt_path")
    reconstruction_checkpoint = (
        resolve_metadata_path(split_root, metadata, "reconstruction_checkpoint")
        if "reconstruction_checkpoint" in metadata
        else None
    )
    has_gt = metadata.get("has_gt", True)
    assert isinstance(has_gt, bool), f"Scene {scene_id!r} has non-boolean has_gt metadata: {has_gt!r}"

    selected_indices = load_json_list(selected_indices_path)
    frame_count = read_frame_count(transforms_path)
    camera_scale = resolve_metadata_number(metadata, "camera_scale")

    required_paths = (transforms_path, colmap_dir, selected_indices_path, prompt_path)
    if target_indices_path is not None:
        required_paths = (*required_paths, target_indices_path)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Prepared scene {scene_id!r} references missing path: {path}")

    return PreparedScene(
        scene_id=scene_id,
        scene_root=scene_root,
        transforms_path=transforms_path,
        colmap_dir=colmap_dir,
        prompt_path=prompt_path,
        camera_scale=camera_scale,
        has_gt=has_gt,
        selected_indices=selected_indices,
        target_indices_path=target_indices_path,
        reconstruction_checkpoint=reconstruction_checkpoint,
        frame_count=frame_count,
    )


def artifixer3d_paths(
    scene: PreparedScene, output_root: Path | None, artifixer3d_plus_inference_split_path: Path | None, steps: int
) -> Artifixer3DPaths:
    """Centralize output layout so all ArtiFixer3D phases agree on paths."""
    output_root = (output_root or scene.scene_root / "artifixer3d").resolve()
    output_name = scene.scene_id
    artifixer3d_plus_inference_split_path = (
        artifixer3d_plus_inference_split_path or scene.scene_root / "split_artifixer3d_plus.json"
    ).resolve()
    distillation_input_dir = output_root / "distillation_input" / output_name
    render_output_root = output_root / "recon_results" / output_name
    render_checkpoint_dir = render_output_root / ARTIFIXER3D_EXPERIMENT / distillation_input_dir.name / f"ours_{steps}"
    return Artifixer3DPaths(
        run_root=output_root / "runs",
        distillation_input_dir=distillation_input_dir,
        distillation_selected_indices_path=output_root / "distillation_input" / f"{output_name}_selected_indices.json",
        override_image_dir=distillation_input_dir / "artifixer_predictions",
        render_output_root=render_output_root,
        render_checkpoint_dir=render_checkpoint_dir,
        artifixer3d_plus_inference_split_path=artifixer3d_plus_inference_split_path,
    )


def artifixer3d_checkpoint(scene: PreparedScene, paths: Artifixer3DPaths, steps: int) -> Path:
    """Mirror 3DGRUT checkpoint naming so reuse checks target the expected file."""
    return paths.run_root / scene.scene_id / paths.distillation_input_dir.name / f"ours_{steps}" / f"ckpt_{steps}.pt"


def generated_frame_indices(scene: PreparedScene) -> list[int]:
    """Compute pseudo-view indices as every non-anchor frame in the prepared split."""
    selected = set(scene.selected_indices)
    assert len(selected) == len(
        scene.selected_indices
    ), f"Selected indices contain duplicates: {scene.selected_indices}"
    if selected:
        assert (
            min(selected) >= 0 and max(selected) < scene.frame_count
        ), f"Selected indices must be in [0, {scene.frame_count - 1}], got {scene.selected_indices}"
    return [index for index in range(scene.frame_count) if index not in selected]


def validate_artifixer_frames(scene: PreparedScene, artifixer_frames_dir: Path) -> None:
    """Fail before training if any required ArtiFixer prediction PNG is missing."""
    artifixer_frames_dir = artifixer_frames_dir.resolve()
    if not artifixer_frames_dir.is_dir():
        raise FileNotFoundError(f"Missing ArtiFixer prediction frame directory: {artifixer_frames_dir}")

    generated_indices = generated_frame_indices(scene)
    assert generated_indices, (
        "ArtiFixer3D needs at least one generated/predicted frame. "
        "The prepared selected_indices cover every frame, so there is nothing to distill."
    )

    missing = [index for index in generated_indices if not (artifixer_frames_dir / f"{index:05d}.png").is_file()]
    if missing:
        preview = ", ".join(f"{index:05d}.png" for index in missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing)"
        raise FileNotFoundError(f"Missing ArtiFixer prediction frames in {artifixer_frames_dir}: {preview}{suffix}")


def validate_depth_targets(scene: PreparedScene, artifixer_frames_dir: Path) -> None:
    """Require one shared-reference distance map for every generated view."""
    generated_indices = generated_frame_indices(scene)
    missing = [index for index in generated_indices if not (artifixer_frames_dir / f"{index:05d}_depth.npy").is_file()]
    if missing:
        preview = ", ".join(f"{index:05d}_depth.npy" for index in missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing)"
        raise FileNotFoundError(f"Missing depth-loop targets in {artifixer_frames_dir}: {preview}{suffix}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_surface_depth_manifest(scene: PreparedScene, manifest_path: Path) -> dict[int, tuple[Path, Path]]:
    """Validate trusted real-geometry depth targets and return their files by frame index."""
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing surface-depth manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema") != "artifixer.surface_depth_manifest_v1":
        raise ValueError("surface-depth manifest must use schema artifixer.surface_depth_manifest_v1")
    if payload.get("uses_base_3dgs_depth") is not False:
        raise ValueError("surface-depth manifest must explicitly set uses_base_3dgs_depth=false")
    if payload.get("depth_source") not in {"PATCHMATCH_FUSED_REAL_ONLY", "MESH_FUSED_REAL_ONLY"}:
        raise ValueError("surface-depth manifest must come from fused real-image geometry, not a 3DGS checkpoint")
    frames = payload.get("frames")
    if not isinstance(frames, dict):
        raise ValueError("surface-depth manifest frames must be a mapping")

    expected = generated_frame_indices(scene)
    if set(frames) != {f"{index:05d}" for index in expected}:
        raise ValueError("surface-depth manifest must contain exactly every generated frame")

    resolved: dict[int, tuple[Path, Path]] = {}
    for index in expected:
        key = f"{index:05d}"
        entry = frames[key]
        if not isinstance(entry, dict):
            raise ValueError(f"surface-depth frame {key} must be a mapping")
        quality = entry.get("quality")
        if not isinstance(quality, dict):
            raise ValueError(f"surface-depth frame {key} is missing quality evidence")
        gates = {
            "min_view_support": (3.0, "min"),
            "max_reprojection_error_px": (2.0, "max"),
            "max_relative_depth_disagreement": (0.02, "max"),
            "max_rgb_spread": (0.04, "max"),
        }
        for name, (threshold, direction) in gates.items():
            value = quality.get(name)
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(f"surface-depth frame {key} quality.{name} must be numeric")
            passed = float(value) >= threshold if direction == "min" else float(value) <= threshold
            if not passed:
                raise ValueError(
                    f"surface-depth frame {key} fails {name}: {value} "
                    f"({'minimum' if direction == 'min' else 'maximum'} {threshold})"
                )

        depth_path = (manifest_path.parent / str(entry["depth_path"])).resolve()
        mask_path = (manifest_path.parent / str(entry["confidence_mask_path"])).resolve()
        for path, hash_field in ((depth_path, "depth_sha256"), (mask_path, "confidence_mask_sha256")):
            if not path.is_file():
                raise FileNotFoundError(f"Missing surface-depth target: {path}")
            expected_hash = entry.get(hash_field)
            if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
                raise ValueError(f"surface-depth frame {key} has invalid {hash_field}")
        resolved[index] = (depth_path, mask_path)
    return resolved


def materialize_surface_depth_targets(
    scene: PreparedScene,
    manifest_path: Path,
    override_image_dir: Path,
) -> None:
    """Replace legacy checkpoint depth sidecars with trusted real-geometry targets."""
    for index, (depth_path, mask_path) in load_surface_depth_manifest(scene, manifest_path).items():
        for source, target in (
            (depth_path, override_image_dir / f"{index:05d}_depth.npy"),
            (mask_path, override_image_dir / f"{index:05d}_mask.png"),
        ):
            if target.exists() or target.is_symlink():
                target.unlink()
            symlink_frame(source, target)


def validate_color_loop_targets(scene: PreparedScene, artifixer_frames_dir: Path) -> None:
    """Require sparse colour-loop RGB/mask sidecars for every override view."""
    generated_indices = generated_frame_indices(scene)
    missing = []
    for index in generated_indices:
        for suffix in ("_color_loop.png", "_color_loop_mask.png"):
            if not (artifixer_frames_dir / f"{index:05d}{suffix}").is_file():
                missing.append(f"{index:05d}{suffix}")
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing)"
        raise FileNotFoundError(f"Missing color-loop targets in {artifixer_frames_dir}: {preview}{suffix}")


def reset_directory(path: Path) -> None:
    """Replace materialized inputs explicitly and reject symlinks before deletion."""
    if path.exists():
        assert path.is_dir() and not path.is_symlink(), f"Expected ArtiFixer3D output directory, got {path}"
        shutil.rmtree(path)
    path.mkdir(parents=True)


def source_frame_path(image_root: Path, file_path: str) -> Path:
    """Resolve source frame file_path entries against the prepared COLMAP root."""
    path = Path(file_path)
    return path if path.is_absolute() else image_root / path


def prediction_image_name(index: int) -> str:
    """Name generated frames by index so they cannot collide with source image basenames."""
    return f"frame_{index:05d}.png"


def selected_image_name(index: int, source: Path) -> str:
    """Name real anchors by frame index while preserving the original image extension."""
    return f"frame_{index:05d}{source.suffix.lower()}"


def symlink_frame(source: Path, target: Path) -> None:
    """Share image bytes without copying and fail if the expected source is absent."""
    if not source.is_file():
        raise FileNotFoundError(f"Missing frame source: {source}")
    target.symlink_to(source.resolve())


def opencv_camera_from_mapping(camera_id: int, intrinsics: dict[str, object]) -> Camera:
    """Build the trajectory OPENCV camera from transforms.json intrinsics."""
    width = int(intrinsics["w"])
    height = int(intrinsics["h"])
    params = np.array(
        [
            float(intrinsics["fl_x"]),
            float(intrinsics["fl_y"]),
            float(intrinsics["cx"]),
            float(intrinsics["cy"]),
            float(intrinsics.get("k1") or 0.0),
            float(intrinsics.get("k2") or 0.0),
            float(intrinsics.get("p1") or 0.0),
            float(intrinsics.get("p2") or 0.0),
        ],
        dtype=np.float64,
    )
    return Camera(id=camera_id, model="OPENCV", width=width, height=height, params=params)


def opencv_camera_from_colmap(camera_id: int, camera: Camera, image_size: tuple[int, int]) -> Camera:
    """Convert source COLMAP cameras to the OPENCV Camera object written to cameras.bin."""
    scale_x = image_size[0] / camera.width
    scale_y = image_size[1] / camera.height
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = camera.params[0]
        cx, cy = camera.params[1:3]
        distortion = (0.0, 0.0, 0.0, 0.0)
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
        distortion = (0.0, 0.0, 0.0, 0.0)
    elif camera.model == "SIMPLE_RADIAL":
        fx = fy = camera.params[0]
        cx, cy, k1 = camera.params[1:4]
        distortion = (k1, 0.0, 0.0, 0.0)
    elif camera.model == "RADIAL":
        fx = fy = camera.params[0]
        cx, cy, k1, k2 = camera.params[1:5]
        distortion = (k1, k2, 0.0, 0.0)
    elif camera.model == "OPENCV":
        fx, fy, cx, cy, *distortion = camera.params
    else:
        assert False, f"Unsupported COLMAP camera model for ArtiFixer3D distillation: {camera.model}"

    params = np.array(
        [fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y, *distortion],
        dtype=np.float64,
    )
    return Camera(id=camera_id, model="OPENCV", width=image_size[0], height=image_size[1], params=params)


def opencv_world_to_camera_from_transforms_frame(frame: dict[str, object], applied_transform: np.ndarray) -> np.ndarray:
    """Convert canonical OpenGL C2W split poses into COLMAP/3DGRUT OpenCV W2C."""
    assert "transform_matrix" in frame, "Distillation frame is missing transform_matrix"
    return opengl_c2w_to_opencv_w2c(frame["transform_matrix"], applied_transform)


def colmap_qvec_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Return a normalized COLMAP qvec in wxyz order."""
    xyzw = Rotation.from_matrix(rotation).as_quat()
    qvec = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    if qvec[0] < 0:
        qvec *= -1
    return qvec / np.linalg.norm(qvec)


def colmap_pose_from_transforms_frame(frame: dict[str, object], applied_transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a split transform_matrix to COLMAP OpenCV W2C qvec/tvec."""
    world_to_camera = opencv_world_to_camera_from_transforms_frame(frame, applied_transform)
    return colmap_qvec_from_rotation(world_to_camera[:3, :3]), world_to_camera[:3, 3]


def colmap_rotation_from_qvec(qvec: np.ndarray) -> Rotation:
    """Read a COLMAP qvec in wxyz order as a scipy rotation."""
    qvec = np.asarray(qvec, dtype=np.float64)
    assert qvec.shape == (4,), f"COLMAP qvec must have shape (4,), got {qvec.shape}"
    return Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])


def validate_selected_anchor_poses(
    scene: PreparedScene,
    frames: list[object],
    applied_transform: np.ndarray,
    source_images: dict[str, ColmapImage],
) -> None:
    """Verify selected split frames round-trip to their exact source COLMAP poses."""
    for index in scene.selected_indices:
        assert 0 <= index < len(frames), f"Selected frame {index} is outside transforms frame range"
        frame = frames[index]
        assert isinstance(frame, dict), f"Selected frame {index} metadata must be a mapping"
        assert "file_path" in frame, f"Selected frame {index} is missing file_path"
        source_basename = Path(str(frame["file_path"])).name
        assert source_basename in source_images, f"Selected frame {source_basename!r} is not in source COLMAP"

        candidate_qvec, candidate_tvec = colmap_pose_from_transforms_frame(frame, applied_transform)
        source_image = source_images[source_basename]
        rotation_error_degrees = (
            colmap_rotation_from_qvec(candidate_qvec) * colmap_rotation_from_qvec(source_image.qvec).inv()
        ).magnitude() * 180.0 / np.pi
        translation_error = float(np.linalg.norm(np.asarray(candidate_tvec) - np.asarray(source_image.tvec)))
        if (
            rotation_error_degrees > POSE_VALIDATION_ROTATION_TOLERANCE_DEGREES
            or translation_error > POSE_VALIDATION_TRANSLATION_TOLERANCE
        ):
            raise AssertionError(
                f"Selected frame {index} ({source_basename}) does not match source COLMAP under "
                "the canonical OpenGL C2W camera pose contract: "
                f"rotation_error_degrees={rotation_error_degrees:.6g}, "
                f"translation_error={translation_error:.6g}. "
                "Regenerate the prepared split instead of reusing bad ArtiFixer3D outputs."
            )


def write_colmap_cameras(path: Path, cameras: list[Camera]) -> None:
    """Write cameras.bin because 3DGRUT trains from COLMAP binary inputs."""
    opencv_model = CAMERA_MODEL_NAMES["OPENCV"]
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            assert camera.model == opencv_model.model_name, f"Expected OPENCV camera {camera.id}, got {camera.model}"
            assert len(camera.params) == 8, f"OPENCV camera {camera.id} must have 8 parameters"
            fid.write(
                struct.pack(
                    "<iiQQ",
                    camera.id,
                    opencv_model.model_id,
                    int(camera.width),
                    int(camera.height),
                )
            )
            fid.write(struct.pack("<" + "d" * len(camera.params), *camera.params))


def write_colmap_images(path: Path, images: list[tuple[int, np.ndarray, np.ndarray, int, str]]) -> None:
    """Write images.bin with poses and empty observations for distillation input."""
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image_id, qvec, tvec, camera_id, name in images:
            fid.write(struct.pack("<idddddddi", image_id, *qvec, *tvec, camera_id))
            fid.write(name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", 0))


def copy_source_points3d(source_colmap_dir: Path, path: Path) -> None:
    """Seed scratch ArtiFixer3D initialization from the source COLMAP points."""
    source = source_colmap_dir / "sparse" / "0" / "points3D.bin"
    if not source.is_file():
        raise FileNotFoundError(f"Missing source COLMAP points for ArtiFixer3D initialization: {source}")
    data = source.read_bytes()
    if len(data) < 8:
        raise ValueError(f"Source COLMAP points file is too small: {source}")
    point_count = struct.unpack("<Q", data[:8])[0]
    if point_count == 0:
        raise ValueError(f"Source COLMAP points file is empty: {source}")
    path.write_bytes(data)


def source_colmap_lookup(colmap_dir: Path) -> tuple[dict[str, ColmapImage], dict[int, Camera]]:
    """Load source COLMAP metadata to preserve exact real-anchor poses and cameras."""
    sparse_dir = colmap_dir / "sparse" / "0"
    images = read_colmap_extrinsics_binary(sparse_dir / "images.bin")
    cameras = read_colmap_intrinsics_binary(sparse_dir / "cameras.bin")
    images_by_basename = {Path(image.name).name: image for image in images}
    return images_by_basename, cameras


def materialize_distillation_input(
    scene: PreparedScene,
    paths: Artifixer3DPaths,
    artifixer_frames_dir: Path,
) -> None:
    """Build a COLMAP scene combining real anchors with ArtiFixer prediction views."""
    validate_artifixer_frames(scene, artifixer_frames_dir)
    transforms = json.loads(scene.transforms_path.read_text())
    frames = transforms["frames"]
    assert isinstance(frames, list), f"{scene.transforms_path} must contain a frames list"
    applied_transform = applied_transform_matrix(transforms)

    image_dir = paths.distillation_input_dir / "images"
    sparse_dir = paths.distillation_input_dir / "sparse" / "0"
    reset_directory(paths.distillation_input_dir)
    image_dir.mkdir()
    sparse_dir.mkdir(parents=True)
    paths.override_image_dir.mkdir()

    source_images, source_cameras = source_colmap_lookup(scene.colmap_dir)
    validate_selected_anchor_poses(scene, frames, applied_transform, source_images)
    selected = set(scene.selected_indices)
    colmap_cameras = []
    colmap_images = []
    for index, frame in enumerate(frames):
        assert isinstance(frame, dict), f"Frame {index} metadata must be a mapping"
        if index in selected:
            assert "file_path" in frame, f"Selected frame {index} is missing file_path"
            source = source_frame_path(scene.colmap_dir, str(frame["file_path"]))
            image_name = selected_image_name(index, source)
            source_basename = Path(str(frame["file_path"])).name
            assert source_basename in source_images, f"Selected frame {source_basename!r} is not in source COLMAP"
            source_image = source_images[source_basename]
            with PILImage.open(source) as image:
                image_size = image.size
            camera_id = len(colmap_cameras) + 1
            colmap_cameras.append(
                opencv_camera_from_colmap(camera_id, source_cameras[source_image.camera_id], image_size)
            )
            qvec, tvec = source_image.qvec, source_image.tvec
        else:
            source = artifixer_frames_dir / f"{index:05d}.png"
            image_name = prediction_image_name(index)
            camera_id = len(colmap_cameras) + 1
            colmap_cameras.append(opencv_camera_from_mapping(camera_id, camera_intrinsics_for_frame(transforms, frame)))
            symlink_frame(source, paths.override_image_dir / f"{index:05d}.png")
            source_mask = artifixer_frames_dir / f"{index:05d}_mask.png"
            if source_mask.is_file():
                symlink_frame(source_mask, paths.override_image_dir / f"{index:05d}_mask.png")
            source_depth = artifixer_frames_dir / f"{index:05d}_depth.npy"
            if source_depth.is_file():
                symlink_frame(source_depth, paths.override_image_dir / f"{index:05d}_depth.npy")
            for suffix in ("_color_loop.png", "_color_loop_mask.png"):
                source_loop = artifixer_frames_dir / f"{index:05d}{suffix}"
                if source_loop.is_file():
                    symlink_frame(source_loop, paths.override_image_dir / f"{index:05d}{suffix}")
            qvec, tvec = colmap_pose_from_transforms_frame(frame, applied_transform)
        symlink_frame(source, image_dir / image_name)

        colmap_images.append((index + 1, qvec, tvec, camera_id, image_name))

    write_colmap_cameras(sparse_dir / "cameras.bin", colmap_cameras)
    write_colmap_images(sparse_dir / "images.bin", colmap_images)
    copy_source_points3d(scene.colmap_dir, sparse_dir / "points3D.bin")
    write_json(paths.distillation_selected_indices_path, scene.selected_indices)


def train_artifixer3d(
    scene: PreparedScene,
    paths: Artifixer3DPaths,
    *,
    artifixer_frames_dir: Path,
    base_checkpoint: Path | None,
    fresh_optimizer_on_resume: bool,
    geometry_locked: bool,
    opacity_prune_locked: bool,
    density_learning_rate: float,
    resume_topology_start_iteration: int | None,
    resume_topology_add_relocate_end_iteration: int | None,
    resume_topology_perturb_end_iteration: int | None,
    resume_topology_max_gaussians: int,
    depth_guided: bool,
    depth_loss_weight: float,
    single_surface_guided: bool,
    single_surface_loss_weight: float,
    single_surface_front_transmittance: float,
    surface_depth_manifest: Path | None,
    geometry_authorization: Path | None,
    color_loop_guided: bool,
    color_loop_loss_weight: float,
    balanced_real_override_sampling: bool,
    real_samples_per_override: int,
    scale_regularization_weight: float,
    opacity_regularization_weight: float,
    override_reconstruction_weight: float | None,
    override_lpips_weight: float | None,
    force_sh0_on_resume: bool,
    config_name: str,
    steps: int,
    checkpoint_iterations: list[int] | None,
    use_wandb: bool,
    replace: bool,
) -> tuple[Path, bool]:
    """Run or reuse the 3DGRUT distillation checkpoint for the prepared scene."""
    checkpoint = artifixer3d_checkpoint(scene, paths, steps)
    artifixer_frames_dir = artifixer_frames_dir.resolve()
    if checkpoint.is_file() and not replace:
        print(f"Skipping ArtiFixer3D distillation; found {checkpoint}", flush=True)
        return checkpoint, True

    materialize_distillation_input(scene, paths, artifixer_frames_dir)
    if single_surface_guided:
        if surface_depth_manifest is None:
            raise ValueError("--single_surface_guided requires --surface_depth_manifest")
        if "3dgut" in config_name.lower():
            raise ValueError(
                "--single_surface_guided requires a 3DGRT config (for example apps/colmap_3dgrt_mcmc); "
                "3DGUT bakes min_transmittance into its CUDA extension and cannot render both thresholds "
                "differentiably in one training process"
            )
        if geometry_authorization is None:
            raise ValueError("--single_surface_guided requires --geometry_authorization")
        geometry_gate = json.loads(geometry_authorization.resolve().read_text())
        if (
            geometry_gate.get("schema") != "artifixer.geometry_gate_v1"
            or geometry_gate.get("verdict") != "PASS_GEOMETRY_GATE"
            or geometry_gate.get("authorization", {}).get("distillation_authorized") is not True
        ):
            raise ValueError("single-surface distillation is forbidden by the geometry authorization")
        if not 0.0 < single_surface_loss_weight:
            raise ValueError("--single_surface_loss_weight must be positive")
        if not 0.03 < single_surface_front_transmittance < 1.0:
            raise ValueError("--single_surface_front_transmittance must be in (0.03, 1.0)")
        materialize_surface_depth_targets(scene, surface_depth_manifest, paths.override_image_dir)
        # Both the expected surface and the front/full spread are evaluated only
        # where fused real-image geometry passed the manifest confidence gates.
        depth_guided = True
    if depth_guided:
        if depth_loss_weight <= 0:
            raise ValueError("--depth_loss_weight must be positive when --depth_guided is enabled")
        validate_depth_targets(scene, artifixer_frames_dir)
    if color_loop_guided:
        if color_loop_loss_weight <= 0:
            raise ValueError("--color_loop_loss_weight must be positive with --color_loop_guided")
        validate_color_loop_targets(scene, artifixer_frames_dir)
    if real_samples_per_override < 1:
        raise ValueError("--real_samples_per_override must be at least 1")
    if scale_regularization_weight < 0:
        raise ValueError("--scale_regularization_weight must be non-negative")
    if opacity_regularization_weight < 0:
        raise ValueError("--opacity_regularization_weight must be non-negative")
    if density_learning_rate <= 0:
        raise ValueError("--density_learning_rate must be positive")
    if override_reconstruction_weight is not None and override_reconstruction_weight < 0:
        raise ValueError("--override_reconstruction_weight must be non-negative")
    if override_lpips_weight is not None and override_lpips_weight < 0:
        raise ValueError("--override_lpips_weight must be non-negative")

    requested_checkpoints = set(checkpoint_iterations or [])
    invalid_checkpoints = sorted(iteration for iteration in requested_checkpoints if iteration <= 0 or iteration > steps)
    if invalid_checkpoints:
        raise ValueError(
            f"checkpoint iterations must be in [1, {steps}], got {invalid_checkpoints}"
        )
    if force_sh0_on_resume and base_checkpoint is None:
        raise ValueError("--force_sh0_on_resume requires --base_checkpoint")
    if fresh_optimizer_on_resume and base_checkpoint is None:
        raise ValueError("--fresh_optimizer_on_resume requires --base_checkpoint")
    if geometry_locked and opacity_prune_locked:
        raise ValueError("--geometry_locked and --opacity_prune_locked are mutually exclusive")
    if opacity_prune_locked and base_checkpoint is None:
        raise ValueError("--opacity_prune_locked requires --base_checkpoint")
    if opacity_prune_locked and depth_guided:
        raise ValueError(
            "--opacity_prune_locked rejects --depth_guided because base-checkpoint depth can preserve the phantom"
        )
    topology_bounds = (
        resume_topology_start_iteration,
        resume_topology_add_relocate_end_iteration,
        resume_topology_perturb_end_iteration,
    )
    topology_requested = any(value is not None for value in topology_bounds)
    if topology_requested:
        if any(value is None for value in topology_bounds):
            raise ValueError(
                "resumed topology refinement requires start, add/relocate end, and perturb end iterations"
            )
        if base_checkpoint is None:
            raise ValueError("resumed topology refinement requires --base_checkpoint")
        if geometry_locked or opacity_prune_locked:
            raise ValueError(
                "resumed topology refinement is mutually exclusive with geometry and opacity-prune locks"
            )
        assert resume_topology_start_iteration is not None
        assert resume_topology_add_relocate_end_iteration is not None
        assert resume_topology_perturb_end_iteration is not None
        if not (
            0 <= resume_topology_start_iteration
            < resume_topology_add_relocate_end_iteration
            <= resume_topology_perturb_end_iteration
            <= steps
        ):
            raise ValueError(
                "resumed topology bounds must satisfy "
                "0 <= start < add/relocate end <= perturb end <= artifixer3d steps"
            )
        if resume_topology_max_gaussians < 1:
            raise ValueError("--resume_topology_max_gaussians must be positive")
    requested_checkpoints.add(steps)
    checkpoint_iterations_override = ",".join(str(iteration) for iteration in sorted(requested_checkpoints))

    overrides = [
        f"path={paths.distillation_input_dir}",
        f"out_dir={paths.run_root}",
        f"selected_indices_file={paths.distillation_selected_indices_path}",
        f"image_path_override={paths.override_image_dir.name}",
        "test_last=False",
        "export_ingp.enabled=False",
        f"experiment_name={scene.scene_id}",
        f"n_iterations={steps}",
        f"use_wandb={'True' if use_wandb else 'False'}",
        f"checkpoint.iterations=[{checkpoint_iterations_override}]",
    ]
    if geometry_locked:
        if base_checkpoint is None:
            raise ValueError("--geometry_locked requires --base_checkpoint")
        disabled_strategy_start = steps + 1
        disabled_strategy_end = steps + 2
        overrides.extend(
            [
                "resume_optimizer=False",
                "model.optimize_position=False",
                "model.optimize_rotation=False",
                "model.optimize_scale=False",
                "model.optimize_density=False",
                "model.optimize_features_albedo=True",
                "model.optimize_features_specular=False",
                f"strategy.relocate.start_iteration={disabled_strategy_start}",
                f"strategy.relocate.end_iteration={disabled_strategy_end}",
                f"strategy.perturb.start_iteration={disabled_strategy_start}",
                f"strategy.perturb.end_iteration={disabled_strategy_end}",
                f"strategy.add.start_iteration={disabled_strategy_start}",
                f"strategy.add.end_iteration={disabled_strategy_end}",
                "loss.use_lpips_override=True",
                "loss.lambda_lpips_override=0.05",
                "loss.lambda_reconlosses_override=1.0",
                "loss.lambda_l1_override=0.8",
                "loss.lambda_ssim_override=0.2",
            ]
        )
    if opacity_prune_locked:
        disabled_strategy_start = steps + 1
        disabled_strategy_end = steps + 2
        overrides.extend(
            [
                "resume_optimizer=False",
                "model.optimize_position=False",
                "model.optimize_rotation=False",
                "model.optimize_scale=False",
                "model.optimize_density=True",
                "model.optimize_features_albedo=False",
                "model.optimize_features_specular=False",
                "model.monotonic_density_on_resume=True",
                f"optimizer.params.density.lr={density_learning_rate:g}",
                f"strategy.relocate.start_iteration={disabled_strategy_start}",
                f"strategy.relocate.end_iteration={disabled_strategy_end}",
                f"strategy.perturb.start_iteration={disabled_strategy_start}",
                f"strategy.perturb.end_iteration={disabled_strategy_end}",
                f"strategy.add.start_iteration={disabled_strategy_start}",
                f"strategy.add.end_iteration={disabled_strategy_end}",
                "loss.use_lpips_override=False",
                "loss.lambda_reconlosses_override=1.0",
                "loss.lambda_l1_override=0.8",
                "loss.lambda_ssim_override=0.2",
            ]
        )
    if topology_requested:
        assert resume_topology_start_iteration is not None
        assert resume_topology_add_relocate_end_iteration is not None
        assert resume_topology_perturb_end_iteration is not None
        overrides.extend(
            [
                "resume_optimizer=False",
                "model.optimize_position=True",
                "model.optimize_rotation=True",
                "model.optimize_scale=True",
                "model.optimize_density=True",
                "model.optimize_features_albedo=True",
                "model.optimize_features_specular=True",
                "model.monotonic_density_on_resume=False",
                f"strategy.relocate.start_iteration={resume_topology_start_iteration}",
                f"strategy.relocate.end_iteration={resume_topology_add_relocate_end_iteration}",
                f"strategy.add.start_iteration={resume_topology_start_iteration}",
                f"strategy.add.end_iteration={resume_topology_add_relocate_end_iteration}",
                f"strategy.add.max_n_gaussians={resume_topology_max_gaussians}",
                f"strategy.perturb.start_iteration={resume_topology_start_iteration}",
                f"strategy.perturb.end_iteration={resume_topology_perturb_end_iteration}",
            ]
        )
    if depth_guided:
        overrides.extend(
            [
                "loss.use_depth_override=True",
                f"loss.lambda_depth_override={depth_loss_weight:g}",
            ]
        )
    if single_surface_guided:
        overrides.extend(
            [
                "loss.use_single_surface_override=True",
                f"loss.lambda_single_surface_override={single_surface_loss_weight:g}",
                f"loss.single_surface_front_transmittance={single_surface_front_transmittance:g}",
                "loss.depth_override_is_metric_surface=True",
            ]
        )
    if color_loop_guided:
        overrides.extend(
            [
                "loss.use_color_loop_override=True",
                f"loss.lambda_color_loop_override={color_loop_loss_weight:g}",
            ]
        )
    if balanced_real_override_sampling:
        overrides.extend(
            [
                "balanced_real_override_sampling=True",
                f"real_samples_per_override={real_samples_per_override}",
            ]
        )
    if scale_regularization_weight > 0:
        overrides.extend(
            [
                "loss.use_scale=True",
                f"loss.lambda_scale={scale_regularization_weight:g}",
            ]
        )
    if opacity_regularization_weight > 0:
        overrides.extend(
            [
                "loss.use_opacity=True",
                f"loss.lambda_opacity={opacity_regularization_weight:g}",
            ]
        )
    if override_reconstruction_weight is not None:
        overrides.extend(
            [
                "loss.use_lpips_override=True",
                f"loss.lambda_reconlosses_override={override_reconstruction_weight:g}",
            ]
        )
    if override_lpips_weight is not None:
        overrides.extend(
            [
                "loss.use_lpips_override=True",
                f"loss.lambda_lpips_override={override_lpips_weight:g}",
            ]
        )
    if force_sh0_on_resume:
        overrides.extend(
            [
                "resume_optimizer=False",
                "model.force_sh0_on_resume=True",
                "model.optimize_features_specular=False",
            ]
        )
    if fresh_optimizer_on_resume:
        overrides.append("resume_optimizer=False")
    if base_checkpoint is not None:
        base_checkpoint = base_checkpoint.resolve()
        assert base_checkpoint.is_file(), f"Missing initial 3DGRUT checkpoint: {base_checkpoint}"
        overrides.append(f"resume={base_checkpoint}")

    threedgrut_training.train_3dgrut(config_name, overrides, threedgrut_training.DEFAULT_THREEDGRUT_CONFIG_DIR)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"3DGRUT did not write expected ArtiFixer3D checkpoint: {checkpoint}")
    return checkpoint, False


def render_artifixer3d(
    scene: PreparedScene,
    paths: Artifixer3DPaths,
    *,
    checkpoint: Path,
    checkpoint_reused: bool,
    replace: bool,
    render_trajectory_path: Path,
) -> Path:
    """Render the distilled checkpoint on the requested camera trajectory."""
    render_trajectory_path = render_trajectory_path.resolve()
    if not render_trajectory_path.is_file():
        raise FileNotFoundError(f"Missing ArtiFixer3D render trajectory: {render_trajectory_path}")
    render_frame_count = trajectory_frame_count(render_trajectory_path)
    use_distillation_trajectory = render_trajectory_path == scene.transforms_path.resolve()
    trajectory_output_subdir = "" if use_distillation_trajectory else None
    expected_render_dir = (
        paths.render_checkpoint_dir
        if trajectory_output_subdir == ""
        else paths.render_checkpoint_dir / render_trajectory_path.stem
    )
    if (
        not replace
        and checkpoint_reused
        and render_outputs_complete(
            expected_render_dir,
            render_frame_count,
            expected_selected_indices=selected_indices_for_render(paths.distillation_selected_indices_path, None),
        )
    ):
        print(f"Skipping ArtiFixer3D render; found complete outputs in {expected_render_dir}", flush=True)
        return expected_render_dir

    output_dir = render_3dgrut_colmap(
        checkpoint=checkpoint,
        colmap_dir=paths.distillation_input_dir,
        output_root=paths.render_output_root,
        experiment_name=ARTIFIXER3D_EXPERIMENT,
        selected_indices=paths.distillation_selected_indices_path,
        trajectory_path=render_trajectory_path,
        trajectory_output_subdir=trajectory_output_subdir,
    )
    return output_dir


def require_render_outputs(
    render_dir: Path,
    frame_count: int,
    expected_selected_indices_path: Path | None = None,
) -> None:
    """Check ArtiFixer3D renders before writing the next inference split."""
    if not render_dir.is_dir():
        raise FileNotFoundError(f"Missing ArtiFixer3D render checkpoint directory: {render_dir}")

    selected_indices_path = render_dir / "selected_indices.json"
    if not selected_indices_path.is_file():
        raise FileNotFoundError(f"Missing ArtiFixer3D selected-indices file: {selected_indices_path}")
    if expected_selected_indices_path is not None:
        assert load_json_list(selected_indices_path) == load_json_list(
            expected_selected_indices_path
        ), f"ArtiFixer3D selected indices at {selected_indices_path} do not match {expected_selected_indices_path}"

    for name in ("renders", "opacity"):
        directory = render_dir / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing ArtiFixer3D render output directory: {directory}")
        missing = [index for index in range(frame_count) if not (directory / f"{index:05d}.png").is_file()]
        if missing:
            preview = ", ".join(f"{index:05d}.png" for index in missing[:10])
            suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing)"
            raise FileNotFoundError(f"Missing ArtiFixer3D {name} frames in {directory}: {preview}{suffix}")


def write_artifixer3d_plus_inference_split(
    scene: PreparedScene,
    paths: Artifixer3DPaths,
    render_dir: Path,
    reconstruction_checkpoint: Path,
    *,
    render_frame_count: int | None = None,
) -> Path:
    """Write the reconstructed_colmap split consumed by ArtiFixer3D+ inference."""
    output_split = paths.artifixer3d_plus_inference_split_path
    require_render_outputs(
        render_dir,
        scene.frame_count if render_frame_count is None else render_frame_count,
        expected_selected_indices_path=paths.distillation_selected_indices_path,
    )
    if not reconstruction_checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing ArtiFixer3D checkpoint for ArtiFixer3D+ inference: {reconstruction_checkpoint}"
        )

    selected_indices_path = render_dir / "selected_indices.json"
    entry = {
        "scene_id": scene.scene_id,
        "transforms_path": path_for_split(output_split, scene.transforms_path),
        "image_root": path_for_split(output_split, scene.colmap_dir),
        "render_dir": path_for_split(output_split, render_dir / "renders"),
        "opacity_dir": path_for_split(output_split, render_dir / "opacity"),
        "selected_indices_path": path_for_split(output_split, selected_indices_path),
        "prompt_path": path_for_split(output_split, scene.prompt_path),
        "reconstruction_checkpoint": path_for_split(output_split, reconstruction_checkpoint),
        "camera_scale": scene.camera_scale,
    }
    if scene.target_indices_path is not None:
        entry["target_indices_path"] = path_for_split(output_split, scene.target_indices_path)
    if not scene.has_gt:
        entry["has_gt"] = False

    write_json(output_split, {"test": {scene.scene_id: entry}})
    print(f"Wrote ArtiFixer3D+ inference split: {output_split}", flush=True)
    return output_split


def parse_phases(value: str) -> set[str]:
    """Validate comma-separated phase names so partial reruns are explicit."""
    phases = {phase.strip() for phase in value.split(",") if phase.strip()}
    valid = {"distill", "render", "prepare_artifixer3d_plus"}
    unknown = phases - valid
    assert not unknown, f"Unknown phases: {sorted(unknown)}"
    assert phases, f"--phases must include at least one of {sorted(valid)}"
    return phases


def require_passed_phantom_qc(report_path: Path | None) -> None:
    """Authorize final 2D+ preparation only after the geometry-only gate passed."""
    if report_path is None:
        raise ValueError("--require_geometry_qc_for_plus requires --geometry_qc_report")
    report_path = report_path.resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing geometry QC report: {report_path}")
    report = json.loads(report_path.read_text())
    if report.get("schema") != "artifixer.phantom_qc_v1":
        raise ValueError("geometry QC report must use schema artifixer.phantom_qc_v1")
    if report.get("verdict") != "PASS_SINGLE_SURFACE":
        raise ValueError(
            f"geometry QC did not pass ({report.get('verdict')}); ArtiFixer3D+ preparation is forbidden"
        )


def run_artifixer3d(args: argparse.Namespace) -> None:
    """Orchestrate requested ArtiFixer3D phases with reuse and replace semantics."""
    phases = parse_phases(args.phases)
    scene = load_prepared_scene(args.scene_root, args.split_path, args.scene_id)
    paths = artifixer3d_paths(
        scene, args.output_root, args.artifixer3d_plus_inference_split_path, args.artifixer3d_steps
    )

    checkpoint = artifixer3d_checkpoint(scene, paths, args.artifixer3d_steps)
    render_trajectory_path = (args.render_trajectory_path or scene.transforms_path).resolve()
    checkpoint_reused = True
    if "distill" in phases:
        assert (
            args.artifixer_frames_dir is not None
        ), "--artifixer_frames_dir is required when --phases includes distill"
        checkpoint, checkpoint_reused = train_artifixer3d(
            scene,
            paths,
            artifixer_frames_dir=args.artifixer_frames_dir,
            base_checkpoint=args.base_checkpoint,
            fresh_optimizer_on_resume=args.fresh_optimizer_on_resume,
            geometry_locked=args.geometry_locked,
            opacity_prune_locked=args.opacity_prune_locked,
            density_learning_rate=args.density_learning_rate,
            resume_topology_start_iteration=args.resume_topology_start_iteration,
            resume_topology_add_relocate_end_iteration=args.resume_topology_add_relocate_end_iteration,
            resume_topology_perturb_end_iteration=args.resume_topology_perturb_end_iteration,
            resume_topology_max_gaussians=args.resume_topology_max_gaussians,
            depth_guided=args.depth_guided,
            depth_loss_weight=args.depth_loss_weight,
            single_surface_guided=args.single_surface_guided,
            single_surface_loss_weight=args.single_surface_loss_weight,
            single_surface_front_transmittance=args.single_surface_front_transmittance,
            surface_depth_manifest=args.surface_depth_manifest,
            geometry_authorization=args.geometry_authorization,
            color_loop_guided=args.color_loop_guided,
            color_loop_loss_weight=args.color_loop_loss_weight,
            balanced_real_override_sampling=args.balanced_real_override_sampling,
            real_samples_per_override=args.real_samples_per_override,
            scale_regularization_weight=args.scale_regularization_weight,
            opacity_regularization_weight=args.opacity_regularization_weight,
            override_reconstruction_weight=args.override_reconstruction_weight,
            override_lpips_weight=args.override_lpips_weight,
            force_sh0_on_resume=args.force_sh0_on_resume,
            config_name=args.config_name,
            steps=args.artifixer3d_steps,
            checkpoint_iterations=args.checkpoint_iterations,
            use_wandb=args.use_wandb,
            replace=args.replace,
        )
    elif "render" in phases and not checkpoint.is_file():
        raise FileNotFoundError(f"Missing ArtiFixer3D checkpoint for render phase: {checkpoint}")

    render_dir = paths.render_checkpoint_dir
    if "render" in phases:
        render_dir = render_artifixer3d(
            scene,
            paths,
            checkpoint=checkpoint,
            checkpoint_reused=checkpoint_reused,
            replace=args.replace,
            render_trajectory_path=render_trajectory_path,
        )
    if "prepare_artifixer3d_plus" in phases:
        if args.require_geometry_qc_for_plus:
            require_passed_phantom_qc(args.geometry_qc_report)
        write_artifixer3d_plus_inference_split(
            scene,
            paths,
            render_dir,
            checkpoint,
            render_frame_count=trajectory_frame_count(render_trajectory_path),
        )

    if "distill" in phases or "render" in phases:
        print(f"artifixer3d_checkpoint={checkpoint}", flush=True)
    if "render" in phases or "prepare_artifixer3d_plus" in phases:
        print(f"artifixer3d_render_dir={render_dir}", flush=True)
    if "prepare_artifixer3d_plus" in phases:
        print(f"artifixer3d_plus_inference_split={paths.artifixer3d_plus_inference_split_path}", flush=True)
