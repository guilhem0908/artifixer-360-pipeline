import json
from pathlib import Path

import numpy as np

from scripts.prepare_depth_synchronized_multiview_windows import (
    phase_aligned_windows,
    prepare,
    temporal_windows,
)


def test_temporal_windows_publish_each_core_once():
    windows = temporal_windows(0, 10, core_size=4, halo_size=1)
    assert [window["window_range"] for window in windows] == [[0, 5], [3, 9], [7, 10]]
    assert [
        value
        for window in windows
        for value in window["output_source_indices"]
        if value != -1
    ] == list(range(10))


def test_phase_aligned_windows_keep_wan_phase_and_publish_midpoint_cores():
    windows = phase_aligned_windows(0, 26, window_size=13, stride=8)
    assert [window["window_range"] for window in windows] == [[0, 13], [8, 21], [16, 26]]
    assert [window["core_range"] for window in windows] == [[0, 10], [10, 18], [18, 26]]
    assert all(window["window_range"][0] % 4 == 0 for window in windows)
    assert [
        value
        for window in windows
        for value in window["output_source_indices"]
        if value != -1
    ] == list(range(26))


def test_long_windows_fill_77_frame_context_and_share_37_frames():
    windows = phase_aligned_windows(0, 117, window_size=77, stride=40)
    assert [window["window_range"] for window in windows] == [[0, 77], [40, 117]]
    assert [window["core_range"] for window in windows] == [[0, 58], [58, 117]]
    assert len(set(windows[0]["source_indices"]).intersection(windows[1]["source_indices"])) == 37
    assert [
        value
        for window in windows
        for value in window["output_source_indices"]
        if value != -1
    ] == list(range(117))


def test_prepare_materializes_window_major_scenes_without_copying_images(tmp_path: Path):
    source = tmp_path / "source"
    render = source / "render"
    opacity = source / "opacity"
    images = source / "images"
    prompt = source / "caption.h5"
    for directory in (render, opacity, images):
        directory.mkdir(parents=True)
    prompt.write_bytes(b"prompt")
    frames = []
    for index in range(8):
        frames.append({"transform_matrix": [[1, 0, 0, index], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})
        (render / f"{index:05d}.png").write_bytes(b"render")
        (opacity / f"{index:05d}.png").write_bytes(b"opacity")
    for index in range(2):
        (images / f"anchor{index}.jpg").write_bytes(b"anchor")
        frames.append(
            {
                "transform_matrix": [[1, 0, 0, index], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "file_path": f"anchor{index}.jpg",
            }
        )
    transforms = source / "transforms.json"
    selected = source / "selected.json"
    targets = source / "targets.json"
    transforms.write_text(json.dumps({"w": 16, "h": 16, "frames": frames}))
    selected.write_text(json.dumps([8, 9]))
    targets.write_text(json.dumps(list(range(8))))
    split = source / "split.json"
    split.write_text(
        json.dumps(
            {
                "test": {
                    "source": {
                        "scene_id": "source",
                        "transforms_path": "transforms.json",
                        "image_root": "images",
                        "render_dir": "render",
                        "opacity_dir": "opacity",
                        "selected_indices_path": "selected.json",
                        "target_indices_path": "targets.json",
                        "prompt_path": "caption.h5",
                        "camera_scale": 1.0,
                        "has_gt": False,
                    }
                }
            }
        )
    )
    rig = source / "rig.json"
    rig.write_text(json.dumps({"view_order": ["A", "B"], "frames_per_view": 4}))
    output = tmp_path / "prepared"
    split_out, manifest_out = prepare(
        source_split=split,
        rig_manifest=rig,
        output_root=output,
        scene_prefix="joint",
        frame_start=0,
        frame_stop=4,
        core_size=2,
        halo_size=1,
        window_size=None,
        window_stride=None,
        replace=False,
    )
    prepared_split = json.loads(split_out.read_text())
    manifest = json.loads(manifest_out.read_text())
    assert len(prepared_split["test"]) == 4
    assert manifest["published_source_indices"] == [0, 1, 2, 3]
    assert len(manifest["windows"]) == 2
    first_render = output / "windows/000/A/renders/00000.png"
    assert first_render.is_symlink()
    assert first_render.resolve() == render / "00000.png"
    first_transforms = json.loads((output / "windows/000/A/transforms.json").read_text())
    assert len(first_transforms["frames"]) == 5  # three targets + two real anchors


def test_prepare_accepts_exact_sparse_global_files_for_requested_interval(tmp_path: Path):
    source = tmp_path / "source"
    render = source / "render"
    opacity = source / "opacity"
    images = source / "images"
    for directory in (render, opacity, images):
        directory.mkdir(parents=True)
    (source / "caption.h5").write_bytes(b"prompt")
    frames = [
        {"transform_matrix": [[1, 0, 0, index], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
        for index in range(8)
    ]
    for index in (1, 2, 5, 6):
        (render / f"{index:05d}.png").write_bytes(b"render")
        (opacity / f"{index:05d}.png").write_bytes(b"opacity")
    for index in range(2):
        (images / f"anchor{index}.jpg").write_bytes(b"anchor")
        frames.append({"transform_matrix": np.eye(4).tolist(), "file_path": f"anchor{index}.jpg"})
    (source / "transforms.json").write_text(json.dumps({"w": 16, "h": 16, "frames": frames}))
    (source / "selected.json").write_text(json.dumps([8, 9]))
    (source / "targets.json").write_text(json.dumps(list(range(8))))
    (source / "split.json").write_text(json.dumps({"test": {"source": {
        "scene_id": "source", "transforms_path": "transforms.json", "image_root": "images",
        "render_dir": "render", "opacity_dir": "opacity", "selected_indices_path": "selected.json",
        "target_indices_path": "targets.json", "prompt_path": "caption.h5", "camera_scale": 1.0,
        "has_gt": False,
    }}}))
    (source / "rig.json").write_text(json.dumps({"view_order": ["A", "B"], "frames_per_view": 4}))
    _, manifest_path = prepare(
        source_split=source / "split.json", rig_manifest=source / "rig.json",
        output_root=tmp_path / "prepared", scene_prefix="sparse", frame_start=1, frame_stop=3,
        core_size=2, halo_size=0, window_size=None, window_stride=None, replace=False,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_render_layout"] == "sparse_requested_view_major"
    assert manifest["required_source_indices"] == [1, 2]
    assert manifest["required_global_target_count"] == 4
