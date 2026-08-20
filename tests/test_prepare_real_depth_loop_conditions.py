import cv2
import numpy as np

from scripts.prepare_real_depth_loop_conditions import apply_atlas, build_real_atlas


def write_png(path, value, grayscale=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (8, 8) if grayscale else (8, 8, 3)
    assert cv2.imwrite(str(path), np.full(shape, value, np.uint8))


def test_real_depth_atlas_requires_repeated_observation_and_anchors_target(tmp_path) -> None:
    pose = np.eye(4).tolist()
    transforms = {
        "w": 8, "h": 8, "fl_x": 100.0, "fl_y": 100.0, "cx": 4.0, "cy": 4.0,
        "frames": [
            {"transform_matrix": pose},
            {"transform_matrix": pose, "file_path": "images/a.png"},
            {"transform_matrix": pose, "file_path": "images/b.png"},
        ],
    }
    image_root = tmp_path / "source"
    write_png(image_root / "images/a.png", 80)
    write_png(image_root / "images/b.png", 120)
    source_depth = tmp_path / "source_depth"
    source_opacity = tmp_path / "source_opacity"
    for index in range(2):
        source_depth.mkdir(exist_ok=True)
        np.save(source_depth / f"{index:05d}.npy", np.ones((8, 8), np.float32))
        write_png(source_opacity / f"{index:05d}.png", 255, grayscale=True)

    atlas = build_real_atlas(
        transforms,
        image_root=image_root,
        source_depth_dir=source_depth,
        source_opacity_dir=source_opacity,
        anchor_start=1,
        anchor_count=2,
        sample_stride=1,
        voxel_pixels=1.0,
        minimum_source_opacity=0.9,
    )
    target_depth = tmp_path / "target_depth"
    target_depth.mkdir()
    np.save(target_depth / "00000.npy", np.ones((8, 8), np.float32))
    input_render = tmp_path / "input_render"
    input_opacity = tmp_path / "input_opacity"
    write_png(input_render / "00000.png", 0)
    write_png(input_opacity / "00000.png", 0, grayscale=True)
    metrics = apply_atlas(
        transforms,
        atlas,
        target_count=1,
        target_depth_dir=target_depth,
        input_render_dir=input_render,
        input_opacity_dir=input_opacity,
        output_render_dir=tmp_path / "output_render",
        output_opacity_dir=tmp_path / "output_opacity",
        minimum_support=2,
        feather_pixels=0,
    )

    output = cv2.imread(str(tmp_path / "output_render/00000.png"))
    assert metrics[0]["real_loop_coverage"] > 0.9
    assert output.mean() >= 79
