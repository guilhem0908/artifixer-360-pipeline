import numpy as np

from scripts.generate_forward_facing14_trajectories import (
    generate,
    repair_isolated_spikes,
)
from scripts.generate_nerfstudio14_trajectories import VIEW_SPECS


def source_with_spike(count: int = 40) -> dict:
    frames = []
    for index in range(count):
        pose = np.eye(4)
        angle = 0.2 * np.sin(index / 5.0)
        pose[:3, :3] = np.asarray(
            [
                [np.cos(angle), 0.0, -np.sin(angle)],
                [0.0, 1.0, 0.0],
                [np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        pose[:3, 3] = [0.08 * index, 0.03 * np.sin(index / 7.0), -0.01 * index]
        if index == 24:
            pose[:3, 3] += [4.0, -3.0, 2.0]
        frames.append({"file_path": f"images/{index:05d}.png", "transform_matrix": pose.tolist()})
    return {"frames": frames}


def test_isolated_large_reversal_is_repaired_without_moving_other_centres():
    centres = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [20.0, 4.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ]
    )
    repaired, indices = repair_isolated_spikes(centres, jump_ratio=3.0)
    assert indices == [4]
    np.testing.assert_allclose(repaired[4], [4.0, 0.0, 0.0])
    np.testing.assert_allclose(repaired[[0, 1, 2, 3, 5, 6, 7, 8]], centres[[0, 1, 2, 3, 5, 6, 7, 8]])


def test_forward_rig_is_smooth_and_all_views_share_each_centre():
    trajectories, combined, manifest = generate(
        source_with_spike(),
        output_frame_count=77,
        jump_ratio=5.0,
        smooth_iterations=2,
        heading_span=3,
    )
    assert manifest["forward_facing"] is True
    assert manifest["frames_per_view"] == 77
    assert manifest["smoothing"]["repaired_isolated_source_indices"] == [24]
    assert manifest["smoothing"]["smoothed_steps"]["maximum_over_median"] < 1.2
    assert manifest["orientation_metrics"]["forward_dot_minimum"] > 0.995
    assert len(combined["frames"]) == 77 * len(VIEW_SPECS)

    names = [name for name, _, _, _ in VIEW_SPECS]
    h000 = np.asarray([frame["transform_matrix"] for frame in trajectories["H000"]["frames"]])
    centres = h000[:, :3, 3]
    tangents = np.gradient(centres, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    forward = -h000[:, :3, 2]
    assert np.min(np.sum(forward * tangents, axis=1)) > 0.995
    for frame_index in range(77):
        rig_centres = [
            np.asarray(trajectories[name]["frames"][frame_index]["transform_matrix"])[:3, 3]
            for name in names
        ]
        np.testing.assert_allclose(rig_centres, np.repeat([rig_centres[0]], len(names), axis=0))


def test_generated_targets_never_claim_source_image_paths():
    trajectories, _, _ = generate(source_with_spike(), output_frame_count=40)
    assert all(
        "file_path" not in frame
        for trajectory in trajectories.values()
        for frame in trajectory["frames"]
    )
