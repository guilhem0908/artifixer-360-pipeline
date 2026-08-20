import numpy as np

from scripts.generate_joint_artifixer_manifest import SNAKE_ORDER, build_manifest
from scripts.generate_nerfstudio14_trajectories import VIEW_SPECS, generate


def make_rig(frame_count: int = 154) -> dict:
    source = {"frames": [{"transform_matrix": np.eye(4).tolist()} for _ in range(frame_count)]}
    _, _, manifest = generate(source, size=64, fov_degrees=110)
    return manifest


def test_joint_windows_cover_every_target_once_and_stay_within_horizon() -> None:
    plan = build_manifest(make_rig(), anchor_start=2156, anchor_count=154)
    outputs = [index for item in plan["items"] for index in item["output_indices"] if index != -1]

    assert len(plan["items"]) == 52
    assert len(outputs) == 14 * 154
    assert len(set(outputs)) == len(outputs)
    assert max(len(item["target_indices"]) for item in plan["items"]) <= 72
    assert all(len(item["neighbor_indices"]) == 12 for item in plan["items"])


def test_phase_prefix_starts_every_window_on_a_multiple_of_four() -> None:
    plan = build_manifest(make_rig(), anchor_start=2156, anchor_count=154)
    assert all(item["global_frame_ids"][0] % 4 == 0 for item in plan["items"])


def test_snake_has_no_large_angular_jump_and_reverses_between_timestamps() -> None:
    rotations = {
        name: np.asarray(rotation)[:3, :3]
        for name, rotation in make_rig(frame_count=1)["local_rotations"].items()
    }
    axes = {name: rotation @ np.array([0.0, 0.0, -1.0]) for name, rotation in rotations.items()}

    cycle = list(SNAKE_ORDER) + [SNAKE_ORDER[0]]
    angles = []
    for first, second in zip(cycle[:-1], cycle[1:]):
        cosine = np.clip(np.dot(axes[first], axes[second]), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cosine)))
    assert max(angles) <= 60.0 + 1e-6
    assert list(reversed(SNAKE_ORDER))[0] == SNAKE_ORDER[-1]


def test_view_specs_still_match_joint_snake() -> None:
    assert {name for name, *_ in VIEW_SPECS} == set(SNAKE_ORDER)
