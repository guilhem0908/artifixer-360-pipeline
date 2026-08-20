import numpy as np

from scripts.generate_nerfstudio14_trajectories import VIEW_SPECS, generate


def source_transforms(frame_count: int = 3) -> dict:
    frames = []
    for index in range(frame_count):
        pose = np.eye(4)
        pose[:3, 3] = (index, 2 * index, -index)
        frames.append({"transform_matrix": pose.tolist()})
    return {"frames": frames}


def test_nerfstudio14_layout_and_centres() -> None:
    trajectories, combined, manifest = generate(source_transforms())
    expected = [name for name, _, _, _ in VIEW_SPECS]
    assert manifest["view_order"] == expected
    assert manifest["groups"] == {
        "horizontal_a": ["H000", "H120", "H240"],
        "horizontal_b": ["H060", "H180", "H300"],
        "upper": ["U000", "U090", "U180", "U270"],
        "lower": ["D000", "D090", "D180", "D270"],
    }
    assert manifest["frames_per_view"] == 3
    assert manifest["total_frames"] == 42
    assert len(combined["frames"]) == 42
    for name in expected:
        assert len(trajectories[name]["frames"]) == 3
        for frame_index, frame in enumerate(trajectories[name]["frames"]):
            assert np.allclose(np.asarray(frame["transform_matrix"])[:3, 3], (frame_index, 2 * frame_index, -frame_index))


def test_canonical_forward_directions() -> None:
    trajectories, _, _ = generate(source_transforms(frame_count=1))
    expected_forwards = {
        "H000": (0, 0, -1),
        "H060": (np.sqrt(3) / 2, 0, -0.5),
        "H180": (0, 0, 1),
        "U000": (0, np.sqrt(0.5), -np.sqrt(0.5)),
        "D000": (0, -np.sqrt(0.5), -np.sqrt(0.5)),
    }
    for name, expected in expected_forwards.items():
        pose = np.asarray(trajectories[name]["frames"][0]["transform_matrix"])
        assert np.allclose(-pose[:3, 2], expected, atol=1e-7)
