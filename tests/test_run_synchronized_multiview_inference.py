import json

import torch

from model_eval.run_synchronized_multiview_inference import (
    build_peer_contexts,
    load_window_manifest,
    merge_window_outputs,
    stack_peer_hidden_states,
    temporal_window_weights,
    temporal_noise_ids,
)
from model_eval.synchronized_multiview import yaw_rotation


def record(value: float) -> dict:
    identity4 = torch.eye(4).repeat(2, 1, 1)
    identity3 = torch.eye(3).repeat(2, 1, 1)
    return {
        "condition": torch.full((4, 2, 3, 3), value),
        "w2cs": identity4,
        "Ks": identity3,
        "neighbor_hidden_states": torch.full((4, 1, 3, 3), 10 + value),
        "neighbor_w2cs": torch.eye(4).unsqueeze(0),
        "neighbor_Ks": torch.eye(3).unsqueeze(0),
    }


def test_peer_context_exposes_other_direction_at_every_latent_timestamp():
    rotations = [
        yaw_rotation(0, device=torch.device("cpu")).tolist(),
        yaw_rotation(60, device=torch.device("cpu")).tolist(),
        yaw_rotation(120, device=torch.device("cpu")).tolist(),
    ]
    records = [record(0), record(1), record(2)]
    contexts = build_peer_contexts(records, rotations, fov_degrees=140.0)
    assert len(contexts) == 3
    hidden = stack_peer_hidden_states(records, contexts, range(1), torch.device("cpu"))
    assert hidden.shape == (1, 4, 5, 3, 3)
    assert contexts[0]["peer_source_indices"] == (1, 2)
    assert int(contexts[0]["real_neighbor_count"]) == 1
    assert int(contexts[0]["peer_neighbor_count"]) == 4
    assert contexts[0]["neighbor_w2cs"].shape == (5, 4, 4)
    # The first peer is yawed relative to the target rather than copied with an
    # incompatible per-view reference frame.
    assert not torch.allclose(contexts[0]["neighbor_w2cs"][1, :3, :3], torch.eye(3))


def test_peer_context_keeps_only_strongest_overlapping_views_when_limited():
    rotations = [
        yaw_rotation(0, device=torch.device("cpu")).tolist(),
        yaw_rotation(45, device=torch.device("cpu")).tolist(),
        yaw_rotation(90, device=torch.device("cpu")).tolist(),
        yaw_rotation(180, device=torch.device("cpu")).tolist(),
    ]
    contexts = build_peer_contexts(
        [record(0), record(1), record(2), record(3)],
        rotations,
        fov_degrees=110.0,
        max_peer_views=1,
    )

    assert contexts[0]["peer_source_indices"] == (1,)
    assert contexts[1]["peer_source_indices"] == (0,)
    assert contexts[2]["peer_source_indices"] == (1,)
    assert contexts[3]["peer_source_indices"] == (2,)
    assert all(len(context["peer_overlap_fractions"]) == 1 for context in contexts)
    assert all(int(context["peer_neighbor_count"]) == 2 for context in contexts)


def test_temporal_noise_ids_are_identical_in_long_window_overlap():
    first = temporal_noise_ids(list(range(0, 77)), padded_num_frames=77, vae_temporal_scale=4)
    second = temporal_noise_ids(list(range(40, 117)), padded_num_frames=77, vae_temporal_scale=4)
    assert first.tolist() == list(range(0, 77, 4))
    assert second.tolist() == list(range(40, 117, 4))
    assert first[10:].tolist() == second[:10].tolist()


def test_raised_cosine_window_weights_form_partition_across_overlap():
    windows = [
        {"source_indices": list(range(0, 5))},
        {"source_indices": list(range(2, 7))},
    ]
    assert temporal_window_weights(windows, 0) == {0: 1.0}
    assert temporal_window_weights(windows, 2) == {0: 1.0, 1: 0.0}
    middle = temporal_window_weights(windows, 3)
    assert abs(middle[0] - 0.5) < 1e-12
    assert abs(middle[1] - 0.5) < 1e-12
    assert temporal_window_weights(windows, 4) == {0: 0.0, 1: 1.0}
    assert temporal_window_weights(windows, 6) == {1: 1.0}
    for source_index in range(7):
        assert abs(sum(temporal_window_weights(windows, source_index).values()) - 1.0) < 1e-12


def test_full_154_frame_windows_have_exactly_two_37_frame_blends():
    windows = [
        {"source_indices": list(range(0, 77))},
        {"source_indices": list(range(40, 117))},
        {"source_indices": list(range(80, 154))},
    ]
    weights = [temporal_window_weights(windows, source_index) for source_index in range(154)]
    assert sum(len(item) == 2 for item in weights) == 74
    assert max(map(len, weights)) == 2
    assert all(abs(sum(item.values()) - 1.0) < 1e-12 for item in weights)
    assert weights[40] == {0: 1.0, 1: 0.0}
    assert weights[76] == {0: 0.0, 1: 1.0}
    assert weights[80] == {1: 1.0, 2: 0.0}
    assert weights[116] == {1: 0.0, 2: 1.0}


def test_merge_window_outputs_blends_every_duplicate_frame(tmp_path):
    from PIL import Image

    windows = [
        {"source_indices": list(range(0, 5))},
        {"source_indices": list(range(2, 7))},
    ]
    view_names = ["H000", "H060"]
    for window_index, value in enumerate((0, 100)):
        for view_name in view_names:
            directory = tmp_path / "window_predictions" / f"window_{window_index:03d}" / view_name
            directory.mkdir(parents=True)
            for source_index in windows[window_index]["source_indices"]:
                Image.fromarray(torch.full((4, 6, 3), value, dtype=torch.uint8).numpy()).save(
                    directory / f"{source_index:05d}.png"
                )

    report = merge_window_outputs(tmp_path, windows, view_names, frames_per_view=7)
    assert report == {
        "method": "pairwise_raised_cosine_rgb_partition_of_unity",
        "overlap_frame_count": 3,
        "maximum_window_contributors": 2,
        "all_window_predictions_retained_until_merge": True,
    }
    expected_values = [0, 0, 0, 50, 100, 100, 100]
    for view_index, view_name in enumerate(view_names):
        for source_index, expected in enumerate(expected_values):
            with Image.open(tmp_path / view_name / f"{source_index:05d}.png") as image:
                assert int(torch.as_tensor(list(image.getdata())).float().mean()) == expected
            assert (tmp_path / "merged_predictions" / f"{view_index * 7 + source_index:05d}.png").is_file()


def test_merge_window_outputs_supports_a_published_subinterval(tmp_path):
    from PIL import Image

    published = list(range(16, 37))
    windows = [
        {
            "source_indices": published,
            "output_source_indices": published,
        }
    ]
    view_names = ["H000", "H060"]
    for view_name in view_names:
        directory = tmp_path / "window_predictions" / "window_000" / view_name
        directory.mkdir(parents=True)
        for source_index in published:
            Image.fromarray(torch.full((4, 6, 3), source_index, dtype=torch.uint8).numpy()).save(
                directory / f"{source_index:05d}.png"
            )

    report = merge_window_outputs(tmp_path, windows, view_names, frames_per_view=154)

    assert report["overlap_frame_count"] == 0
    assert len(list((tmp_path / "merged_predictions").glob("*.png"))) == 42
    assert not (tmp_path / "merged_predictions" / "00000.png").exists()
    assert (tmp_path / "merged_predictions" / "00016.png").is_file()
    assert (tmp_path / "merged_predictions" / f"{154 + 36:05d}.png").is_file()


def test_window_manifest_rejects_missing_published_frame(tmp_path):
    path = tmp_path / "windows.json"
    raw = {
        "schema_version": 1,
        "view_order": ["A", "B"],
        "frames_per_view": 3,
        "published_source_indices": [0, 1, 2],
        "windows": [
            {
                "window_index": 0,
                "scene_ids": ["w0_A", "w0_B"],
                "source_indices": [0, 1],
                "output_source_indices": [0, 1],
            }
        ],
    }
    path.write_text(json.dumps(raw))
    try:
        load_window_manifest(path, view_names=["A", "B"], frames_per_view=3, dataset_size=2)
    except ValueError as error:
        assert "registered source frames" in str(error)
    else:
        raise AssertionError("missing published frame must fail closed")
