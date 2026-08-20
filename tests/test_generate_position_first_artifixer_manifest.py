from scripts.generate_joint_artifixer_manifest import SNAKE_ORDER
from scripts.generate_position_first_artifixer_manifest import build_manifest


def fixture(frame_count: int = 154) -> tuple[dict, dict, list[int]]:
    frames = []
    lanes = []
    target = 0
    for lane_index, view in enumerate(SNAKE_ORDER):
        start = len(frames)
        positions = range(frame_count) if lane_index % 2 == 0 else reversed(range(frame_count))
        for position in positions:
            frames.append(
                {
                    "target_index": target,
                    "source_index": position,
                    "kind": "lane",
                }
            )
            target += 1
        stop = len(frames)
        lanes.append({"name": view, "start": start, "stop": stop})
        if lane_index + 1 < len(SNAKE_ORDER):
            frames.append(
                {
                    "target_index": target,
                    "source_index": 0,
                    "kind": "stationary_turn",
                }
            )
            target += 1
    snake = {"frames": frames, "lanes": lanes}
    rig = {"frames_per_view": frame_count, "view_order": list(SNAKE_ORDER)}
    selected = list(range(target, target + 77))
    return snake, rig, selected


def test_position_first_items_never_split_a_fourteen_view_group() -> None:
    snake, rig, selected = fixture()
    manifest = build_manifest(snake, rig, selected)

    assert len(manifest["items"]) == 52
    assert manifest["target_count"] == 154 * 14
    assert max(len(item["target_indices"]) for item in manifest["items"]) == 70
    assert all(len(item["target_indices"]) % 14 == 0 for item in manifest["items"])
    assert all(
        group["count"] == 14
        for item in manifest["items"]
        for group in item["position_groups"]
    )
    assert all(len(item["neighbor_indices"]) == 12 for item in manifest["items"])


def test_views_are_consecutive_and_snake_reverses_at_each_position() -> None:
    snake, rig, selected = fixture(frame_count=8)
    manifest = build_manifest(snake, rig, selected, core_positions=3, halo_positions=1)
    first = manifest["items"][0]["position_groups"]

    assert first[0]["views"] == list(SNAKE_ORDER)
    assert first[1]["views"] == list(reversed(SNAKE_ORDER))
    assert first[0]["views"][-1] == first[1]["views"][0]


def test_outputs_cover_only_lane_targets_once() -> None:
    snake, rig, selected = fixture(frame_count=9)
    manifest = build_manifest(snake, rig, selected)
    outputs = [
        value
        for item in manifest["items"]
        for value in item["output_indices"]
        if value != -1
    ]
    connectors = set(manifest["excluded_connector_indices"])

    assert len(outputs) == 9 * 14
    assert len(outputs) == len(set(outputs))
    assert not connectors.intersection(outputs)
    assert all(item["global_frame_ids"][0] == 0 for item in manifest["items"])
