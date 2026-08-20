from scripts.generate_joint_artifixer_manifest import SNAKE_ORDER
from scripts.generate_position_first_artifixer_manifest import build_manifest
from scripts.generate_viewmajor_position_first_manifest import synthetic_snake_from_view_major_rig


def test_view_major_targets_are_reordered_position_first_without_splitting() -> None:
    frame_count = 8
    rig = {"frames_per_view": frame_count, "view_order": list(SNAKE_ORDER)}
    selected = list(range(14 * frame_count, 14 * frame_count + frame_count))
    manifest = build_manifest(
        synthetic_snake_from_view_major_rig(rig),
        rig,
        selected,
        context_views=4,
    )

    first_targets = manifest["items"][0]["target_indices"][:14]
    expected = [view_index * frame_count for view_index in range(14)]
    assert first_targets == expected
    assert all(len(item["target_indices"]) % 14 == 0 for item in manifest["items"])
