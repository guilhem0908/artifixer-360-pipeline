#!/usr/bin/env python3
"""Build ArtiFixer windows with all angular views grouped by source position.

The prepared Gauvain trajectory is stored as long, view-major lanes.  This
manifest presents it to the video model position-first instead: the fourteen
views at one camera centre are consecutive, the angular order reverses at the
next centre, and every inference item contains complete position groups only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.generate_joint_artifixer_manifest import SNAKE_ORDER, evenly_spaced_indices
except ModuleNotFoundError:  # Direct execution as ``python scripts/...py``.
    from generate_joint_artifixer_manifest import SNAKE_ORDER, evenly_spaced_indices


def lane_targets_by_position(snake: dict, rig: dict) -> dict[str, list[int]]:
    frame_count = int(rig["frames_per_view"])
    rig_views = [str(value) for value in rig["view_order"]]
    if set(rig_views) != set(SNAKE_ORDER):
        raise ValueError("rig does not contain the required fourteen-view spherical snake")

    entries = snake.get("frames")
    lanes = snake.get("lanes")
    if not isinstance(entries, list) or not isinstance(lanes, list):
        raise ValueError("snake manifest is missing frames or lanes")
    result: dict[str, list[int]] = {}
    for lane in lanes:
        name = str(lane["name"])
        if name not in rig_views:
            continue
        lane_entries = entries[int(lane["start"]) : int(lane["stop"])]
        by_position = {
            int(entry["source_index"]): int(entry["target_index"])
            for entry in lane_entries
            if entry.get("kind") == "lane"
        }
        expected = list(range(frame_count))
        if sorted(by_position) != expected:
            raise ValueError(f"lane {name} does not cover positions 0..{frame_count - 1}")
        result[name] = [by_position[position] for position in expected]
    if set(result) != set(rig_views):
        raise ValueError("snake manifest does not contain every rig lane")
    return result


def build_manifest(
    snake: dict,
    rig: dict,
    selected_indices: list[int],
    *,
    context_views: int = 12,
    core_positions: int = 3,
    halo_positions: int = 1,
    maximum_item_frames: int = 77,
) -> dict:
    if not selected_indices or len(set(selected_indices)) != len(selected_indices):
        raise ValueError("selected_indices must be non-empty and unique")
    if context_views <= 0 or context_views > len(selected_indices):
        raise ValueError("context_views exceeds the available real observations")
    if core_positions <= 0 or halo_positions < 0:
        raise ValueError("core_positions must be positive and halo_positions non-negative")

    frame_count = int(rig["frames_per_view"])
    views_per_position = len(SNAKE_ORDER)
    maximum_window_positions = core_positions + 2 * halo_positions
    if maximum_window_positions * views_per_position > maximum_item_frames:
        raise ValueError("a complete position window would exceed maximum_item_frames")

    lane_targets = lane_targets_by_position(snake, rig)
    per_position: list[list[int]] = []
    per_position_views: list[list[str]] = []
    for position in range(frame_count):
        cycle = list(SNAKE_ORDER if position % 2 == 0 else reversed(SNAKE_ORDER))
        per_position.append([lane_targets[view][position] for view in cycle])
        per_position_views.append(cycle)

    fixed_neighbors = [
        selected_indices[index]
        for index in evenly_spaced_indices(len(selected_indices), context_views)
    ]
    lane_target_set = {target for group in per_position for target in group}
    if lane_target_set.intersection(selected_indices):
        raise ValueError("real observations overlap generated trajectory targets")

    items = []
    for core_start in range(0, frame_count, core_positions):
        core_stop = min(frame_count, core_start + core_positions)
        window_start = max(0, core_start - halo_positions)
        window_stop = min(frame_count, core_stop + halo_positions)
        positions = list(range(window_start, window_stop))
        targets = [target for position in positions for target in per_position[position]]
        outputs = [
            target if core_start <= position < core_stop else -1
            for position in positions
            for target in per_position[position]
        ]
        if len(targets) > maximum_item_frames:
            raise AssertionError("position-first inference item exceeds frame budget")
        items.append(
            {
                "target_indices": targets,
                "neighbor_indices": fixed_neighbors,
                "output_indices": outputs,
                # Local IDs deliberately keep every item on a VAE phase boundary.
                # Position groups are never split to manufacture phase padding.
                "global_frame_ids": list(range(len(targets))),
                "core_positions": [core_start, core_stop],
                "window_positions": [window_start, window_stop],
                "position_groups": [
                    {
                        "position": position,
                        "offset": (position - window_start) * views_per_position,
                        "count": views_per_position,
                        "views": per_position_views[position],
                    }
                    for position in positions
                ],
            }
        )

    output_ids = [
        value for item in items for value in item["output_indices"] if value != -1
    ]
    expected_ids = [target for group in per_position for target in group]
    if output_ids != expected_ids or len(set(output_ids)) != len(expected_ids):
        raise AssertionError("core windows do not cover each lane target exactly once")

    connector_targets = sorted(
        int(entry["target_index"])
        for entry in snake["frames"]
        if entry.get("kind") == "stationary_turn"
    )
    return {
        "schema_version": 1,
        "layout": "gauvain_position_first_14view_snake_windows",
        "frame_count": frame_count,
        "views_per_position": views_per_position,
        "target_count": len(expected_ids),
        "excluded_connector_count": len(connector_targets),
        "excluded_connector_indices": connector_targets,
        "context_views": context_views,
        "core_positions": core_positions,
        "halo_positions": halo_positions,
        "maximum_item_frames": maximum_item_frames,
        "snake_order": list(SNAKE_ORDER),
        "selected_neighbors": fixed_neighbors,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snake_manifest", type=Path)
    parser.add_argument("rig_manifest", type=Path)
    parser.add_argument("selected_indices", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--context-views", type=int, default=12)
    parser.add_argument("--core-positions", type=int, default=3)
    parser.add_argument("--halo-positions", type=int, default=1)
    parser.add_argument("--maximum-item-frames", type=int, default=77)
    args = parser.parse_args()
    manifest = build_manifest(
        json.loads(args.snake_manifest.read_text()),
        json.loads(args.rig_manifest.read_text()),
        json.loads(args.selected_indices.read_text()),
        context_views=args.context_views,
        core_positions=args.core_positions,
        halo_positions=args.halo_positions,
        maximum_item_frames=args.maximum_item_frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    maximum = max(len(item["target_indices"]) for item in manifest["items"])
    print(
        f"items={len(manifest['items'])} targets={manifest['target_count']} "
        f"max_item_frames={maximum} complete_position_groups=true"
    )


if __name__ == "__main__":
    main()
