#!/usr/bin/env python3
"""Build contiguous joint 14-view ArtiFixer inference windows.

The source trajectory is view-major on disk, but ArtiFixer must see nearby
directions together.  This manifest reorders it into an alternating spherical
snake, keeps every item within the 81-frame training horizon, and emits only
the three core timestamps from each five-timestamp window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SNAKE_ORDER = (
    "H000",
    "U000",
    "U090",
    "H060",
    "H120",
    "D090",
    "D180",
    "H180",
    "U180",
    "U270",
    "H240",
    "H300",
    "D270",
    "D000",
)


def evenly_spaced_indices(count: int, requested: int) -> list[int]:
    if requested <= 0 or requested > count:
        raise ValueError(f"requested must be in [1, {count}], got {requested}")
    result = [min(count - 1, int(index * count / requested)) for index in range(requested)]
    return list(dict.fromkeys(result))


def build_global_stream(view_order: list[str], frame_count: int) -> tuple[list[int], list[tuple[int, str]]]:
    if set(view_order) != set(SNAKE_ORDER):
        raise ValueError(f"manifest views do not match the required 14-view rig: {view_order}")
    view_index = {view: index for index, view in enumerate(view_order)}
    target_indices: list[int] = []
    identities: list[tuple[int, str]] = []
    for timestamp in range(frame_count):
        cycle = SNAKE_ORDER if timestamp % 2 == 0 else tuple(reversed(SNAKE_ORDER))
        for view in cycle:
            target_indices.append(view_index[view] * frame_count + timestamp)
            identities.append((timestamp, view))
    return target_indices, identities


def build_manifest(
    rig: dict,
    *,
    anchor_start: int,
    anchor_count: int,
    context_views: int = 12,
    core_timestamps: int = 3,
    halo_timestamps: int = 1,
) -> dict:
    frame_count = int(rig["frames_per_view"])
    view_order = [str(view) for view in rig["view_order"]]
    if len(view_order) != 14:
        raise ValueError(f"expected 14 views, got {len(view_order)}")
    if anchor_count != frame_count:
        raise ValueError(
            "joint same-centre conditioning requires one real anchor per trajectory timestamp: "
            f"anchor_count={anchor_count}, frame_count={frame_count}"
        )
    if context_views < 1 or context_views > anchor_count:
        raise ValueError(f"invalid context_views={context_views}")
    if core_timestamps <= 0 or halo_timestamps < 0:
        raise ValueError("core_timestamps must be positive and halo_timestamps non-negative")

    global_targets, identities = build_global_stream(view_order, frame_count)
    items = []
    for core_start in range(0, frame_count, core_timestamps):
        core_stop = min(frame_count, core_start + core_timestamps)
        window_start = max(0, core_start - halo_timestamps)
        window_stop = min(frame_count, core_stop + halo_timestamps)
        stream_start = window_start * len(view_order)
        stream_stop = window_stop * len(view_order)

        # Align the local VAE phase with the conceptual full snake stream.
        phase_prefix = stream_start % 4
        stream_start -= phase_prefix
        global_ids = list(range(stream_start, stream_stop))
        target_indices = [global_targets[index] for index in global_ids]
        output_indices = []
        for index in global_ids:
            timestamp, _view = identities[index]
            output_indices.append(global_targets[index] if core_start <= timestamp < core_stop else -1)

        window_timestamps = list(range(window_start, window_stop))
        neighbor_timestamps = list(window_timestamps)
        for timestamp in evenly_spaced_indices(anchor_count, context_views):
            if timestamp not in neighbor_timestamps:
                neighbor_timestamps.append(timestamp)
            if len(neighbor_timestamps) == context_views:
                break
        if len(neighbor_timestamps) < context_views:
            for timestamp in range(anchor_count):
                if timestamp not in neighbor_timestamps:
                    neighbor_timestamps.append(timestamp)
                if len(neighbor_timestamps) == context_views:
                    break
        neighbor_timestamps = sorted(neighbor_timestamps[:context_views])
        items.append(
            {
                "target_indices": target_indices,
                "neighbor_indices": [anchor_start + timestamp for timestamp in neighbor_timestamps],
                "output_indices": output_indices,
                "global_frame_ids": global_ids,
                "core_timestamps": [core_start, core_stop],
                "window_timestamps": [window_start, window_stop],
                "phase_prefix_frames": phase_prefix,
            }
        )

    maximum_frames = max(len(item["target_indices"]) for item in items)
    if maximum_frames > 73:
        raise AssertionError(f"joint item exceeds pre-padding frame budget: {maximum_frames}")
    output_ids = [index for item in items for index in item["output_indices"] if index != -1]
    expected_outputs = frame_count * len(view_order)
    if len(output_ids) != expected_outputs or len(set(output_ids)) != expected_outputs:
        raise AssertionError("core windows do not cover every target exactly once")

    return {
        "schema_version": 1,
        "layout": "world_locked14_joint_snake",
        "frame_count": frame_count,
        "target_count": expected_outputs,
        "anchor_start": anchor_start,
        "anchor_count": anchor_count,
        "context_views": context_views,
        "core_timestamps": core_timestamps,
        "halo_timestamps": halo_timestamps,
        "snake_order": list(SNAKE_ORDER),
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rig_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--anchor-start", type=int, required=True)
    parser.add_argument("--anchor-count", type=int, required=True)
    parser.add_argument("--context-views", type=int, default=12)
    parser.add_argument("--core-timestamps", type=int, default=3)
    parser.add_argument("--halo-timestamps", type=int, default=1)
    args = parser.parse_args()
    rig = json.loads(args.rig_manifest.read_text())
    manifest = build_manifest(
        rig,
        anchor_start=args.anchor_start,
        anchor_count=args.anchor_count,
        context_views=args.context_views,
        core_timestamps=args.core_timestamps,
        halo_timestamps=args.halo_timestamps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"items={len(manifest['items'])} targets={manifest['target_count']} "
        f"max_item_frames={max(len(item['target_indices']) for item in manifest['items'])}"
    )


if __name__ == "__main__":
    main()
