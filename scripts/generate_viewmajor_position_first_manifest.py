#!/usr/bin/env python3
"""Build complete-position ArtiFixer windows from a view-major 14-view rig."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.generate_position_first_artifixer_manifest import build_manifest
except ModuleNotFoundError:  # Direct execution as ``python scripts/...py``.
    from generate_position_first_artifixer_manifest import build_manifest


def synthetic_snake_from_view_major_rig(rig: dict) -> dict:
    frame_count = int(rig["frames_per_view"])
    frames = []
    lanes = []
    for view_index, view in enumerate(rig["view_order"]):
        start = len(frames)
        for position in range(frame_count):
            frames.append(
                {
                    "target_index": view_index * frame_count + position,
                    "source_index": position,
                    "kind": "lane",
                }
            )
        lanes.append({"name": str(view), "start": start, "stop": len(frames)})
    return {"frames": frames, "lanes": lanes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rig_manifest", type=Path)
    parser.add_argument("selected_indices", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--context-views", type=int, default=12)
    parser.add_argument("--core-positions", type=int, default=3)
    parser.add_argument("--halo-positions", type=int, default=1)
    parser.add_argument("--maximum-item-frames", type=int, default=77)
    args = parser.parse_args()
    rig = json.loads(args.rig_manifest.read_text())
    manifest = build_manifest(
        synthetic_snake_from_view_major_rig(rig),
        rig,
        json.loads(args.selected_indices.read_text()),
        context_views=args.context_views,
        core_positions=args.core_positions,
        halo_positions=args.halo_positions,
        maximum_item_frames=args.maximum_item_frames,
    )
    manifest["source_target_layout"] = "view_major"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    maximum = max(len(item["target_indices"]) for item in manifest["items"])
    print(
        f"items={len(manifest['items'])} targets={manifest['target_count']} "
        f"max_item_frames={maximum} complete_position_groups=true source=view_major"
    )


if __name__ == "__main__":
    main()
