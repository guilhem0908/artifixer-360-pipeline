#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Materialize ArtiFixer snake lane predictions in view-major stitcher order."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def prediction_index(input_root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(input_root.glob("**/pred/[0-9][0-9][0-9][0-9][0-9].png")):
        index = int(path.stem)
        if index in result:
            raise ValueError(f"duplicate output index {index:05d}: {result[index]} and {path}")
        result[index] = path
    if not result:
        raise FileNotFoundError(f"no ArtiFixer prediction PNGs below {input_root}")
    return result


def extract(
    predictions: dict[int, Path],
    snake_manifest: dict,
    rig_manifest: dict,
    output_dir: Path,
    *,
    symlink: bool = True,
) -> dict[str, object]:
    view_order = [str(value) for value in rig_manifest["view_order"]]
    frames_per_view = int(rig_manifest["frames_per_view"])
    lanes = {str(lane["name"]): lane for lane in snake_manifest["lanes"]}
    missing_lanes = [view for view in view_order if view not in lanes]
    if missing_lanes:
        raise ValueError(f"snake manifest is missing rig lanes: {missing_lanes}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to replace non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_entries = snake_manifest["frames"]
    blocks = []
    for view_index, view in enumerate(view_order):
        lane = lanes[view]
        lane_entries = frame_entries[lane["start"] : lane["stop"]]
        by_source = {
            int(entry["source_index"]): int(entry["target_index"])
            for entry in lane_entries
        }
        expected_sources = list(range(frames_per_view))
        if sorted(by_source) != expected_sources:
            raise ValueError(
                f"lane {view} does not contain exactly source frames 0..{frames_per_view - 1}"
            )
        block_start = view_index * frames_per_view
        for source_index in expected_sources:
            target_index = by_source[source_index]
            source_path = predictions.get(target_index)
            if source_path is None:
                raise FileNotFoundError(f"missing prediction for snake target {target_index:05d}")
            destination = output_dir / f"{block_start + source_index:05d}.png"
            if symlink:
                destination.symlink_to(source_path.resolve())
            else:
                os.link(source_path, destination)
        blocks.append(
            {
                "view": view,
                "start": block_start,
                "stop": block_start + frames_per_view,
                "snake_lane_index": lane["lane_index"],
                "snake_direction": lane["direction"],
            }
        )
    return {
        "schema_version": 1,
        "layout": "view_major_from_world_locked_spherical_lane_snake",
        "view_order": view_order,
        "frames_per_view": frames_per_view,
        "output_count": len(view_order) * frames_per_view,
        "blocks": blocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--snake-manifest", type=Path, required=True)
    parser.add_argument("--rig-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--hardlink", dest="symlink", action="store_false")
    args = parser.parse_args()
    result = extract(
        prediction_index(args.prediction_root),
        json.loads(args.snake_manifest.read_text()),
        json.loads(args.rig_manifest.read_text()),
        args.output_dir,
        symlink=args.symlink,
    )
    output_manifest = args.output_manifest or args.output_dir / "manifest.json"
    output_manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"views={len(result['view_order'])} frames_per_view={result['frames_per_view']} "
        f"outputs={result['output_count']}"
    )


if __name__ == "__main__":
    main()
