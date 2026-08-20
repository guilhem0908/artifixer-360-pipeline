#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Build phase-aligned overlapping inference windows for a snake trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evenly_spaced(values: list[int], requested: int) -> list[int]:
    if not values:
        raise ValueError("selected anchor list is empty")
    if requested <= 0 or requested > len(values):
        raise ValueError(f"context_views must be in [1, {len(values)}]")
    indices = [min(len(values) - 1, int(index * len(values) / requested)) for index in range(requested)]
    result = [values[index] for index in indices]
    if len(set(result)) != requested:
        raise AssertionError("evenly spaced selection produced duplicate anchors")
    return result


def translation_break_positions(
    trajectory: dict,
    *,
    ratio: float,
    minimum_step: float = 1e-9,
) -> tuple[list[int], dict[str, float]]:
    if not np.isfinite(ratio) or ratio <= 1.0:
        raise ValueError("translation break ratio must be finite and greater than 1")
    frames = trajectory.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("trajectory must contain at least two frames")
    centres = np.asarray(
        [np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, 3] for frame in frames]
    )
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    positive = steps[steps > minimum_step]
    if positive.size == 0:
        return [], {"median_positive_step": 0.0, "threshold": 0.0, "maximum_step": 0.0}
    median = float(np.median(positive))
    threshold = median * ratio
    breaks = [int(index + 1) for index in np.flatnonzero(steps > threshold)]
    return breaks, {
        "median_positive_step": median,
        "threshold": threshold,
        "maximum_step": float(np.max(steps)),
    }


def build_manifest(
    target_indices: list[int],
    selected_indices: list[int],
    *,
    context_views: int = 12,
    core_size: int = 49,
    halo_size: int = 12,
    vae_phase: int = 4,
    maximum_item_frames: int = 77,
    break_positions: list[int] | None = None,
) -> dict[str, object]:
    if not target_indices or len(set(target_indices)) != len(target_indices):
        raise ValueError("target_indices must be non-empty and unique")
    if not selected_indices or len(set(selected_indices)) != len(selected_indices):
        raise ValueError("selected_indices must be non-empty and unique")
    if set(target_indices).intersection(selected_indices):
        raise ValueError("targets and selected real anchors must be disjoint")
    if core_size <= 0 or halo_size < 0 or vae_phase <= 0 or maximum_item_frames <= 0:
        raise ValueError("window sizes, VAE phase, and maximum item frames must be valid")

    neighbors = evenly_spaced(selected_indices, context_views)
    target_count = len(target_indices)
    break_positions = sorted(set(break_positions or []))
    if any(position <= 0 or position >= target_count for position in break_positions):
        raise ValueError(f"break positions must be inside (0, {target_count})")
    blocks = list(zip([0, *break_positions], [*break_positions, target_count]))
    items: list[dict[str, object]] = []
    for block_index, (block_start, block_stop) in enumerate(blocks):
        block_count = block_stop - block_start
        for local_core_start in range(0, block_count, core_size):
            local_core_stop = min(block_count, local_core_start + core_size)
            local_window_start = max(0, local_core_start - halo_size)
            local_window_stop = min(block_count, local_core_stop + halo_size)
            local_window_start -= local_window_start % vae_phase
            local_positions = list(range(local_window_start, local_window_stop))
            positions = [block_start + position for position in local_positions]
            targets = [target_indices[position] for position in positions]
            outputs = [
                target_indices[position]
                if local_core_start <= local_position < local_core_stop
                else -1
                for position, local_position in zip(positions, local_positions)
            ]
            if len(targets) > maximum_item_frames:
                raise ValueError(
                    f"window {len(items)} contains {len(targets)} targets, exceeding {maximum_item_frames}"
                )
            items.append(
                {
                    "target_indices": targets,
                    "neighbor_indices": neighbors,
                    "output_indices": outputs,
                    "global_frame_ids": local_positions,
                    "block_index": block_index,
                    "core_range": [
                        block_start + local_core_start,
                        block_start + local_core_stop,
                    ],
                    "window_range": [
                        block_start + local_window_start,
                        block_start + local_window_stop,
                    ],
                }
            )

    output_ids = [
        value
        for item in items
        for value in item["output_indices"]
        if value != -1
    ]
    if output_ids != target_indices:
        raise AssertionError("core windows do not cover the target stream exactly once in order")
    return {
        "schema_version": 1,
        "layout": "world_locked_spherical_lane_snake_windows",
        "target_count": target_count,
        "context_views": context_views,
        "core_size": core_size,
        "halo_size": halo_size,
        "vae_phase": vae_phase,
        "maximum_item_frames": maximum_item_frames,
        "break_positions": break_positions,
        "blocks": [{"start": start, "stop": stop} for start, stop in blocks],
        "selected_neighbors": neighbors,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_indices", type=Path)
    parser.add_argument("selected_indices", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--context-views", type=int, default=12)
    parser.add_argument("--core-size", type=int, default=49)
    parser.add_argument("--halo-size", type=int, default=12)
    parser.add_argument("--vae-phase", type=int, default=4)
    parser.add_argument("--maximum-item-frames", type=int, default=77)
    parser.add_argument(
        "--break-position",
        dest="break_positions",
        action="append",
        type=int,
        default=[],
        help="Start a fresh temporal block at this target-stream position; may be repeated.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        help="Optional trajectory used to detect translation discontinuities.",
    )
    parser.add_argument(
        "--translation-break-ratio",
        type=float,
        help="Break before any translation step larger than this multiple of the positive-step median.",
    )
    args = parser.parse_args()
    try:
        targets = json.loads(args.target_indices.read_text())
        selected = json.loads(args.selected_indices.read_text())
        if not isinstance(targets, list) or not isinstance(selected, list):
            raise ValueError("index files must contain JSON arrays")
        detected_breaks: list[int] = []
        break_metrics = None
        if args.translation_break_ratio is not None:
            if args.trajectory is None:
                raise ValueError("--translation-break-ratio requires --trajectory")
            detected_breaks, break_metrics = translation_break_positions(
                json.loads(args.trajectory.read_text()),
                ratio=args.translation_break_ratio,
            )
        manifest = build_manifest(
            targets,
            selected,
            context_views=args.context_views,
            core_size=args.core_size,
            halo_size=args.halo_size,
            vae_phase=args.vae_phase,
            maximum_item_frames=args.maximum_item_frames,
            break_positions=[*args.break_positions, *detected_breaks],
        )
        if break_metrics is not None:
            manifest["translation_break_detection"] = {
                "ratio": args.translation_break_ratio,
                **break_metrics,
                "positions": detected_breaks,
            }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    maximum = max(len(item["target_indices"]) for item in manifest["items"])
    print(f"items={len(manifest['items'])} targets={manifest['target_count']} max_item_frames={maximum}")


if __name__ == "__main__":
    main()
