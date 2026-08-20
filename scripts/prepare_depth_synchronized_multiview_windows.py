#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Prepare window-major 14-view inputs for depth-synchronized ArtiFixer.

The source scene stores target renders in view-major order followed by real
anchor frames.  This tool materializes overlapping temporal windows as one
scene per (window, view), while preserving one common set of real anchors and
publishing every requested source timestamp exactly once through core masks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def relative_path(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def temporal_windows(start: int, stop: int, core_size: int, halo_size: int) -> list[dict[str, Any]]:
    if start < 0 or stop <= start:
        raise ValueError("frame interval must satisfy 0 <= start < stop")
    if core_size <= 0 or halo_size < 0:
        raise ValueError("core_size must be positive and halo_size non-negative")
    result = []
    for core_start in range(start, stop, core_size):
        core_stop = min(stop, core_start + core_size)
        window_start = max(start, core_start - halo_size)
        window_stop = min(stop, core_stop + halo_size)
        sources = list(range(window_start, window_stop))
        outputs = [value if core_start <= value < core_stop else -1 for value in sources]
        result.append(
            {
                "window_index": len(result),
                "window_range": [window_start, window_stop],
                "core_range": [core_start, core_stop],
                "source_indices": sources,
                "output_source_indices": outputs,
            }
        )
    published = [value for window in result for value in window["output_source_indices"] if value != -1]
    if published != list(range(start, stop)):
        raise AssertionError("window cores do not cover the requested interval exactly once")
    return result


def phase_aligned_windows(start: int, stop: int, window_size: int, stride: int) -> list[dict[str, Any]]:
    """Build overlapping windows whose starts retain one VAE temporal phase.

    Wan treats its first frame specially and then compresses groups of four.
    A stride divisible by four keeps corresponding RGB timestamps in the same
    temporal VAE phase across adjacent windows.  Core boundaries are placed at
    overlap midpoints so no published frame comes from the outer overlap edge.
    """
    if start < 0 or stop <= start:
        raise ValueError("frame interval must satisfy 0 <= start < stop")
    if window_size <= 1 or stride <= 0 or stride >= window_size:
        raise ValueError("phase-aligned windows require 0 < stride < window_size")
    if (window_size - 1) % 4:
        raise ValueError("window_size must satisfy Wan's 1 + 4*n frame layout")
    if stride % 4:
        raise ValueError("stride must be divisible by four to preserve VAE temporal phase")

    starts = [start]
    while starts[-1] + window_size < stop:
        starts.append(starts[-1] + stride)
    ranges = [(value, min(stop, value + window_size)) for value in starts]
    boundaries = []
    for (_, previous_stop), (current_start, _) in zip(ranges, ranges[1:]):
        overlap_stop = min(previous_stop, stop)
        if overlap_stop <= current_start:
            raise ValueError("phase-aligned windows must overlap")
        boundaries.append((current_start + overlap_stop) // 2)

    result = []
    for index, (window_start, window_stop) in enumerate(ranges):
        core_start = start if index == 0 else boundaries[index - 1]
        core_stop = stop if index == len(ranges) - 1 else boundaries[index]
        if not window_start <= core_start < core_stop <= window_stop:
            raise AssertionError("midpoint core escaped its source window")
        sources = list(range(window_start, window_stop))
        outputs = [value if core_start <= value < core_stop else -1 for value in sources]
        result.append(
            {
                "window_index": index,
                "window_range": [window_start, window_stop],
                "core_range": [core_start, core_stop],
                "source_indices": sources,
                "output_source_indices": outputs,
            }
        )
    published = [value for window in result for value in window["output_source_indices"] if value != -1]
    if published != list(range(start, stop)):
        raise AssertionError("phase-aligned cores do not publish the interval exactly once")
    return result


def symlink_relative(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def prepare(
    *,
    source_split: Path,
    rig_manifest: Path,
    output_root: Path,
    scene_prefix: str,
    frame_start: int,
    frame_stop: int | None,
    core_size: int,
    halo_size: int,
    window_size: int | None = None,
    window_stride: int | None = None,
    replace: bool = False,
) -> tuple[Path, Path]:
    split_raw = json.loads(source_split.read_text())
    entries = split_raw.get("test")
    if not isinstance(entries, dict) or len(entries) != 1:
        raise ValueError("source split must contain exactly one test scene")
    source_entry = next(iter(entries.values()))
    split_root = source_split.parent.resolve()
    rig = json.loads(rig_manifest.read_text())
    view_order = [str(value) for value in rig["view_order"]]
    frames_per_view = int(rig["frames_per_view"])
    if len(view_order) < 2 or frames_per_view <= 0:
        raise ValueError("invalid rig manifest")
    frame_stop = frames_per_view if frame_stop is None else frame_stop
    if frame_stop > frames_per_view:
        raise ValueError("requested stop exceeds rig frames_per_view")
    if (window_size is None) != (window_stride is None):
        raise ValueError("window_size and window_stride must be provided together")
    if window_size is None:
        windows = temporal_windows(frame_start, frame_stop, core_size, halo_size)
        window_strategy = "symmetric_core_halo"
    else:
        windows = phase_aligned_windows(frame_start, frame_stop, window_size, window_stride)
        window_strategy = "wan_phase_aligned_midpoint_core"

    transforms_path = resolve(split_root, source_entry["transforms_path"])
    selected_path = resolve(split_root, source_entry["selected_indices_path"])
    render_dir = resolve(split_root, source_entry["render_dir"])
    opacity_dir = resolve(split_root, source_entry["opacity_dir"])
    image_root = resolve(split_root, source_entry["image_root"])
    prompt_path = resolve(split_root, source_entry["prompt_path"])
    transforms = json.loads(transforms_path.read_text())
    selected_indices = json.loads(selected_path.read_text())
    total_targets = len(view_order) * frames_per_view
    if len(transforms.get("frames", [])) < total_targets:
        raise ValueError("source transforms do not contain the complete view-major rig")
    if not selected_indices or any(index < total_targets for index in selected_indices):
        raise ValueError("selected real anchors must follow every target view")
    anchors = [transforms["frames"][index] for index in selected_indices]
    if not all("file_path" in frame for frame in anchors):
        raise ValueError("every selected anchor must reference a real image")
    required_source_indices = sorted(
        {
            source_index
            for window in windows
            for source_index in window["source_indices"]
        }
    )
    required_global_indices = [
        view_index * frames_per_view + source_index
        for view_index in range(len(view_order))
        for source_index in required_source_indices
    ]
    for directory, suffix in ((render_dir, ".png"), (opacity_dir, ".png")):
        missing = [
            index
            for index in required_global_indices
            if not (directory / f"{index:05d}{suffix}").is_file()
        ]
        if missing:
            preview = ",".join(f"{value:05d}" for value in missing[:8])
            raise ValueError(
                f"source target directory lacks {len(missing)} requested files "
                f"({preview}): {directory}"
            )

    if output_root.exists():
        if not replace:
            raise FileExistsError(output_root)
        if output_root.is_symlink() or not output_root.is_dir():
            raise ValueError(f"refusing to replace non-directory output: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    split_output = output_root / "split.json"
    manifest_output = output_root / "window_manifest.json"
    split_document: dict[str, dict[str, Any]] = {"test": {}}
    manifest_windows = []
    metadata = {key: value for key, value in transforms.items() if key != "frames"}

    for window in windows:
        scene_ids = []
        for view_index, view_name in enumerate(view_order):
            scene_id = f"{scene_prefix}_w{window['window_index']:03d}_{view_name}"
            scene_ids.append(scene_id)
            scene_root = output_root / "windows" / f"{window['window_index']:03d}" / view_name
            local_render = scene_root / "renders"
            local_opacity = scene_root / "opacity"
            local_render.mkdir(parents=True)
            local_opacity.mkdir(parents=True)
            target_frames = []
            for local_index, source_index in enumerate(window["source_indices"]):
                global_index = view_index * frames_per_view + source_index
                target_frames.append(transforms["frames"][global_index])
                symlink_relative(render_dir / f"{global_index:05d}.png", local_render / f"{local_index:05d}.png")
                symlink_relative(
                    opacity_dir / f"{global_index:05d}.png", local_opacity / f"{local_index:05d}.png"
                )
            local_transforms = {**metadata, "frames": [*target_frames, *anchors]}
            local_transforms_path = scene_root / "transforms.json"
            local_selected_path = scene_root / "selected_indices.json"
            local_targets_path = scene_root / "target_indices.json"
            local_transforms_path.write_text(json.dumps(local_transforms, indent=2) + "\n")
            local_selected_path.write_text(
                json.dumps(list(range(len(target_frames), len(target_frames) + len(anchors))), indent=2) + "\n"
            )
            local_targets_path.write_text(json.dumps(list(range(len(target_frames))), indent=2) + "\n")
            split_document["test"][scene_id] = {
                "scene_id": scene_id,
                "transforms_path": relative_path(local_transforms_path, output_root),
                "image_root": relative_path(image_root, output_root),
                "render_dir": relative_path(local_render, output_root),
                "opacity_dir": relative_path(local_opacity, output_root),
                "selected_indices_path": relative_path(local_selected_path, output_root),
                "target_indices_path": relative_path(local_targets_path, output_root),
                "prompt_path": relative_path(prompt_path, output_root),
                "camera_scale": float(source_entry["camera_scale"]),
                "has_gt": False,
            }
        manifest_windows.append({**window, "scene_ids": scene_ids})

    split_output.write_text(json.dumps(split_document, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "layout": "window_major_depth_synchronized_multiview",
        "view_order": view_order,
        "frames_per_view": frames_per_view,
        "published_source_indices": list(range(frame_start, frame_stop)),
        "window_strategy": window_strategy,
        "core_size": core_size,
        "halo_size": halo_size,
        "window_size": window_size,
        "window_stride": window_stride,
        "real_anchor_count": len(anchors),
        "source_render_layout": (
            "complete_view_major"
            if len(required_global_indices) == total_targets
            else "sparse_requested_view_major"
        ),
        "required_source_indices": required_source_indices,
        "required_global_target_count": len(required_global_indices),
        "source_split": str(source_split.resolve()),
        "rig_manifest": str(rig_manifest.resolve()),
        "windows": manifest_windows,
    }
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")
    return split_output, manifest_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--rig-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-prefix", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int)
    parser.add_argument("--core-size", type=int, default=9)
    parser.add_argument("--halo-size", type=int, default=2)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--window-stride", type=int)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    split, manifest = prepare(
        source_split=args.source_split,
        rig_manifest=args.rig_manifest,
        output_root=args.output_root,
        scene_prefix=args.scene_prefix,
        frame_start=args.frame_start,
        frame_stop=args.frame_stop,
        core_size=args.core_size,
        halo_size=args.halo_size,
        window_size=args.window_size,
        window_stride=args.window_stride,
        replace=args.replace,
    )
    raw = json.loads(manifest.read_text())
    print(f"split={split}")
    print(f"window_manifest={manifest}")
    print(f"windows={len(raw['windows'])}")
    print(f"views={len(raw['view_order'])}")
    print(f"published_frames={len(raw['published_source_indices'])}")


if __name__ == "__main__":
    main()
