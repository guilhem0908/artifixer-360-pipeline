#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Extract complete 14-view position-first clips from a preserved snake MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from scripts.generate_joint_artifixer_manifest import SNAKE_ORDER


def extract(
    video: Path,
    snake_manifest_path: Path,
    rig_manifest_path: Path,
    output_dir: Path,
    *,
    start: int,
    stop: int,
) -> dict[str, object]:
    snake = json.loads(snake_manifest_path.read_text())
    rig = json.loads(rig_manifest_path.read_text())
    frames = snake["frames"]
    lanes = {str(item["name"]): item for item in snake["lanes"]}
    if set(rig["view_order"]) != set(SNAKE_ORDER):
        raise ValueError("rig views do not match the fourteen-view spherical order")
    if not 0 <= start < stop <= int(rig["frames_per_view"]):
        raise ValueError("requested source position range is invalid")
    if output_dir.exists():
        raise FileExistsError(output_dir)

    by_view: dict[str, dict[int, int]] = {}
    for view in rig["view_order"]:
        lane = lanes[str(view)]
        entries = frames[int(lane["start"]) : int(lane["stop"])]
        by_view[str(view)] = {
            int(entry["source_index"]): int(entry["target_index"])
            for entry in entries
            if entry.get("kind") == "lane"
        }

    records: list[dict[str, object]] = []
    for source_pose in range(start, stop):
        views = SNAKE_ORDER if source_pose % 2 == 0 else tuple(reversed(SNAKE_ORDER))
        for view in views:
            records.append(
                {
                    "order_index": len(records),
                    "source_pose": source_pose,
                    "view": view,
                    "snake_frame_index": by_view[view][source_pose],
                }
            )
    by_video_index = {int(record["snake_frame_index"]): record for record in records}
    if len(by_video_index) != len(records):
        raise AssertionError("snake frame selection contains duplicates")

    output_dir.mkdir(parents=True)
    capture = cv2.VideoCapture(str(video))
    frame_index = 0
    written = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        record = by_video_index.get(frame_index)
        if record is not None:
            destination = output_dir / f"{int(record['order_index']):05d}.png"
            if not cv2.imwrite(str(destination), image):
                raise RuntimeError(f"could not write {destination}")
            written += 1
        frame_index += 1
    capture.release()
    if frame_index != len(frames):
        raise ValueError(f"video contains {frame_index} frames, snake manifest contains {len(frames)}")
    if written != len(records):
        raise ValueError(f"wrote {written} frames, expected {len(records)}")

    report = {
        "schema": "artifixer.posefirst_views_from_snake_video_v1",
        "source_video": str(video.resolve()),
        "source_positions": list(range(start, stop)),
        "views_per_position": len(SNAKE_ORDER),
        "frame_count": len(records),
        "ordering": "all views of source pose i, then all views of source pose i+1",
        "records": records,
    }
    (output_dir.parent / "position_first_input_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--snake-manifest", required=True, type=Path)
    parser.add_argument("--rig-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--stop", required=True, type=int)
    args = parser.parse_args()
    report = extract(
        args.video,
        args.snake_manifest,
        args.rig_manifest,
        args.output_dir,
        start=args.start,
        stop=args.stop,
    )
    print(
        f"POSEFIRST_INPUT_PASS positions={len(report['source_positions'])} "
        f"views_per_position={report['views_per_position']} frames={report['frame_count']}"
    )


if __name__ == "__main__":
    main()
