#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Build the looping README preview from the versioned comparison video.

The source MP4 is a 2048x512 side-by-side clip (raw 3DGRUT ERP on the left,
ArtiFixer3D+ on the right) with a 44-pixel caption band burned into the top of
every frame. This script decodes it, redraws the caption band in English,
downsizes the frames and writes

* an animated, infinitely looping WebP small enough to play inline in the
  README, and
* a still JPEG of one frame for places where an animation is not wanted.

It only needs OpenCV, NumPy and Pillow, runs on a CPU in well under a minute
and has no random component.

Example:

    python scripts/build_readme_comparison.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "docs" / "assets" / "readme"
DEFAULT_SOURCE = ASSET_DIR / "gauvain_154f_3DGRUT-brut-ERP_vs_ArtiFixer3Dplus-451k.mp4"
DEFAULT_WEBP = ASSET_DIR / "comparison_raw_3dgrut_vs_artifixer3dplus_451k.webp"
DEFAULT_STILL = ASSET_DIR / "comparison_raw_3dgrut_vs_artifixer3dplus_451k_preview.jpg"

CAPTION_BAND_HEIGHT = 44
CAPTION_BAND_COLOR = (36, 37, 36)
CAPTION_TEXT_COLOR = (255, 255, 255)
LEFT_CAPTION = "Raw 3DGRUT render  |  ERP  |  154 frames"
RIGHT_CAPTION = "ArtiFixer3D+  |  451k"


def read_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    """Decode every frame of ``video_path`` as RGB uint8 arrays."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"no frame decoded from {video_path}")
    if fps <= 0:
        raise ValueError(f"invalid frame rate reported for {video_path}")
    return frames, fps


def build_caption_band(width: int, height: int) -> np.ndarray:
    """Render the English caption band for a side-by-side frame."""
    band = Image.new("RGB", (width, height), CAPTION_BAND_COLOR)
    draw = ImageDraw.Draw(band)
    font = ImageFont.load_default(size=round(height * 0.6))
    for caption, centre_x in ((LEFT_CAPTION, width // 4), (RIGHT_CAPTION, 3 * width // 4)):
        draw.text((centre_x, height // 2), caption, font=font, fill=CAPTION_TEXT_COLOR, anchor="mm")
    return np.asarray(band)


def relabel(frame: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Replace the burned-in caption band of one frame."""
    if frame.shape[1] != band.shape[1]:
        raise ValueError("caption band width does not match the frame width")
    output = frame.copy()
    output[: band.shape[0]] = band
    return output


def build(
    source: Path,
    webp_path: Path,
    still_path: Path,
    width: int,
    frame_step: int,
    quality: int,
    still_frame: int,
) -> dict[str, object]:
    frames, fps = read_frames(source)
    source_height, source_width = frames[0].shape[:2]
    if width % 2 or width <= 0:
        raise ValueError("width must be a positive even number")
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")
    if not 0 <= still_frame < len(frames):
        raise ValueError(f"still_frame must be in [0, {len(frames) - 1}]")

    band = build_caption_band(source_width, CAPTION_BAND_HEIGHT)
    height = round(source_height * width / source_width)
    selected = frames[::frame_step]
    resized = [
        Image.fromarray(relabel(frame, band)).resize((width, height), Image.Resampling.LANCZOS)
        for frame in selected
    ]
    frame_duration_ms = round(1000.0 * frame_step / fps)

    webp_path.parent.mkdir(parents=True, exist_ok=True)
    resized[0].save(
        webp_path,
        format="WEBP",
        save_all=True,
        append_images=resized[1:],
        duration=frame_duration_ms,
        loop=0,
        quality=quality,
        method=4,
        minimize_size=True,
    )

    still = Image.fromarray(relabel(frames[still_frame], band))
    still = still.resize((1600, round(source_height * 1600 / source_width)), Image.Resampling.LANCZOS)
    still.save(still_path, format="JPEG", quality=88, optimize=True)

    return {
        "source": source.name,
        "source_frames": len(frames),
        "source_fps": fps,
        "source_size": [source_width, source_height],
        "webp": webp_path.name,
        "webp_frames": len(resized),
        "webp_size": [width, height],
        "webp_frame_duration_ms": frame_duration_ms,
        "webp_bytes": webp_path.stat().st_size,
        "still": still_path.name,
        "still_source_frame": still_frame,
        "still_bytes": still_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="side-by-side comparison MP4")
    parser.add_argument("--webp", type=Path, default=DEFAULT_WEBP, help="output animated WebP")
    parser.add_argument("--still", type=Path, default=DEFAULT_STILL, help="output still JPEG")
    parser.add_argument("--width", type=int, default=1024, help="output width of the animation in pixels")
    parser.add_argument("--frame-step", type=int, default=1, help="keep one source frame out of this many")
    parser.add_argument("--quality", type=int, default=52, help="lossy WebP quality (0-100)")
    parser.add_argument("--still-frame", type=int, default=77, help="source frame index used for the JPEG")
    args = parser.parse_args()
    report = build(
        source=args.source,
        webp_path=args.webp,
        still_path=args.still,
        width=args.width,
        frame_step=args.frame_step,
        quality=args.quality,
        still_frame=args.still_frame,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
