#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Measure and draw the spherical coverage of the 14-view perspective rig.

The rig definition comes from ``generate_nerfstudio14_trajectories`` and the
projection from ``stitch_artifixer_frustums360``, i.e. the same code paths the
pipeline uses. For every direction of the sphere (sampled on an equirectangular
grid and weighted by solid angle) the script counts how many of the 14 frusta
see it, compares that with the footprint of the source pinhole camera, and
writes

* ``docs/results/rig_coverage.json`` with the statistics, and
* ``docs/assets/readme/rig_coverage.png`` with two panels: overlap count and
  the angular ownership used by the stitcher.

CPU only, no random component:

    python scripts/render_rig_coverage_figure.py
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from scripts.generate_nerfstudio14_trajectories import (
        DEFAULT_FOV_DEGREES,
        DEFAULT_SIZE,
        VIEW_SPECS,
        generate,
    )
    from scripts.stitch_artifixer_frustums360 import FrustumProjector, Manifest
except ModuleNotFoundError:  # Direct execution as ``python scripts/...py``.
    from generate_nerfstudio14_trajectories import (
        DEFAULT_FOV_DEGREES,
        DEFAULT_SIZE,
        VIEW_SPECS,
        generate,
    )
    from stitch_artifixer_frustums360 import FrustumProjector, Manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE = REPO_ROOT / "docs" / "assets" / "readme" / "rig_coverage.png"
DEFAULT_REPORT = REPO_ROOT / "docs" / "results" / "rig_coverage.json"

# Intrinsics of the source video used during development, as recorded in
# docs/gauvain_artifixer_no_go_2026-07-22.md.
SOURCE_WIDTH = 1972
SOURCE_HEIGHT = 1087
SOURCE_FOCAL_PX = 924.0169

COUNT_COLORS = {
    1: (239, 243, 255),
    2: (198, 219, 239),
    3: (107, 174, 214),
    4: (33, 113, 181),
    5: (8, 48, 107),
    6: (3, 19, 43),
}
VIEW_COLORS = (
    (31, 119, 180), (174, 199, 232), (44, 160, 44), (152, 223, 138), (23, 190, 207), (158, 218, 229),
    (255, 127, 14), (255, 187, 120), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
)
BACKGROUND = (255, 255, 255)
INK = (30, 30, 30)
FOOTPRINT = (255, 214, 0)


def build_projector(size: int, fov_degrees: float, erp_width: int) -> FrustumProjector:
    """Instantiate the stitcher's projector for a rig placed at the origin."""
    source = {"frames": [{"transform_matrix": np.eye(4).tolist()}]}
    _, _, manifest = generate(source, size=size, fov_degrees=fov_degrees)
    with tempfile.TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = Manifest.load(manifest_path)
    return FrustumProjector(loaded, erp_width, erp_width // 2)


def solid_angle_weights(width: int, height: int) -> np.ndarray:
    """Per-pixel share of the sphere on an equirectangular grid (sums to 1)."""
    latitude = np.pi / 2 - ((np.arange(height) + 0.5) / height) * np.pi
    weights = np.repeat(np.cos(latitude)[:, None], width, axis=1)
    return weights / weights.sum()


def source_footprint(width: int, height: int) -> np.ndarray:
    """Directions seen by the forward-looking source pinhole camera."""
    longitude = ((np.arange(width) + 0.5) / width) * (2 * np.pi) - np.pi
    latitude = np.pi / 2 - ((np.arange(height) + 0.5) / height) * np.pi
    lon, lat = np.meshgrid(longitude, latitude)
    x = np.cos(lat) * np.sin(lon)
    y = -np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = (
            (z > 0)
            & (np.abs(x / z) <= (SOURCE_WIDTH / 2) / SOURCE_FOCAL_PX)
            & (np.abs(y / z) <= (SOURCE_HEIGHT / 2) / SOURCE_FOCAL_PX)
        )
    return inside


def analytic_source_fraction() -> float:
    """Closed-form share of the sphere inside the source camera frustum."""
    half_h = math.atan((SOURCE_WIDTH / 2) / SOURCE_FOCAL_PX)
    half_v = math.atan((SOURCE_HEIGHT / 2) / SOURCE_FOCAL_PX)
    return 4 * math.asin(math.sin(half_h) * math.sin(half_v)) / (4 * math.pi)


def measure(projector: FrustumProjector, size: int, fov_degrees: float) -> tuple[dict[str, object], np.ndarray]:
    order = projector.manifest.view_order
    coverage = np.sum([projector.maps[view].valid for view in order], axis=0)
    weights = solid_angle_weights(projector.width, projector.height)
    footprint = source_footprint(projector.width, projector.height)
    report = {
        "rig": {
            "views": len(order),
            "view_order": list(order),
            "view_size_px": size,
            "fov_degrees": fov_degrees,
            "yaw_pitch_degrees": {name: [yaw, pitch] for name, yaw, pitch, _ in VIEW_SPECS},
        },
        "erp_grid": [projector.width, projector.height],
        "views_per_direction": {
            "min": int(coverage.min()),
            "max": int(coverage.max()),
            "solid_angle_weighted_mean": round(float((coverage * weights).sum()), 4),
        },
        "sphere_share_seen_by_at_least": {
            str(count): round(float(((coverage >= count) * weights).sum()), 4)
            for count in range(1, int(coverage.max()) + 1)
        },
        # Pairs sharing at least max(64, pixels / 10000) grid pixels, as the
        # stitcher defines an overlap when it solves exposure gains.
        "overlapping_view_pairs": len(projector.overlap_pairs),
        "ownership_share": {
            view: round(float(((projector.owner_labels == index) * weights).sum()), 4)
            for index, view in enumerate(order)
        },
        "source_camera": {
            "image_size_px": [SOURCE_WIDTH, SOURCE_HEIGHT],
            "focal_px": SOURCE_FOCAL_PX,
            "sphere_share_analytic": round(analytic_source_fraction(), 4),
            "sphere_share_on_grid": round(float((footprint * weights).sum()), 4),
        },
    }
    return report, coverage


def outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    kernel = np.ones((2 * thickness + 1, 2 * thickness + 1), np.uint8)
    as_uint8 = mask.astype(np.uint8)
    return (cv2.dilate(as_uint8, kernel) - cv2.erode(as_uint8, kernel)).astype(bool)


def label_boundaries(labels: np.ndarray) -> np.ndarray:
    edges = np.zeros(labels.shape, dtype=bool)
    edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edges[1:, :] |= labels[1:, :] != labels[:-1, :]
    return edges


def draw(projector: FrustumProjector, coverage: np.ndarray, report: dict[str, object], path: Path) -> None:
    width, height = projector.width, projector.height
    footprint_edge = outline(source_footprint(width, height))

    count_panel = np.zeros((height, width, 3), dtype=np.uint8)
    for count, color in COUNT_COLORS.items():
        count_panel[coverage == count] = color
    count_panel[label_boundaries(coverage)] = (245, 245, 245)
    count_panel[footprint_edge] = FOOTPRINT

    owner_panel = np.asarray(VIEW_COLORS, dtype=np.uint8)[projector.owner_labels]
    owner_panel[label_boundaries(projector.owner_labels)] = (255, 255, 255)
    owner_panel[footprint_edge] = FOOTPRINT

    panel_width, panel_height = 780, 390
    margin, gap, header, footer = 20, 40, 64, 108
    canvas = Image.new(
        "RGB", (2 * margin + 2 * panel_width + gap, header + panel_height + footer), BACKGROUND
    )
    draw_context = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=22)
    text_font = ImageFont.load_default(size=17)
    label_font = ImageFont.load_default(size=15)

    stats = report["views_per_direction"]
    titles = (
        f"Views covering each direction (min {stats['min']}, mean "
        f"{stats['solid_angle_weighted_mean']:.2f}, max {stats['max']})",
        "Angular ownership used when stitching to ERP",
    )
    for index, (panel, title) in enumerate(zip((count_panel, owner_panel), titles)):
        left = margin + index * (panel_width + gap)
        image = Image.fromarray(panel).resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        canvas.paste(image, (left, header))
        draw_context.rectangle(
            (left - 1, header - 1, left + panel_width, header + panel_height), outline=INK, width=1
        )
        draw_context.text((left, 14), title, font=title_font, fill=INK)
        for longitude, anchor in ((-180, "la"), (-90, "ma"), (0, "ma"), (90, "ma"), (180, "ra")):
            x = left + (longitude + 180) / 360 * panel_width
            draw_context.text(
                (x, header + panel_height + 6), f"{longitude}°", font=label_font, fill=INK, anchor=anchor
            )

    # Left legend: overlap counts and the source footprint.
    legend_y = header + panel_height + 38
    x = margin
    for count, color in COUNT_COLORS.items():
        if not np.any(coverage == count):
            continue
        draw_context.rectangle((x, legend_y, x + 22, legend_y + 22), fill=color, outline=INK)
        draw_context.text((x + 30, legend_y + 11), f"{count} views", font=text_font, fill=INK, anchor="lm")
        x += 118
    second_row = legend_y + 34
    draw_context.rectangle((margin, second_row + 8, margin + 22, second_row + 14), fill=FOOTPRINT, outline=INK)
    share = report["source_camera"]["sphere_share_analytic"]
    draw_context.text(
        (margin + 30, second_row + 11),
        f"field of view of the source camera at one pose ({100 * share:.2f} % of the sphere)",
        font=text_font,
        fill=INK,
        anchor="lm",
    )

    # Right panel: view names at each optical axis.
    left = margin + panel_width + gap
    for view in projector.manifest.view_order:
        row, column = np.unravel_index(int(np.argmax(projector.maps[view].axis_score)), coverage.shape)
        x = left + min(max((column + 0.5) / width * panel_width, 24), panel_width - 24)
        y = header + (row + 0.5) / height * panel_height
        draw_context.text((x, y), view, font=label_font, fill=INK, anchor="mm", stroke_width=3, stroke_fill=BACKGROUND)
    rig = report["rig"]
    draw_context.text(
        (left, legend_y + 11),
        f"{rig['views']} pinhole views sharing one camera centre, {rig['fov_degrees']:.0f}° field of view, "
        f"{rig['view_size_px']} x {rig['view_size_px']} px each",
        font=text_font,
        fill=INK,
        anchor="lm",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="side of each square view in pixels")
    parser.add_argument("--fov-degrees", type=float, default=DEFAULT_FOV_DEGREES)
    parser.add_argument("--erp-width", type=int, default=1024, help="width of the 2:1 ERP grid (pipeline default)")
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    projector = build_projector(args.size, args.fov_degrees, args.erp_width)
    report, coverage = measure(projector, args.size, args.fov_degrees)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    draw(projector, coverage, report, args.figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
