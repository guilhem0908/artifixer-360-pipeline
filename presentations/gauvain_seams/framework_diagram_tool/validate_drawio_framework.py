#!/usr/bin/env python3
"""Structural and export validation for the diagrams.net framework figure."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"
DRAWIO = GENERATED / "artifixer360_framework.drawio"
SVG = GENERATED / "artifixer360_framework.svg"
PDF = GENERATED / "artifixer360_framework.pdf"
PNG = GENERATED / "artifixer360_framework.png"

REQUIRED_IDS = {
    "pipeline_boundary",
    "capture_group",
    "loop_group",
    "synthesis_group",
    "input_frame",
    "input_image",
    "input_title",
    "output_image",
    "output_title",
    "frame_extraction",
    "colmap",
    "posed_real",
    "initial_scene",
    "shared_scene",
    "snake_renders",
    "artifixer",
    "pseudo_observations",
    "distillation",
    "sync_render",
    "final_plus",
    "erp_fusion",
    "input_to_frames",
    "frames_to_colmap",
    "colmap_to_real",
    "real_to_initial",
    "initial_to_shared",
    "shared_to_renders",
    "renders_to_artifixer",
    "artifixer_to_pseudo",
    "pseudo_to_distill",
    "distill_feedback",
    "anchor_junction",
    "anchors_trunk",
    "anchors_to_artifixer",
    "anchors_to_distill",
    "refined_scene_to_sync",
    "sync_to_plus",
    "plus_to_fusion",
    "fusion_to_output",
}

REQUIRED_TEXT = (
    "Pinhole input video",
    "Frame extraction",
    "COLMAP",
    "Posed real frames",
    "Initial 3DGRUT /\n3D Gaussian Scene",
    "Shared 3D\nGaussian Scene",
    "World-locked overlapping\npinhole renders",
    "ArtiFixer",
    "Restored\npseudo-observations",
    "Geometry-constrained\nArtiFixer3D distillation",
    "Synchronized\npanoramic rendering",
    "Final ArtiFixer3D+",
    "Spherical fusion /\nERP projection",
    "Equirectangular 360°\nvideo (ERP)",
    "real-image anchors",
)

FORBIDDEN = (
    re.compile(r"\bcubemap\b", re.I),
    re.compile(r"cube\s*(?:-|→|to)\s*ERP", re.I),
    re.compile(r"\b154\b|\b210\b|\b14 directions\b|\b15 fps\b", re.I),
    re.compile(r"bearlake|room tour", re.I),
)


def parse_geometry(cell: ET.Element) -> tuple[float, float, float, float] | None:
    geom = cell.find("mxGeometry")
    if geom is None or geom.get("relative") == "1":
        return None
    return tuple(float(geom.get(key, "0")) for key in ("x", "y", "width", "height"))


def overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
    tolerance: float = 0.01,
) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (
        ox - tolerance <= ix
        and oy - tolerance <= iy
        and ix + iw <= ox + ow + tolerance
        and iy + ih <= oy + oh + tolerance
    )


def validate() -> None:
    for path in (DRAWIO, SVG, PDF, PNG):
        assert path.is_file() and path.stat().st_size > 0, f"missing/empty export: {path}"

    root = ET.parse(DRAWIO).getroot()
    model = root.find("./diagram/mxGraphModel")
    assert model is not None
    assert model.get("pageWidth") == "1600" and model.get("pageHeight") == "900"

    cells = {cell.get("id"): cell for cell in root.findall(".//mxCell") if cell.get("id")}
    missing = sorted(REQUIRED_IDS - cells.keys())
    assert not missing, f"missing required diagram cells: {missing}"
    assert len(cells) == len(set(cells)), "cell IDs are not unique"

    text = "\n".join(cell.get("value", "") for cell in cells.values())
    for required in REQUIRED_TEXT:
        assert required in text, f"missing required text: {required!r}"
    for pattern in FORBIDDEN:
        assert not pattern.search(text), f"forbidden text found: {pattern.pattern}"

    boundary = cells["pipeline_boundary"]
    assert "dashed=1" in boundary.get("style", ""), "global pipeline boundary is not dashed"
    dashed_cells = [cell.get("id") for cell in cells.values() if "dashed=1" in cell.get("style", "")]
    assert dashed_cells == ["pipeline_boundary"], f"only the global boundary may be dashed: {dashed_cells}"
    bx, by, bw, bh = parse_geometry(boundary)  # type: ignore[misc]

    group_ids = ("capture_group", "loop_group", "synthesis_group")
    assert len(group_ids) == 3, "expected exactly three categorical sub-region containers"
    group_geometries = {group_id: parse_geometry(cells[group_id]) for group_id in group_ids}
    assert all(geometry is not None for geometry in group_geometries.values())
    for group_id, geometry in group_geometries.items():
        assert contains((bx, by, bw, bh), geometry), f"{group_id} is not inside the global pipeline boundary"  # type: ignore[arg-type]
    for index, group_id in enumerate(group_ids):
        for other_id in group_ids[index + 1 :]:
            assert not overlaps(group_geometries[group_id], group_geometries[other_id]), (  # type: ignore[arg-type]
                f"categorical containers overlap: {group_id}, {other_id}"
            )

    # All three regions share one top grid. The two deep processing regions
    # share a bottom grid; the short synthesis region is content-sized so it
    # does not leave a large decorative empty rectangle.
    group_y = {geometry[1] for geometry in group_geometries.values()}  # type: ignore[index]
    assert len(group_y) == 1, (
        f"categorical panels are not aligned: {group_geometries}"
    )
    assert group_geometries["capture_group"][3] == group_geometries["loop_group"][3], (
        f"deep processing panels do not share a bottom grid: {group_geometries}"
    )

    stage_groups = {
        "capture_group": ("frame_extraction", "colmap", "posed_real", "initial_scene"),
        "loop_group": ("shared_scene", "snake_renders", "artifixer", "pseudo_observations", "distillation"),
        "synthesis_group": ("sync_render", "final_plus", "erp_fusion"),
    }
    for group_id, member_ids in stage_groups.items():
        group_geometry = group_geometries[group_id]
        for member_id in member_ids:
            member_geometry = parse_geometry(cells[member_id])
            assert member_geometry is not None
            assert contains(group_geometry, member_geometry), f"{member_id} escapes {group_id}"  # type: ignore[arg-type]

    for outside_id in ("input_frame", "input_image", "input_title", "output_frame", "output_image", "output_title"):
        geometry = parse_geometry(cells[outside_id])
        assert geometry is not None
        x, y, w, h = geometry
        assert x + w <= bx or x >= bx + bw, f"{outside_id} is not outside the internal pipeline boundary"

    # Input and output must be visually balanced: identical outer geometry and
    # the exact same vertical center.
    input_geometry = parse_geometry(cells["input_frame"])
    output_geometry = parse_geometry(cells["output_frame"])
    assert input_geometry is not None and output_geometry is not None
    _, input_y, input_w, input_h = input_geometry
    _, output_y, output_w, output_h = output_geometry
    assert input_y == output_y and input_w == output_w and input_h == output_h, (
        f"input/output cards are not symmetric: {input_geometry}, {output_geometry}"
    )

    for inside_id in ("frame_extraction", "colmap", "posed_real", "initial_scene", "shared_scene", "snake_renders", "artifixer", "pseudo_observations", "distillation", "sync_render", "final_plus", "erp_fusion"):
        geometry = parse_geometry(cells[inside_id])
        assert geometry is not None
        x, y, w, h = geometry
        assert bx <= x and by <= y and x + w <= bx + bw and y + h <= by + bh, f"{inside_id} lies outside the internal boundary"

    # Labels were deliberately placed in dedicated boxes. Verify they do not
    # collide with the main process cards.
    cards = [parse_geometry(cells[cell_id]) for cell_id in ("frame_extraction", "colmap", "posed_real", "initial_scene", "shared_scene", "snake_renders", "artifixer", "pseudo_observations", "distillation", "sync_render", "final_plus", "erp_fusion")]
    for label_id in ("anchors_label", "feedback_label", "pipeline_label", "refined_scene_label"):
        label_geometry = parse_geometry(cells[label_id])
        assert label_geometry is not None
        assert not any(overlaps(label_geometry, card) for card in cards if card is not None), f"{label_id} overlaps a process card"

    edge_count = sum(cell.get("edge") == "1" for cell in cells.values())
    vertex_count = sum(cell.get("vertex") == "1" for cell in cells.values())
    assert edge_count >= 16, f"too few explicit arrows/connectors: {edge_count}"
    assert vertex_count >= 40, f"diagram is unexpectedly flattened: {vertex_count} vertices"

    image_cells = [cell for cell in cells.values() if cell.get("style", "").startswith("shape=image")]
    assert len(image_cells) == 2
    assert all("data:image/" in cell.get("style", "") and "%3Bbase64," in cell.get("style", "") for cell in image_cells)

    svg_text = SVG.read_text(errors="ignore")
    assert "stroke-dasharray" in svg_text
    assert "data:image/png;base64" in svg_text or "data:image/png%3Bbase64" in svg_text
    assert SVG.stat().st_size > 100_000

    with PNG.open("rb") if False else Image.open(PNG) as image:  # keep Pillow as the image source of truth
        assert image.size == (1600, 900), f"unexpected normalized review image size: {image.size}"
        assert image.mode in {"RGB", "RGBA"}

    assert PDF.read_bytes().startswith(b"%PDF-")

    # --- Geometric integrity re-derivation (independent of visual review) -----
    # These checks encode the user's explicit "propre" requirements so that any
    # future layout edit that reintroduces an obstruction fails loudly.
    def style_dict(cell: ET.Element) -> dict:
        out = {}
        for token in cell.get("style", "").split(";"):
            if "=" in token:
                key, value = token.split("=", 1)
                out[key] = value
        return out

    def raw_geo(cell_id: str):
        cell = cells[cell_id]
        geometry = cell.find("mxGeometry")
        if geometry is None or geometry.get("relative") == "1":
            return None
        return [float(geometry.get(k, "0")) for k in ("x", "y", "width", "height")]

    def rotated_aabb(cell_id: str):
        geometry = raw_geo(cell_id)
        if geometry is None:
            return None
        x, y, w, h = geometry
        rotation = float(style_dict(cells[cell_id]).get("rotation", "0") or 0)
        cx, cy = x + w / 2, y + h / 2
        if rotation == 0:
            return (x, y, x + w, y + h)
        radians = math.radians(rotation)
        cos_a, sin_a = abs(math.cos(radians)), abs(math.sin(radians))
        bw, bh = w * cos_a + h * sin_a, w * sin_a + h * cos_a
        return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)

    def boxes_overlap(a, b, min_area_side: float = 0.5) -> bool:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix > min_area_side and iy > min_area_side

    def anchor(cell_id: str, ax: float, ay: float):
        x, y, w, h = raw_geo(cell_id)
        return (x + ax * w, y + ay * h)

    def segment_crosses_rect(p1, p2, rect, margin: float = -2.0) -> bool:
        rx, ry, rw, rh = rect
        rx -= margin
        ry -= margin
        rw += 2 * margin
        rh += 2 * margin
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        t0, t1 = 0.0, 1.0
        for pe, qe in ((-dx, x1 - rx), (dx, (rx + rw) - x1), (-dy, y1 - ry), (dy, (ry + rh) - y1)):
            if pe == 0:
                if qe < 0:
                    return False
            else:
                t = qe / pe
                if pe < 0:
                    if t > t1:
                        return False
                    t0 = max(t0, t)
                else:
                    if t < t0:
                        return False
                    t1 = min(t1, t)
        return t0 < t1

    def point_is_segment_endpoint(point, segment, tolerance: float = 0.01) -> bool:
        return any(
            abs(point[0] - endpoint[0]) <= tolerance
            and abs(point[1] - endpoint[1]) <= tolerance
            for endpoint in segment
        )

    def connector_conflict(segment_a, segment_b, tolerance: float = 0.01):
        """Return an overlap/crossing description for two orthogonal segments.

        A single shared endpoint is valid: it is how explicit branch junctions
        are represented. Collinear overlap and every other crossing are not.
        """

        (ax1, ay1), (ax2, ay2) = segment_a
        (bx1, by1), (bx2, by2) = segment_b
        a_horizontal = abs(ay1 - ay2) <= tolerance
        b_horizontal = abs(by1 - by2) <= tolerance

        if a_horizontal == b_horizontal:
            if a_horizontal:
                if abs(ay1 - by1) > tolerance:
                    return None
                overlap_start = max(min(ax1, ax2), min(bx1, bx2))
                overlap_end = min(max(ax1, ax2), max(bx1, bx2))
                if overlap_end - overlap_start > tolerance:
                    return ("collinear-overlap", (overlap_start, ay1), (overlap_end, ay1))
                return None
            if abs(ax1 - bx1) > tolerance:
                return None
            overlap_start = max(min(ay1, ay2), min(by1, by2))
            overlap_end = min(max(ay1, ay2), max(by1, by2))
            if overlap_end - overlap_start > tolerance:
                return ("collinear-overlap", (ax1, overlap_start), (ax1, overlap_end))
            return None

        horizontal = segment_a if a_horizontal else segment_b
        vertical = segment_b if a_horizontal else segment_a
        hx1, hy = horizontal[0]
        hx2, _ = horizontal[1]
        vx, vy1 = vertical[0]
        _, vy2 = vertical[1]
        if (
            min(hx1, hx2) - tolerance <= vx <= max(hx1, hx2) + tolerance
            and min(vy1, vy2) - tolerance <= hy <= max(vy1, vy2) + tolerance
        ):
            point = (vx, hy)
            if point_is_segment_endpoint(point, segment_a) and point_is_segment_endpoint(point, segment_b):
                return None
            return ("crossing", point)
        return None

    boundary = raw_geo("pipeline_boundary")
    bx, by, bw, bh = boundary  # type: ignore[misc]

    stage_ids = [
        "frame_extraction", "colmap", "posed_real", "initial_scene", "shared_scene",
        "snake_renders", "artifixer", "pseudo_observations", "distillation",
        "sync_render", "final_plus", "erp_fusion",
    ]
    # Every process stage is fully inside the dashed internal-pipeline boundary.
    for stage_id in stage_ids:
        x, y, w, h = raw_geo(stage_id)  # type: ignore[misc]
        assert bx <= x and by <= y and x + w <= bx + bw and y + h <= by + bh, (
            f"stage {stage_id} escapes the internal-pipeline boundary"
        )

    # Input and output frames stay outside that boundary.
    for outside_id in ("input_frame", "input_image", "input_title", "output_frame", "output_image", "output_title"):
        x, y, w, h = raw_geo(outside_id)  # type: ignore[misc]
        assert x + w <= bx or x >= bx + bw, f"{outside_id} is not outside the internal boundary"

    # No connector may pass through the interior of an unrelated card/thumbnail.
    obstacle_ids = stage_ids + ["input_frame", "output_frame"]
    obstacles = {cell_id: raw_geo(cell_id) for cell_id in obstacle_ids}
    edge_cells = [cell for cell in cells.values() if cell.get("edge") == "1"]
    crossings = []
    edge_polylines = {}
    for cell in edge_cells:
        st = style_dict(cell)
        source, target = cell.get("source"), cell.get("target")
        polyline = [anchor(source, float(st.get("exitX", 0.5)), float(st.get("exitY", 0.5)))]
        geometry = cell.find("mxGeometry")
        array = geometry.find("Array[@as='points']") if geometry is not None else None
        if array is not None:
            for point in array.findall("mxPoint"):
                polyline.append((float(point.get("x")), float(point.get("y"))))
        polyline.append(anchor(target, float(st.get("entryX", 0.0)), float(st.get("entryY", 0.5))))
        edge_polylines[cell.get("id")] = polyline
        for index in range(len(polyline) - 1):
            x1, y1 = polyline[index]
            x2, y2 = polyline[index + 1]
            assert abs(x1 - x2) <= 0.01 or abs(y1 - y2) <= 0.01, (
                f"{cell.get('id')} contains a diagonal segment: "
                f"{polyline[index]} -> {polyline[index + 1]}"
            )
            for obstacle_id, rect in obstacles.items():
                if obstacle_id in (source, target):
                    continue
                if segment_crosses_rect(polyline[index], polyline[index + 1], rect):
                    crossings.append((cell.get("id"), obstacle_id))
    assert not crossings, f"connectors cross unrelated card interiors: {sorted(set(crossings))}"

    # Connectors may meet only at an explicit endpoint. This catches both
    # arrow-on-arrow crossings and duplicate collinear routes, including the
    # subtle case where two branches leave a junction along the same segment.
    connector_conflicts = []
    edge_ids = sorted(edge_polylines)
    for edge_index, edge_id in enumerate(edge_ids):
        polyline_a = edge_polylines[edge_id]
        for other_id in edge_ids[edge_index + 1 :]:
            polyline_b = edge_polylines[other_id]
            for segment_a_index in range(len(polyline_a) - 1):
                segment_a = (polyline_a[segment_a_index], polyline_a[segment_a_index + 1])
                for segment_b_index in range(len(polyline_b) - 1):
                    segment_b = (polyline_b[segment_b_index], polyline_b[segment_b_index + 1])
                    conflict = connector_conflict(segment_a, segment_b)
                    if conflict is not None:
                        connector_conflicts.append((edge_id, other_id, conflict))
    assert not connector_conflicts, f"connector overlaps/crossings: {connector_conflicts}"

    def bend_count(polyline) -> int:
        directions = []
        for index in range(len(polyline) - 1):
            x1, y1 = polyline[index]
            x2, y2 = polyline[index + 1]
            if abs(x1 - x2) <= 0.01 and abs(y1 - y2) <= 0.01:
                continue
            direction = "H" if abs(y1 - y2) <= 0.01 else "V"
            if not directions or directions[-1] != direction:
                directions.append(direction)
        return max(0, len(directions) - 1)

    # The visible scientific flow is almost entirely straight. Only the short
    # refined-scene hand-off to synthesis needs one orthogonal corner.
    straight_edge_ids = {
        "input_to_frames",
        "frames_to_colmap",
        "colmap_to_real",
        "real_to_initial",
        "initial_to_shared",
        "shared_to_renders",
        "renders_to_artifixer",
        "artifixer_to_pseudo",
        "pseudo_to_distill",
        "distill_feedback",
        "anchors_to_artifixer",
        "sync_to_plus",
        "plus_to_fusion",
        "fusion_to_output",
    }
    edge_bends = {edge_id: bend_count(polyline) for edge_id, polyline in edge_polylines.items()}
    for edge_id in straight_edge_ids:
        assert edge_bends[edge_id] == 0, f"{edge_id} is no longer straight: {edge_bends[edge_id]} bends"
    assert edge_bends["refined_scene_to_sync"] <= 1, (
        "refined_scene_to_sync has too many bends: "
        f"{edge_bends['refined_scene_to_sync']}"
    )

    # All main connectors are explicitly orthogonal and square-cornered. This
    # prevents auto-routing from reintroducing diagonal or curved segments.
    for cell in edge_cells:
        style = style_dict(cell)
        assert style.get("edgeStyle") == "orthogonalEdgeStyle", f"{cell.get('id')} is not orthogonal"
        assert style.get("rounded") == "0", f"{cell.get('id')} has rounded connector corners"

    # The multi-line snake title and its caption must not sit under the pinhole
    # thumbnails (rotated bounding boxes are accounted for).
    title_box = rotated_aabb("snake_renders_title")
    caption_box = rotated_aabb("snake_caption")
    thumbnails = [rotated_aabb(f"pinhole_view_{index}") for index in range(1, 5)]
    for index, thumb in enumerate(thumbnails, start=1):
        assert not boxes_overlap(title_box, thumb), f"snake title overlaps pinhole_view_{index}"
        assert not boxes_overlap(caption_box, thumb), f"snake caption overlaps pinhole_view_{index}"
    assert not boxes_overlap(title_box, caption_box), "snake title overlaps its caption"

    # Categorical background rectangles use the three functional fills.
    stage_fills = {style_dict(cells[stage_id]).get("fillColor") for stage_id in stage_ids}
    assert {"#E5EEF6", "#F7ECD9", "#E4EEEA"} <= stage_fills, (
        f"missing functional category fills; found {sorted(f for f in stage_fills if f)}"
    )

    print(
        "PASS",
        {
            "tool": "diagrams.net 31.1.5",
            "vertices": vertex_count,
            "edges": edge_count,
            "images": len(image_cells),
            "page": [1600, 900],
            "arrow_card_crossings": len(crossings),
            "arrow_arrow_conflicts": len(connector_conflicts),
            "max_main_flow_bends": max(edge_bends[edge_id] for edge_id in straight_edge_ids | {"refined_scene_to_sync"}),
            "geometric_integrity": "verified",
            "drawio": str(DRAWIO),
            "svg": str(SVG),
            "pdf": str(PDF),
            "png": str(PNG),
        },
    )


if __name__ == "__main__":
    validate()
