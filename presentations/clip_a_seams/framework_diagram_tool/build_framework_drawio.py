#!/usr/bin/env python3
"""Generate the editable diagrams.net source for the scientific framework figure.

The diagram is authored as native mxGraph objects rather than as a slide.  The
source .drawio file is the canonical editable artifact; SVG/PDF/PNG exports are
produced by render_drawio.sh.
"""

from __future__ import annotations

import base64
import html
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = ROOT / "framework_figure_20260731" / "assets"
GENERATED = HERE / "generated"
DRAWIO = GENERATED / "artifixer360_framework.drawio"

PAGE_W = 1600
PAGE_H = 900
FONT = "Noto Sans"

COLORS = {
    "paper": "#F1F1EF",
    "ink": "#303537",
    "muted": "#71787B",
    "line": "#AEB5B8",
    "dash": "#8F989C",
    "blue": "#4D83AF",
    "blue_fill": "#E5EEF6",
    "blue_soft": "#F0F5F8",
    "orange": "#C58435",
    "orange_fill": "#F7ECD9",
    "green": "#538875",
    "green_fill": "#E4EEEA",
    "capture_group": "#EDF3F7",
    "loop_group": "#EEF3F0",
    "synthesis_group": "#EDF3F7",
    "white": "#FFFFFF",
}


def image_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
    # mxGraph style strings are semicolon-delimited. Encoding the semicolon in
    # the data-URI header prevents the image value from being split into two
    # style tokens when diagrams.net parses the source file.
    return f"data:{mime}%3Bbase64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


class Diagram:
    def __init__(self) -> None:
        self.mxfile = ET.Element(
            "mxfile",
            {
                "host": "Electron",
                "modified": "2026-08-03T00:00:00.000Z",
                "agent": "ArtiFixer framework generator",
                "version": "31.1.5",
                "type": "device",
            },
        )
        diagram = ET.SubElement(self.mxfile, "diagram", {"id": uuid.uuid4().hex[:12], "name": "Framework"})
        self.model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": "1600",
                "dy": "900",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(PAGE_W),
                "pageHeight": str(PAGE_H),
                "math": "0",
                "shadow": "0",
                "background": COLORS["paper"],
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.ids: set[str] = {"0", "1"}

    def _id(self, value: str) -> str:
        if value in self.ids:
            raise ValueError(f"duplicate mxCell id: {value}")
        self.ids.add(value)
        return value

    def vertex(
        self,
        cell_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        value: str = "",
        style: str,
        parent: str = "1",
    ) -> str:
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": self._id(cell_id),
                "value": value,
                "style": style,
                "vertex": "1",
                "parent": parent,
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
        )
        return cell_id

    def text(
        self,
        cell_id: str,
        value: str,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        size: int,
        color: str = COLORS["ink"],
        bold: bool = False,
        align: str = "center",
        valign: str = "middle",
        fill: str = "none",
        stroke: str = "none",
        rotation: float | None = None,
    ) -> str:
        style = (
            "text;html=0;whiteSpace=wrap;overflow=hidden;"
            f"align={align};verticalAlign={valign};fontFamily={FONT};fontSize={size};"
            f"fontColor={color};fontStyle={1 if bold else 0};spacing=0;"
            f"fillColor={fill};strokeColor={stroke};"
        )
        if rotation is not None:
            style += f"rotation={rotation};"
        return self.vertex(cell_id, x, y, w, h, value=value, style=style)

    def box(
        self,
        cell_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str,
        stroke: str,
        stroke_width: float = 1.8,
        rounded: bool = True,
        arc_size: float = 12,
        dashed: bool = False,
        dash_pattern: str = "8 6",
        opacity: int = 100,
    ) -> str:
        style = (
            f"rounded={1 if rounded else 0};arcSize={arc_size};whiteSpace=wrap;html=0;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
            f"opacity={opacity};shadow=0;glass=0;"
        )
        if dashed:
            style += f"dashed=1;dashPattern={dash_pattern};"
        return self.vertex(cell_id, x, y, w, h, style=style)

    def image(self, cell_id: str, uri: str, x: float, y: float, w: float, h: float) -> str:
        style = (
            "shape=image;verticalLabelPosition=bottom;verticalAlign=top;"
            "imageAspect=0;aspect=fixed;html=0;strokeColor=none;fillColor=none;"
            f"image={uri};"
        )
        return self.vertex(cell_id, x, y, w, h, style=style)

    def ellipse(self, cell_id: str, x: float, y: float, diameter: float, *, fill: str, opacity: int = 100) -> str:
        style = (
            "ellipse;whiteSpace=wrap;html=0;strokeColor=none;"
            f"fillColor={fill};opacity={opacity};shadow=0;"
        )
        return self.vertex(cell_id, x, y, diameter, diameter, style=style)

    def edge(
        self,
        cell_id: str,
        source: str,
        target: str,
        *,
        color: str,
        width: float = 1.55,
        dashed: bool = False,
        points: list[tuple[float, float]] | None = None,
        exit_xy: tuple[float, float] = (1.0, 0.5),
        entry_xy: tuple[float, float] = (0.0, 0.5),
        arrow: bool = True,
    ) -> str:
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
            f"html=0;strokeColor={color};strokeWidth={width};"
            f"endArrow={'block' if arrow else 'none'};endFill={1 if arrow else 0};endSize=7;"
            f"exitX={exit_xy[0]};exitY={exit_xy[1]};exitDx=0;exitDy=0;"
            f"entryX={entry_xy[0]};entryY={entry_xy[1]};entryDx=0;entryDy=0;"
        )
        if dashed:
            style += "dashed=1;dashPattern=7 5;"
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": self._id(cell_id),
                "style": style,
                "edge": "1",
                "parent": "1",
                "source": source,
                "target": target,
            },
        )
        geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if points:
            array = ET.SubElement(geom, "Array", {"as": "points"})
            for x, y in points:
                ET.SubElement(array, "mxPoint", {"x": str(x), "y": str(y)})
        return cell_id

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.indent(self.mxfile, space="  ")
        path.write_bytes(ET.tostring(self.mxfile, encoding="utf-8", xml_declaration=True))


def card(
    d: Diagram,
    cell_id: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    body: str,
    fill: str,
    stroke: str,
    title_size: int = 15,
    body_size: int = 11,
    title_height: float = 42,
    dashed: bool = False,
    center_title: bool = False,
) -> str:
    d.box(
        cell_id,
        x,
        y,
        w,
        h,
        fill=fill,
        stroke=stroke,
        stroke_width=1.25,
        dashed=dashed,
        dash_pattern="4 4",
    )
    title_y = y + 10 if center_title else y + 8
    title_h = h - 20 if center_title else title_height
    d.text(f"{cell_id}_title", title, x + 10, title_y, w - 20, title_h, size=title_size, color=COLORS["ink"], bold=True)
    d.text(
        f"{cell_id}_body",
        body,
        x + 12,
        y + title_height + 12,
        w - 24,
        h - title_height - 20,
        size=body_size,
        color=COLORS["muted"],
    )
    return cell_id


def build() -> Path:
    d = Diagram()

    # Title and the four-container hierarchy used by publication framework
    # figures: one global dashed boundary plus three categorical sub-regions.
    # Keep an invisible title cell for stable editability/IDs, but leave the
    # top margin visually empty as requested. The complete figure footprint is
    # centered on the 16:9 page by the global boundary below.
    d.text("title", "", 235, 22, 1140, 48, size=25, color=COLORS["ink"])
    d.box(
        "pipeline_boundary",
        220,
        88,
        1160,
        724,
        fill="none",
        stroke=COLORS["dash"],
        stroke_width=1.35,
        rounded=True,
        arc_size=8,
        dashed=True,
        dash_pattern="6 6",
    )
    d.text("pipeline_label", "", 650, 74, 300, 24, size=12, color=COLORS["muted"])

    # The three internal regions follow one shared alignment grid.  Rounded,
    # content-sized containers mirror the compact phase boxes used in paper
    # framework figures without introducing decorative chrome.
    d.box(
        "capture_group",
        238,
        112,
        214,
        674,
        fill=COLORS["capture_group"],
        stroke=COLORS["blue"],
        stroke_width=1.0,
        rounded=True,
        arc_size=10,
    )
    d.box(
        "loop_group",
        470,
        112,
        526,
        674,
        fill=COLORS["loop_group"],
        stroke=COLORS["green"],
        stroke_width=1.0,
        rounded=True,
        arc_size=10,
    )
    d.box(
        "synthesis_group",
        1014,
        112,
        348,
        236,
        fill=COLORS["synthesis_group"],
        stroke=COLORS["blue"],
        stroke_width=1.0,
        rounded=True,
        arc_size=10,
    )

    # Plain in-panel labels reproduce the restrained hierarchy of academic
    # framework figures; the tinted region itself carries the category.
    d.box("capture_header", 238, 112, 214, 42, fill="none", stroke="none", rounded=False)
    d.text("capture_header_text", "Capture & geometry", 246, 119, 198, 28, size=12, color=COLORS["blue"], bold=True)
    d.box("loop_header", 470, 112, 526, 42, fill="none", stroke="none", rounded=False)
    d.text("loop_header_text", "Render–restore–distill loop", 478, 119, 510, 28, size=12, color=COLORS["green"], bold=True)
    d.box("synthesis_header", 1014, 112, 348, 42, fill="none", stroke="none", rounded=False)
    d.text("synthesis_header_text", "Panoramic synthesis", 1022, 119, 332, 28, size=12, color=COLORS["blue"], bold=True)

    # Input, the first processing stage, final fusion, and output share one
    # exact horizontal axis. The equal-sized outer cards form a balanced pair.
    d.box("input_frame", 20, 144, 180, 224, fill="none", stroke="none", stroke_width=1.0)
    d.box("input_image_border", 31, 155, 158, 90.6, fill=COLORS["white"], stroke=COLORS["line"], stroke_width=1.0, rounded=False)
    d.image("input_image", image_uri(ASSETS / "input_video_frame025.png"), 33, 157, 154, 86.6)
    d.text("input_title", "Pinhole input video", 33, 253, 154, 34, size=15, color=COLORS["ink"], bold=True)
    d.text("input_body", "", 33, 291, 154, 40, size=10, color=COLORS["muted"])

    d.box("output_frame", 1400, 144, 180, 224, fill="none", stroke="none", stroke_width=1.0)
    d.box("output_image_border", 1411, 157, 158, 81, fill=COLORS["white"], stroke=COLORS["line"], stroke_width=1.0, rounded=False)
    d.image("output_image", image_uri(ASSETS / "output_erp_frame025.png"), 1413, 159, 154, 77)
    d.text("output_title", "Equirectangular 360°\nvideo (ERP)", 1413, 245, 154, 50, size=13, color=COLORS["blue"], bold=True)
    d.text("output_body", "", 1413, 297, 154, 40, size=9, color=COLORS["muted"])

    # Capture and initialization stages.
    card(
        d,
        "frame_extraction",
        250,
        214,
        190,
        84,
        title="Frame extraction",
        body="",
        fill=COLORS["blue_fill"],
        stroke=COLORS["blue"],
        title_size=15,
        body_size=10,
        title_height=34,
        center_title=True,
    )
    card(
        d,
        "colmap",
        250,
        326,
        190,
        108,
        title="COLMAP",
        body="",
        fill=COLORS["blue_fill"],
        stroke=COLORS["blue"],
        title_size=18,
        body_size=11,
        title_height=38,
        center_title=True,
    )
    card(
        d,
        "posed_real",
        250,
        458,
        190,
        92,
        title="Posed real frames",
        body="",
        fill=COLORS["blue_soft"],
        stroke=COLORS["blue"],
        title_size=14,
        body_size=10,
        title_height=36,
        center_title=True,
    )
    card(
        d,
        "initial_scene",
        250,
        575,
        190,
        142,
        title="Initial 3DGRUT /\n3D Gaussian Scene",
        body="",
        fill=COLORS["blue_fill"],
        stroke=COLORS["blue"],
        title_size=14,
        body_size=9,
        title_height=55,
    )
    d.text(
        "initial_scene_caption",
        "",
        268,
        639,
        154,
        30,
        size=9,
        color=COLORS["muted"],
    )
    # Native Gaussian dots give the 3D representation a compact visual cue.
    for idx, (x, y, dia, opacity) in enumerate(
        [
            (270, 687, 9, 80),
            (288, 674, 7, 65),
            (306, 695, 11, 90),
            (326, 679, 8, 75),
            (347, 698, 7, 70),
            (365, 675, 10, 90),
            (387, 691, 8, 75),
            (408, 677, 6, 65),
        ],
        start=1,
    ):
        d.ellipse(f"initial_dot_{idx}", x, y, dia, fill=COLORS["blue"], opacity=opacity)

    # Central shared scene and render–restore–distill loop.
    card(
        d,
        "shared_scene",
        486,
        575,
        210,
        142,
        title="Shared 3D\nGaussian Scene",
        body="",
        fill=COLORS["blue_fill"],
        stroke=COLORS["blue"],
        title_size=15,
        body_size=9,
        title_height=54,
    )
    d.text(
        "shared_scene_caption",
        "",
        517,
        637,
        148,
        30,
        size=9,
        color=COLORS["muted"],
    )
    for idx, (x, y, dia, opacity) in enumerate(
        [
            (516, 679, 10, 90),
            (538, 666, 7, 65),
            (556, 687, 12, 90),
            (579, 668, 8, 75),
            (600, 688, 7, 70),
            (620, 665, 11, 90),
            (642, 684, 8, 75),
        ],
        start=1,
    ):
        d.ellipse(f"shared_dot_{idx}", x, y, dia, fill=COLORS["blue"], opacity=opacity)

    card(
        d,
        "snake_renders",
        486,
        185,
        210,
        142,
        title="World-locked overlapping\npinhole renders",
        body="",
        fill=COLORS["blue_soft"],
        stroke=COLORS["blue"],
        title_size=14,
        body_size=9,
        title_height=40,
    )
    # Overlapping pinhole thumbnails; not six cubemap faces.
    # A gentle arc sits in its own vertical band between the title (ends y=254)
    # and the caption (starts y=306); reduced rotation keeps each rotated
    # bounding box clear of the title text above it.
    for idx, (x, y, rot) in enumerate([(505, 243, -10), (541, 239, -4), (579, 239, 4), (617, 243, 10)], start=1):
        style = (
            "rounded=0;whiteSpace=wrap;html=0;fillColor=#FFFFFF;"
            f"strokeColor={COLORS['blue']};strokeWidth=1.2;rotation={rot};shadow=0;"
        )
        d.vertex(f"pinhole_view_{idx}", x, y, 42, 31, style=style)
    d.text(
        "snake_caption",
        "spherical snake",
        489,
        289,
        204,
        24,
        size=9,
        color=COLORS["muted"],
    )

    card(
        d,
        "artifixer",
        712,
        185,
        120,
        142,
        title="ArtiFixer",
        body="video-to-video",
        fill=COLORS["orange_fill"],
        stroke=COLORS["orange"],
        title_size=16,
        body_size=9,
        title_height=42,
    )
    card(
        d,
        "pseudo_observations",
        848,
        185,
        142,
        142,
        title="Restored\npseudo-observations",
        body="",
        fill=COLORS["orange_fill"],
        stroke=COLORS["orange"],
        title_size=12,
        body_size=9,
        title_height=54,
        center_title=True,
    )
    card(
        d,
        "distillation",
        748,
        575,
        242,
        142,
        title="Geometry-constrained\nArtiFixer3D distillation",
        body="geometry fixed · appearance refined",
        fill=COLORS["green_fill"],
        stroke=COLORS["green"],
        title_size=14,
        body_size=10,
        title_height=56,
    )

    # Final panoramic synthesis stages.
    card(
        d,
        "sync_render",
        1024,
        185,
        100,
        142,
        title="Synchronized\npanoramic rendering",
        body="",
        fill=COLORS["blue_fill"],
        stroke=COLORS["blue"],
        title_size=12,
        body_size=9,
        title_height=56,
        center_title=True,
    )
    card(
        d,
        "final_plus",
        1138,
        185,
        100,
        142,
        title="Final ArtiFixer3D+",
        body="",
        fill=COLORS["orange_fill"],
        stroke=COLORS["orange"],
        title_size=12,
        body_size=9,
        title_height=48,
        center_title=True,
    )
    card(
        d,
        "erp_fusion",
        1252,
        185,
        100,
        142,
        title="Spherical fusion /\nERP projection",
        body="",
        fill=COLORS["blue_soft"],
        stroke=COLORS["blue"],
        title_size=11,
        body_size=9,
        title_height=60,
        center_title=True,
    )

    # Main operation flow: thick, orthogonal, unobstructed arrows.
    d.edge(
        "input_to_frames",
        "input_frame",
        "frame_extraction",
        color=COLORS["blue"],
        # Input and the first internal stage share one centerline.
        exit_xy=(1, 0.5),
        entry_xy=(0, 0.5),
    )
    d.edge("frames_to_colmap", "frame_extraction", "colmap", color=COLORS["blue"], exit_xy=(0.5, 1), entry_xy=(0.5, 0))
    d.edge("colmap_to_real", "colmap", "posed_real", color=COLORS["blue"], exit_xy=(0.5, 1), entry_xy=(0.5, 0))
    d.edge("real_to_initial", "posed_real", "initial_scene", color=COLORS["blue"], exit_xy=(0.5, 1), entry_xy=(0.5, 0))
    d.edge("initial_to_shared", "initial_scene", "shared_scene", color=COLORS["blue"])
    d.edge("shared_to_renders", "shared_scene", "snake_renders", color=COLORS["blue"], exit_xy=(0.5, 0), entry_xy=(0.5, 1))
    d.edge("renders_to_artifixer", "snake_renders", "artifixer", color=COLORS["orange"])
    d.edge("artifixer_to_pseudo", "artifixer", "pseudo_observations", color=COLORS["orange"])
    d.edge(
        "pseudo_to_distill",
        "pseudo_observations",
        "distillation",
        color=COLORS["green"],
        exit_xy=(0.5, 1),
        entry_xy=(171 / 242, 0),
    )
    d.edge(
        "distill_feedback",
        "distillation",
        "shared_scene",
        color=COLORS["green"],
        exit_xy=(0, 101 / 142),
        entry_xy=(1, 101 / 142),
    )
    d.text("feedback_label", "3D feedback", 699, 648, 47, 20, size=8, color=COLORS["green"], bold=True, fill=COLORS["loop_group"])

    # Real anchors use one fine conditioning trunk, then split into two
    # independent branches. The right-hand lane sits in the inter-panel gutter
    # rather than visually merging with the green panel border.
    d.vertex(
        "anchor_junction",
        770,
        162,
        4,
        4,
        style="ellipse;html=0;fillColor=none;strokeColor=none;opacity=0;",
    )
    d.edge(
        "anchors_trunk",
        "posed_real",
        "anchor_junction",
        color=COLORS["blue"],
        width=1.0,
        dashed=False,
        points=[(462, 504), (462, 164)],
        exit_xy=(1, 0.5),
        entry_xy=(0.5, 0.5),
        arrow=False,
    )
    d.edge(
        "anchors_to_artifixer",
        "anchor_junction",
        "artifixer",
        color=COLORS["blue"],
        width=1.0,
        dashed=False,
        exit_xy=(0.5, 0.5),
        entry_xy=(0.5, 0),
    )
    d.edge(
        "anchors_to_distill",
        "anchor_junction",
        "distillation",
        color=COLORS["blue"],
        width=1.0,
        dashed=False,
        points=[(1005, 164), (1005, 603.4)],
        exit_xy=(0.5, 0.5),
        entry_xy=(1, 0.2),
    )
    d.text("anchors_label", "real-image anchors", 542, 151, 152, 21, size=10, color=COLORS["blue"], bold=False, fill=COLORS["loop_group"])

    # Once the distillation update has been fed back into the shared scene, its
    # refined state exits through a short, dedicated lane to final rendering.
    # This replaces the former page-wide U-shaped connector.
    d.edge(
        "refined_scene_to_sync",
        "distillation",
        "sync_render",
        color=COLORS["blue"],
        points=[(1074, 646)],
        exit_xy=(1, 0.5),
        entry_xy=(0.5, 1),
    )
    d.text(
        "refined_scene_label",
        "refined shared scene",
        1083,
        455,
        122,
        22,
        size=9,
        color=COLORS["blue"],
        align="left",
        fill=COLORS["paper"],
    )
    d.edge("sync_to_plus", "sync_render", "final_plus", color=COLORS["orange"])
    d.edge("plus_to_fusion", "final_plus", "erp_fusion", color=COLORS["blue"])
    d.edge("fusion_to_output", "erp_fusion", "output_frame", color=COLORS["blue"], exit_xy=(1, 0.5), entry_xy=(0, 0.5))

    d.write(DRAWIO)
    print(DRAWIO)
    return DRAWIO


if __name__ == "__main__":
    build()
