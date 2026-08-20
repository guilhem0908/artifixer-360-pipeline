# Scientific framework diagram — diagrams.net source

## Tool choice

The canonical figure is authored in **diagrams.net / draw.io Desktop 31.1.5** as native mxGraph objects.

Why this tool was selected for this figure:

- orthogonal connectors and explicit arrowheads remain clear and independently editable;
- dashed rounded containers are first-class vector objects;
- process stages can use categorical background fills without turning the entire figure into a raster image;
- input and output can be kept outside the global internal-pipeline boundary;
- the `.drawio` source is editable in the browser or desktop application;
- the same source exports deterministically to SVG, PDF, and PNG through the local CLI.

Alternatives considered:

- **Graphviz**: excellent automatic graph layout and routing, but less direct manual control over this paper-style mixed layout and image placement;
- **TikZ/PGF**: publication-quality and reproducible, but slower for supervisor-driven visual revisions and less approachable for non-LaTeX collaborators;
- **Mermaid**: fast for simple flows, but too restrictive for the nested dashed boundary, image frames, feedback loop, and precise paper composition;
- **PowerPoint/Google Slides shapes**: adequate as a delivery surface, but not the canonical authoring tool for this diagram.

Official documentation used for the selection:

- diagrams.net desktop repository and CLI: https://github.com/jgraph/drawio-desktop
- diagrams.net export documentation: https://www.drawio.com/doc/faq/export-diagram
- diagrams.net embedded diagrams: https://www.drawio.com/doc/faq/embed-html-options
- Graphviz attributes: https://graphviz.org/doc/info/attrs.html
- Graphviz cluster gallery: https://graphviz.org/Gallery/directed/cluster.html
- PGF/TikZ manual: https://tikz.dev/

## Generated artifacts

- `generated/artifixer360_framework.drawio` — canonical editable source
- `generated/artifixer360_framework.svg` — vector export with embedded diagram data
- `generated/artifixer360_framework.pdf` — 16:9 vector PDF
- `generated/artifixer360_framework.png` — rendered review image

## Rebuild

```bash
./render_drawio.sh
./validate_drawio_framework.py
```

The local renderer bootstraps the pinned diagrams.net AppImage and an Xvfb
binary into `/dev/shm/artifixer-drawio-31.1.5` on first use, then reuses that
cache. The repository therefore keeps only source, scripts, and generated
artifacts. No ABCI job or GPU inference is involved.

## Current visual contract

- input and output are outside one dashed global internal-pipeline rectangle;
- the global rectangle contains exactly three persistent sub-region rectangles:
  capture/geometry, render–restore–distill, and panoramic synthesis;
- the three sub-region rectangles share one top/bottom alignment grid and use
  restrained corner radii;
- input and output use identical card dimensions and the same vertical center;
- all internal stages sit inside their corresponding sub-region;
- blue = observations / geometry, orange = restoration, green = distillation / 3D feedback;
- the central `Shared 3D Gaussian Scene → renders → ArtiFixer → pseudo-observations → distillation → shared scene` loop is explicit;
- real-image anchors are shown as a fine dashed branch;
- panoramic synthesis uses synchronized overlapping pinhole views, not a cubemap-to-ERP conversion;
- the principal input → output row is straight and horizontally aligned;
- all connectors are orthogonal, square-cornered, and avoid process cards,
  title/body text boxes, and one another;
- all principal-flow connectors are straight except the single lower hand-off
  from the refined shared scene to panoramic synthesis, which is limited to two
  bends.

## Uncertainty

The current PowerPoint/Google Slides deck remains a separately editable delivery artifact. The `.drawio`/SVG/PDF figure is now the clean canonical diagram source. If the final deck must visually embed this exact figure, the recommended next step is to place the SVG on the slide while keeping the `.drawio` source beside the deck; importing arbitrary SVG sub-elements into Google Slides as independently editable native shapes is not reliably supported.
