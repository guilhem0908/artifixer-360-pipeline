# Framework pipeline figure — composition provenance

This file records the official project and paper figures consulted for **scientific composition only**. The delivered framework figure is an original diagram built from native, editable diagrams.net objects; no source illustration was copied into it.

## Diagram authoring tool

The clean framework figure is now authored in **diagrams.net / draw.io Desktop
31.1.5**, not directly in PowerPoint or Google Slides. Its canonical editable
source is:

- `framework_diagram_tool/generated/artifixer360_framework.drawio`

Deterministic vector and review exports are generated from that source:

- `framework_diagram_tool/generated/artifixer360_framework.svg`
- `framework_diagram_tool/generated/artifixer360_framework.pdf`
- `framework_diagram_tool/generated/artifixer360_framework.png`

Official tool references:

- diagrams.net desktop repository and CLI:
  https://github.com/jgraph/drawio-desktop
- diagrams.net export documentation:
  https://www.drawio.com/doc/faq/export-diagram
- diagrams.net embedded diagrams:
  https://www.drawio.com/doc/faq/embed-html-options

## Reference figures

- **ArtiFixer: Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models**  
  Official NVIDIA project: https://research.nvidia.com/labs/sil/projects/artifixer/  
  Project method figure: https://research.nvidia.com/labs/sil/projects/artifixer/assets/method.png  
  Used as a composition reference for separating observed views, video restoration, and 3D distillation.

- **Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models** (often referenced internally alongside ArtiFixer3D)  
  arXiv: https://arxiv.org/abs/2503.01774  
  Official NVIDIA project: https://research.nvidia.com/labs/toronto-ai/difix3d/  
  Project pipeline figure: https://research.nvidia.com/labs/toronto-ai/difix3d/assets/pipeline.jpg  
  Used as a reference for the progressive 3D-update loop and the distinct final post-rendering stage.

- **3D Gaussian Splatting for Real-Time Radiance Field Rendering**  
  arXiv: https://arxiv.org/abs/2308.04079  
  Official project: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/  
  Used as a reference for conventional operation-flow arrows, compact Gaussian-scene notation, and the render/optimization feedback loop.

- **Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions**  
  arXiv: https://arxiv.org/abs/2303.12789  
  Official project: https://instruct-nerf2nerf.github.io/  
  Used as a reference for a visually explicit render → edit → update-3D loop.

- **GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting**  
  arXiv: https://arxiv.org/abs/2311.16037  
  Official project: https://buaacyw.github.io/gaussian-editor/  
  Used as an additional reconstruction/editing reference for rendering from a shared 3D Gaussian representation, applying a 2D model, and feeding corrections back through 3D optimization.

## Visual conventions retained

The August 3, 2026 revision follows the composition principles visible in the
official method figures above without reproducing their artwork:

- one clearly delimited global method area;
- three explicit internal sub-regions with persistent background rectangles,
  rather than header bars floating over a single undivided canvas;
- one shared alignment grid for the three regions, with symmetric input/output
  cards on the same horizontal centerline;
- orthogonal connectors with square corners and visible arrowheads;
- straight principal-flow arrows wherever the scientific dependency permits,
  with only the refined-scene-to-synthesis hand-off using a two-bend bus;
- a separated conditioning/anchor route, visually lighter than the main flow;
- a closed render–restore–distill feedback path around one shared 3D scene;
- input and final output shown outside the internal method boundary;
- no connector routed through a process card, label, or image.

## Implemented pipeline sources inspected

The scientific content of the new slide was grounded in the current repository and run artifacts, especially:

- `scripts/abci_roomtour_standard_full_pipeline_4gpu.pbs`
- `scripts/generate_spherical_snake_trajectory.py`
- `scripts/generate_spherical_snake_inference_manifest.py`
- `model_eval/run_inference.py`
- `scripts/build_depth_canonical_targets.py`
- `data_processing/artifixer3d.py`
- `data_processing/run_artifixer3d.py`
- `scripts/extract_spherical_snake_lanes.py`
- `scripts/stitch_artifixer_frustums360.py`
- `scripts/reorient_erp_to_source_camera.py`
- the locked full-pipeline contract and published manifests/QC supplied with the task.
