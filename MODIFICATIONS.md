# Modifications to upstream ArtiFixer

This repository is based on `nv-tlabs/ArtiFixer` at commit `c320752` and is
distributed under the same Apache‑2.0 license.

The 360° research extension adds or changes:

- arbitrary COLMAP trajectory preparation and rendering;
- long bounded inference manifests with real-anchor context;
- world-locked perspective rigs and continuous spherical snake trajectories;
- synchronized peer-view inference and depth-aware latent/noise consensus;
- depth, color-loop, single-surface and geometry-lock distillation controls;
- ERP projection and stitching from overlapping perspective frustums;
- panorama-specific quantitative diagnostics;
- public documentation and handover utilities.

The linked `3DGRUT-ArtiFixer-360` submodule contains the corresponding 3DGRUT
changes. Original NVIDIA notices remain in modified upstream files. Git history
and this document identify the derivative modifications.
