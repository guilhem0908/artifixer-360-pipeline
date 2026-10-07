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
- documentation and handover utilities.

The corresponding 3DGRUT changes are in `3DGRUT-ArtiFixer-360`, a private fork of
NVIDIA's `nv-tlabs/3DGRUT-ArtiFixer` linked as a submodule; cloning it needs access
to that fork. Original NVIDIA notices remain in modified upstream files. Git history
and this document identify the derivative modifications.
