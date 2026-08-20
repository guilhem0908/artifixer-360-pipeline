# Handover guide

This guide is intended for a new researcher taking over the framework.

The dated, experiment-level record with the operational 117-frame protocol,
ABCI runtime hashes, VEnhancer post-processing, failure modes and off-trajectory
results is [artifixer_abci_handover_2026-08-20.md](artifixer_abci_handover_2026-08-20.md).

## 1. Clone and validate

```bash
git clone --recurse-submodules \
  https://github.com/guilhem0908/artifixer-360-pipeline.git
cd artifixer-360-pipeline
python scripts/validate_artifixer360_install.py
```

The validator must report the 3DGRUT submodule, Python entry points and system
tools. A missing model checkpoint is a warning unless `--checkpoint` is given.

## 2. Build the environment

Use `Dockerfile.cuda12` on the current ABCI CUDA-12 nodes. Mount the repository,
working data, model cache and output root into the container. Always set
`PYTHONPATH` to the repository root.

## 3. Prepare one scene

The scene must already have COLMAP `images/` and `sparse/0/`. Run the upstream
preparation command from the README. Before continuing, verify:

- every intended source image is registered;
- the source-camera render is recognizable and not dominated by phantoms;
- metric scale is finite and positive;
- the base 3DGRUT checkpoint and `split.json` exist.

Do not diagnose an ArtiFixer failure using a broken initial reconstruction.

## 4. Choose one experimental branch

### Snake branch

Use [SNAKE_PIPELINE.md](SNAKE_PIPELINE.md) when studying whether small,
continuous perspective rotations are easier for the video model than six large
cubemap jumps.

### Synchronized multi-view branch

Use [DEPTH_LOOP_PIPELINE.md](DEPTH_LOOP_PIPELINE.md) when studying explicit
communication between directions at the same camera position.

Do not mix their metrics or claim that they are the same inference mechanism.

## 5. Required outputs for every experiment

Keep the following together:

```text
run/
  config or command transcript
  source and trajectory manifests
  raw perspective renders
  ArtiFixer predictions
  synchronization metadata (if applicable)
  ArtiFixer3D checkpoint and geometry-lock audit
  ERP PNG frames and MP4
  depth-overlap, temporal and seam QC JSON
  SHA256SUMS
```

Never compare frames from different source indices or camera poses.

## 6. Minimal smoke test

Use 3–5 adjacent source positions and reduce inference steps. The smoke test is
only technical. It does not authorize a full run unless:

- all expected frames are produced;
- the view order matches the rig manifest;
- the horizontal loop pair is present when requested;
- no real anchor is overwritten;
- the geometry-lock audit passes for a locked distillation.

## 7. Known failure modes

- **Black holes:** missing geometry or opacity in the base reconstruction.
- **Double/phantom structures:** inconsistent geometry, poses or incompatible
  generated pseudo-observations.
- **Cubemap edge breaks:** perspective feature sharing without an explicit 3D
  correspondence or spherical topology.
- **Stable but blurry video:** temporal metrics can improve by smoothing;
  always check edge retention.
- **Good pseudo-views, bad final ERP:** distillation did not preserve the
  synchronized constraints.
- **Different results after resumption:** check optimizer state, topology
  schedule, random seed and checkpoint provenance.

## 8. Teaching session outline

1. Explain the input/output contract and pipeline diagram.
2. Run the installation validator.
3. Generate a small snake and inspect its manifest.
4. Run or inspect a synchronized 14-view window.
5. Show the geometry-lock audit and depth-overlap report.
6. Stitch perspective views to ERP.
7. Reproduce one metric table from JSON, not from screenshots.

## 9. Included operational programs

The repository also includes the fail-closed ABCI preflight, the all-frame
COLMAP driver, the validated 117-frame synchronized pilot, the complete snake
job pair, the position-first comparison and the VEnhancer yaw-ensemble helper.
The PBS files deliberately require `AFROOT` and do not contain a username,
allocation or group ID.
