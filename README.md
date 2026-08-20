<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
SPDX-License-Identifier: Apache-2.0
-->

# ArtiFixer 360 Pipeline

Experimental research extension of
[NVIDIA ArtiFixer](https://github.com/nv-tlabs/ArtiFixer) for turning a posed
perspective video into a repaired 360° equirectangular video through shared
perspective-view processing, depth-aware loop constraints and 3D distillation.

> **Research status:** the code and experiment recipes are reproducible, but
> the current method does not guarantee seam-free or geometrically correct
> panoramas. The public release includes the negative-result diagnostics as
> well as the working components. See [Limitations](#limitations).

## What is released

- COLMAP/3DGRUT preparation inherited from upstream ArtiFixer.
- A world-locked 14-view perspective rig with horizontal, upward and downward
  overlap.
- A continuous spherical **snake trajectory** with small angular connectors.
- Position/view ordering and bounded inference manifests (77-frame maximum by
  default).
- Standard ArtiFixer2D inference over the snake sequence.
- Joint multi-view inference with peer features, depth-aware noise consensus
  and an explicit `H300 ↔ H000` horizontal loop.
- Canonical depth/color-loop target construction.
- Geometry-locked ArtiFixer3D distillation and ArtiFixer3D+.
- Perspective-frustum to ERP projection, angular ownership, feathering,
  graph-cut variants and temporal label stabilization.
- Quantitative checks for depth overlap, loop turns, temporal warp, ERP seams,
  edge preservation and geometry-lock integrity.

No source videos, private datasets, generated results, checkpoints, credentials
or cluster-specific paths are included.

## Framework overview

```text
Perspective video
      │
      ▼
COLMAP cameras + initial 3DGRUT scene
      │
      ├───────────────┐
      ▼               ▼
Spherical snake   World-locked 14-view rig
(small turns)     (joint depth/loop windows)
      │               │
      ▼               ▼
ArtiFixer2D       Synchronized ArtiFixer2D
      └───────┬───────┘
              ▼
Canonical depth/color-loop targets
              │
              ▼
Geometry-locked ArtiFixer3D distillation
              │
              ▼
Render the 14 perspective directions
              │
              ▼
Direct perspective → ERP projection
              │
              ▼
Optional ArtiFixer3D+ → final ERP video
```

The model never receives an ERP as an ordinary planar image in the snake
branch. It receives an ordered sequence of perspective frames. The 3D and
spherical constraints are supplied by the trajectory, depth reprojection,
loop graph, distillation targets and ERP renderer.

## Installation

Clone recursively because the 3DGRUT fork contains required distillation
changes:

```bash
git clone --recurse-submodules \
  https://github.com/guilhem0908/artifixer-360-pipeline.git
cd artifixer-360-pipeline
```

Build one of the upstream CUDA containers:

```bash
docker build -f Dockerfile.cuda12 -t artifixer360:cuda12 .
```

Download the public ArtiFixer checkpoint:

```bash
huggingface-cli download nvidia/ArtiFixer artifixer-14b.pt \
  --local-dir /data/checkpoints/artifixer
```

Validate the checkout without running inference:

```bash
python scripts/validate_artifixer360_install.py \
  --checkpoint /data/checkpoints/artifixer/artifixer-14b.pt
```

## Inputs

The first reconstruction stage expects:

```text
scene/
  images/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

Prepare the initial scene as documented by upstream ArtiFixer:

```bash
python -m data_processing.prepare_colmap_artifixer_inputs \
  --colmap_dir /data/scene \
  --output_root /data/prepared/base_scene \
  --reconstruction_steps 30000
```

The panorama extension then reuses the prepared transforms, selected real
anchors, metric scale and 3DGRUT checkpoint. It does not require the example
scene used during development.

## Run the pipeline

The exact commands, expected files and validation gates are in:

- [Snake pipeline](docs/SNAKE_PIPELINE.md) — perspective winding trajectory,
  ArtiFixer2D, canonical targets, distillation, ArtiFixer3D+ and ERP output.
- [Synchronized multi-view pipeline](docs/DEPTH_LOOP_PIPELINE.md) — joint
  14-view windows, depth-aware consensus and horizontal loop closure.
- [ABCI execution](docs/ABCI.md) — portable PBS template and four-GPU launch.
- [Handover guide](docs/HANDOVER.md) — installation, smoke test, data contract,
  expected outputs and troubleshooting.
- [Dated ABCI handover](docs/artifixer_abci_handover_2026-08-20.md) — exact
  validated run, runtime pins, off-trajectory findings and recovery status.

Start from the portable variable template:

```bash
cp configs/artifixer360.example.env configs/artifixer360.env
```

All examples use environment variables instead of hard-coded usernames,
groups, scene names or storage roots.

## Evaluation

Use several complementary measurements; no single RGB score proves correct
geometry:

| Metric | Meaning | Direction |
|---|---|---|
| Depth-overlap MAE | Relative depth disagreement after cross-view reprojection | lower |
| Loop-pair error | Agreement of the final and initial horizontal directions | lower |
| Circular temporal warp MAE | Frame-to-frame ERP stability after optical flow | lower |
| ERP seam ratio | Wrap discontinuity relative to adjacent columns | lower |
| Edge-strength ratio | Structural detail retained from high-confidence input | closer to 1 |
| Geometry-lock audit | Bitwise equality of position/rotation/scale/density | must pass |

Example:

```bash
python scripts/evaluate_depth_synchronized_multiview.py --help
python scripts/evaluate_temporal_panorama.py --help
python scripts/audit_artifixer3d_checkpoint_lock.py --help
```

## Repository layout

```text
data_processing/   COLMAP/3DGRUT preparation and ArtiFixer3D distillation
model_eval/        standard and synchronized ArtiFixer inference
model_training/    upstream model plus multi-view feature plumbing
scripts/           trajectory, synchronization, stitching and QC tools
configs/           portable environment template
docs/              end-to-end and handover documentation
tests/             focused unit tests for the 360 extension
thirdparty/        the linked 3DGRUT-ArtiFixer-360 fork
```

## Limitations

- Temporal feature sharing is not equivalent to sharing a persistent 3D
  surface.
- A perspective-video model does not learn spherical wrap or pole topology by
  default.
- The snake makes rotations less out-of-distribution, but cannot enforce
  bidirectional loop closure by itself.
- Depth/loop conditioning improved synchronized pseudo-views in our diagnostic
  experiment, but the following 3D distillation did not consistently preserve
  that improvement in the final ERP.
- Hidden regions remain generative hypotheses, not captured photographic
  truth.

These limitations are part of the release because they determine which future
experiments are scientifically meaningful.

## Upstream attribution and license

This repository is a derivative of the official NVIDIA ArtiFixer codebase.
Original source files retain their NVIDIA copyright and SPDX notices. Modified
files are documented in [MODIFICATIONS.md](MODIFICATIONS.md). The project is
distributed under Apache‑2.0; see [LICENSE](LICENSE) and
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

Please cite the original ArtiFixer work:

```bibtex
@inproceedings{delutio2026artifixer,
  title={ArtiFixer: Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models},
  author={de Lutio, Riccardo and Fischer, Tobias and Chang, Yen-Yu and Zhang, Yuxuan and
          Wu, Jay Zhangjie and Ren, Xuanchi and Shen, Tianchang and Tothova, Katarina and
          Gojcic, Zan and Turki, Haithem},
  booktitle={SIGGRAPH},
  year={2026}
}
```

## Maintainer

Experimental 360° extension and handover: Guilhem Carmouze.
