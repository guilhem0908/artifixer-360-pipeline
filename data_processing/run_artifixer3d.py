#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the ArtiFixer 360 research pipeline by Guilhem Carmouze, 2026.

"""Run ArtiFixer3D distillation and prepare ArtiFixer3D+ inference metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_processing import artifixer3d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene_root",
        type=Path,
        required=True,
        help="Prepared scene root produced by data_processing.prepare_colmap_artifixer_inputs.",
    )
    parser.add_argument(
        "--artifixer_frames_dir",
        type=Path,
        default=None,
        help="ArtiFixer prediction frame directory, usually <run>/<scene>/frames/batch_0000/pred. "
        "Required when --phases includes distill.",
    )
    parser.add_argument(
        "--split_path",
        type=Path,
        default=None,
        help="Prepared split JSON. Defaults to <scene_root>/split.json.",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Scene id to process. Required only when the split contains multiple scenes.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="ArtiFixer3D output root. Defaults to <scene_root>/artifixer3d.",
    )
    parser.add_argument(
        "--artifixer3d_plus_inference_split_path",
        type=Path,
        default=None,
        help="Output reconstructed_colmap split for ArtiFixer3D+ inference. "
        "Defaults to <scene_root>/split_artifixer3d_plus.json.",
    )
    parser.add_argument(
        "--render_trajectory_path",
        type=Path,
        default=None,
        help=(
            "Optional transforms-style trajectory for post-training ArtiFixer3D rendering. "
            "Defaults to the prepared distillation trajectory from the split."
        ),
    )
    parser.add_argument(
        "--base_checkpoint",
        type=Path,
        default=None,
        help="Optional initial 3DGRUT checkpoint to resume from. Defaults to training ArtiFixer3D from scratch.",
    )
    parser.add_argument(
        "--fresh_optimizer_on_resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume Gaussian parameters from --base_checkpoint but initialize fresh optimizer state.",
    )
    parser.add_argument(
        "--geometry_locked",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Guide distillation from --base_checkpoint while freezing Gaussian positions, rotations, scales, "
            "and densities; only appearance is optimized and MCMC geometry edits are disabled."
        ),
    )
    parser.add_argument(
        "--opacity_prune_locked",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resume from --base_checkpoint with positions, rotations, scales and appearance frozen; "
            "only density is optimized and each Gaussian density is constrained not to exceed its resume value."
        ),
    )
    parser.add_argument(
        "--density_learning_rate",
        type=float,
        default=0.005,
        help="Density learning rate used by --opacity_prune_locked.",
    )
    parser.add_argument(
        "--resume_topology_start_iteration",
        type=int,
        default=None,
        help=(
            "Absolute resumed global step after which MCMC topology edits are re-enabled. "
            "Requires --base_checkpoint and both topology end-iteration arguments."
        ),
    )
    parser.add_argument(
        "--resume_topology_add_relocate_end_iteration",
        type=int,
        default=None,
        help="Exclusive global-step bound for resumed MCMC add/relocate operations.",
    )
    parser.add_argument(
        "--resume_topology_perturb_end_iteration",
        type=int,
        default=None,
        help="Exclusive global-step bound for resumed MCMC position perturbations.",
    )
    parser.add_argument(
        "--resume_topology_max_gaussians",
        type=int,
        default=1_000_000,
        help="Maximum Gaussian count allowed while the resumed topology window is active.",
    )
    parser.add_argument(
        "--depth_guided",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Constrain override-view geometry with <frame>_depth.npy targets stored beside the "
            "ArtiFixer frames. This closes angular and temporal loops against one shared depth field."
        ),
    )
    parser.add_argument(
        "--depth_loss_weight",
        type=float,
        default=0.5,
        help="Weight of the relative log-distance loss used by --depth_guided.",
    )
    parser.add_argument(
        "--single_surface_guided",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Penalize ray-depth changes between an early-termination render and the full render. "
            "Requires a real-geometry --surface_depth_manifest and is intended to remove translucent duplicate layers."
        ),
    )
    parser.add_argument(
        "--single_surface_loss_weight",
        type=float,
        default=0.05,
        help="Weight of the differentiable early/full ray-depth spread loss.",
    )
    parser.add_argument(
        "--single_surface_front_transmittance",
        type=float,
        default=0.5,
        help="Early-stop transmittance used to expose the front surface for the single-surface loss.",
    )
    parser.add_argument(
        "--surface_depth_manifest",
        type=Path,
        default=None,
        help=(
            "Immutable artifixer.surface_depth_manifest_v1 JSON made from real-image geometry. "
            "Base-3DGS-derived depth is rejected."
        ),
    )
    parser.add_argument(
        "--geometry_authorization",
        type=Path,
        default=None,
        help="Immutable PASS_GEOMETRY_GATE contract required by --single_surface_guided.",
    )
    parser.add_argument(
        "--color_loop_guided",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable sparse depth-corresponded RGB consistency targets on override views.",
    )
    parser.add_argument(
        "--color_loop_loss_weight",
        type=float,
        default=0.05,
        help="Weight of the masked RGB loop-consistency loss.",
    )
    parser.add_argument(
        "--balanced_real_override_sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Alternate exactly one real anchor and one ArtiFixer override during 3DGRUT distillation.",
    )
    parser.add_argument(
        "--real_samples_per_override",
        type=int,
        default=1,
        help=(
            "Number of real anchor views sampled before each ArtiFixer override when "
            "--balanced_real_override_sampling is enabled. Values above one strengthen geometry anchoring."
        ),
    )
    parser.add_argument(
        "--scale_regularization_weight",
        type=float,
        default=0.0,
        help="Optional Gaussian-scale penalty used during geometry refinement; zero disables it.",
    )
    parser.add_argument(
        "--opacity_regularization_weight",
        type=float,
        default=0.0,
        help="Optional Gaussian-opacity penalty used during geometry refinement; zero disables it.",
    )
    parser.add_argument(
        "--override_reconstruction_weight",
        type=float,
        default=None,
        help="Optional L1/SSIM weight for ArtiFixer override views; omitted keeps the selected config default.",
    )
    parser.add_argument(
        "--override_lpips_weight",
        type=float,
        default=None,
        help="Optional LPIPS weight for ArtiFixer override views; omitted keeps the selected config default.",
    )
    parser.add_argument(
        "--force_sh0_on_resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Zero/freeze resumed higher-order SH features for the controlled view-independent A/B run.",
    )
    parser.add_argument(
        "--artifixer3d_steps",
        type=int,
        default=30000,
        help="3DGRUT training iterations for the ArtiFixer3D checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_iterations",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional intermediate 3DGRUT checkpoint iterations produced during the same "
            "continuous optimization. The final --artifixer3d_steps checkpoint is always included."
        ),
    )
    parser.add_argument(
        "--config_name",
        default="apps/colmap_3dgut_sparse_mcmc_lpips",
        help="3DGRUT config used for ArtiFixer3D pseudo-supervised distillation.",
    )
    parser.add_argument(
        "--phases",
        default="distill,render,prepare_artifixer3d_plus",
        help="Comma-separated phases to run. Valid phases: distill, render, prepare_artifixer3d_plus.",
    )
    parser.add_argument(
        "--require_geometry_qc_for_plus",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refuse ArtiFixer3D+ split publication unless --geometry_qc_report is a PASS_SINGLE_SURFACE report.",
    )
    parser.add_argument(
        "--geometry_qc_report",
        type=Path,
        default=None,
        help="artifixer.phantom_qc_v1 JSON used by the fail-closed ArtiFixer3D+ publication gate.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        default=False,
        help="Regenerate ArtiFixer3D outputs even when existing outputs are present.",
    )
    parser.add_argument(
        "--use_wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weights & Biases logging for the 3DGRUT distillation run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifixer3d.run_artifixer3d(args)


if __name__ == "__main__":
    main()
