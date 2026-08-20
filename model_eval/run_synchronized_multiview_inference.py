# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint ArtiFixer inference with one closed spherical latent loop.

All frustums are initialized before denoising.  Transformer evaluation is
micro-batched to keep the 14B model within memory, while scheduler updates and
overlap consensus happen globally at every step.  The decoded images are
therefore direct ArtiFixer outputs whose horizontal and vertical overlaps were
constrained throughout generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from model_eval.checkpoint_loading import load_transformer_checkpoint, validate_checkpoint_args
from model_eval.run_inference import (
    barrier_if_distributed,
    build_parser,
    compute_context_parallel_padding,
    compute_frame_padding,
    create_context_parallel_meshes,
    create_dataset,
    get_eval_pipe,
    init_distributed,
    latent_num_frames_from_rgb_num_frames,
    pad_temporal,
    pipeline_frames_per_block,
    save_image,
    validate_evalset_args,
)
from model_eval.synchronized_multiview import (
    build_depth_aware_latent_reprojection_graph,
    build_latent_reprojection_graph,
    perspective_reprojection_grid_rotations,
    synchronize_low_frequency_latent_views,
    synchronize_latent_views,
    validate_reprojection_graph,
)


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--synchronization-blend", type=float, default=0.7)
    parser.add_argument(
        "--initial-noise-consensus-iterations",
        type=int,
        default=0,
        help="Depth-aware consensus iterations applied to noise before opacity mixing.",
    )
    parser.add_argument(
        "--post-step-latent-blend",
        type=float,
        default=0.0,
        help="Additional depth-aware latent consensus after each scheduler step.",
    )
    parser.add_argument(
        "--final-latent-consensus-iterations",
        type=int,
        default=0,
        help="Full depth-aware latent consensus iterations immediately before decoding.",
    )
    parser.add_argument(
        "--final-low-frequency-consensus-iterations",
        type=int,
        default=0,
        help="Depth-aware pre-decode consensus restricted to low spatial frequencies.",
    )
    parser.add_argument(
        "--final-low-frequency-kernel-size",
        type=int,
        default=5,
        help="Odd spatial averaging kernel for low-frequency latent consensus.",
    )
    parser.add_argument(
        "--synchronization-target",
        choices=("latents", "noise_predictions"),
        default="latents",
        help=(
            "Synchronize post-scheduler latents (legacy) or fuse denoiser noise predictions before "
            "the scheduler (MultiDiffusion-style, avoids repeatedly averaging image content)."
        ),
    )
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.02)
    parser.add_argument("--view-microbatch-size", type=int, default=2)
    parser.add_argument(
        "--neighbor-prope-chunk-cameras",
        type=int,
        default=16,
        help=(
            "Camera-block size for the inference-only PRoPE transform of the full peer context. "
            "This bounds temporary float32 memory without dropping or separately attending to features."
        ),
    )
    parser.add_argument(
        "--peer-conditioning",
        action="store_true",
        help="Expose every target view to encoded same-window features from all other rig views.",
    )
    parser.add_argument(
        "--max-peer-views",
        type=int,
        default=None,
        help=(
            "Limit peer conditioning to the strongest geometrically overlapping rig views. "
            "Real-image neighbors are always retained and are not counted in this limit."
        ),
    )
    parser.add_argument(
        "--peer-features-only",
        action="store_true",
        help=(
            "Use the checkpoint's internal PRoPE neighbor cross-attention for cross-view coherence "
            "without any external latent/noise reprojection consensus."
        ),
    )
    parser.add_argument(
        "--deterministic-overlap-noise",
        action="store_true",
        help=(
            "Key initial latent noise by global temporal position and rig view so overlapping "
            "windows reuse exactly the same noise instead of restarting their shared context."
        ),
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        help="Optional view-major radial depth directory used for occlusion-aware synchronization.",
    )
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.08)
    parser.add_argument("--minimum-depth-overlap-fraction", type=float, default=0.002)
    parser.add_argument("--depth-spatial-valid-fraction", type=float, default=0.5)
    parser.add_argument("--depth-temporal-valid-fraction", type=float, default=0.5)
    parser.add_argument(
        "--required-loop-pair",
        action="append",
        default=[],
        metavar="VIEW_A:VIEW_B",
        help="Require both directed depth-aware graph edges for this closure; may be repeated.",
    )
    parser.add_argument(
        "--window-manifest",
        type=Path,
        help="Optional window-major manifest for full clips with overlapping temporal halos.",
    )
    args = parser.parse_args()
    validate_checkpoint_args(parser, args)
    validate_evalset_args(parser, args)
    if args.inference_pipeline != "bidirectional":
        parser.error("synchronized multiview inference requires --inference_pipeline bidirectional")
    if not 0.0 <= args.synchronization_blend <= 1.0:
        parser.error("--synchronization-blend must be in [0, 1]")
    if args.initial_noise_consensus_iterations < 0:
        parser.error("--initial-noise-consensus-iterations must be non-negative")
    if not 0.0 <= args.post_step_latent_blend <= 1.0:
        parser.error("--post-step-latent-blend must be in [0, 1]")
    if args.final_latent_consensus_iterations < 0:
        parser.error("--final-latent-consensus-iterations must be non-negative")
    if args.final_low_frequency_consensus_iterations < 0:
        parser.error("--final-low-frequency-consensus-iterations must be non-negative")
    if args.final_low_frequency_kernel_size < 3 or args.final_low_frequency_kernel_size % 2 == 0:
        parser.error("--final-low-frequency-kernel-size must be an odd integer >= 3")
    if args.final_latent_consensus_iterations and args.final_low_frequency_consensus_iterations:
        parser.error("full and low-frequency final latent consensus are mutually exclusive")
    if not 0.0 <= args.minimum_overlap_fraction < 1.0:
        parser.error("--minimum-overlap-fraction must be in [0, 1)")
    if args.view_microbatch_size <= 0:
        parser.error("--view-microbatch-size must be positive")
    if args.max_peer_views is not None and args.max_peer_views <= 0:
        parser.error("--max-peer-views must be positive when set")
    if args.depth_relative_tolerance < 0:
        parser.error("--depth-relative-tolerance must be non-negative")
    if not 0 <= args.minimum_depth_overlap_fraction < 1:
        parser.error("--minimum-depth-overlap-fraction must be in [0, 1)")
    for value in args.required_loop_pair:
        parts = value.split(":")
        if len(parts) != 2 or not all(parts):
            parser.error(f"--required-loop-pair must be VIEW_A:VIEW_B, got {value!r}")
    if args.depth_dir is None and args.required_loop_pair:
        parser.error("--required-loop-pair requires --depth-dir")
    if args.peer_features_only:
        if not args.peer_conditioning:
            parser.error("--peer-features-only requires --peer-conditioning")
        if args.depth_dir is not None or args.required_loop_pair:
            parser.error("--peer-features-only forbids depth reprojection arguments")
        if (
            args.initial_noise_consensus_iterations
            or args.synchronization_blend
            or args.post_step_latent_blend
            or args.final_latent_consensus_iterations
            or args.final_low_frequency_consensus_iterations
        ):
            parser.error("--peer-features-only requires every external consensus control to be zero")
    return args


def load_rig(manifest_path: Path) -> tuple[list[str], list[list[list[float]]], float, int]:
    manifest = json.loads(manifest_path.read_text())
    names = [str(value) for value in manifest["view_order"]]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("manifest view_order must contain unique views")
    rotations = []
    for name in names:
        matrix = manifest["local_rotations"][name]
        if len(matrix) == 4:
            matrix = [row[:3] for row in matrix[:3]]
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ValueError(f"manifest rotation for {name} must be 3x3 or 4x4")
        rotations.append(matrix)
    fov_degrees = float(manifest["fov_degrees"])
    frames_per_view = int(manifest["frames_per_view"])
    if frames_per_view <= 0:
        raise ValueError("manifest frames_per_view must be positive")
    return names, rotations, fov_degrees, frames_per_view


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous()


def temporal_noise_ids(
    source_indices: list[int],
    *,
    padded_num_frames: int,
    vae_temporal_scale: int,
) -> torch.Tensor:
    """Map a phase-aligned RGB interval to stable global latent identifiers."""
    if not source_indices or any(
        second != first + 1 for first, second in zip(source_indices, source_indices[1:])
    ):
        raise ValueError("source_indices must be one non-empty contiguous temporal interval")
    if source_indices[0] % vae_temporal_scale:
        raise ValueError("source window must start on the VAE temporal phase")
    if padded_num_frames < len(source_indices):
        raise ValueError("padded_num_frames cannot be shorter than source_indices")
    padded = [
        *source_indices,
        *range(source_indices[-1] + 1, source_indices[-1] + 1 + padded_num_frames - len(source_indices)),
    ]
    return torch.tensor(padded[::vae_temporal_scale], dtype=torch.long)


def prepare_item(
    pipe,
    item: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    *,
    source_indices: list[int] | None = None,
    view_index: int = 0,
) -> dict[str, Any]:
    """Encode one view and retain only compact tensors needed for denoising."""
    rgb_rendered = item["rgb_rendered"].unsqueeze(0).to(device)
    original_num_frames = int(rgb_rendered.shape[1])
    vae_temporal_scale = pipe.vae.config.scale_factor_temporal
    pad_frames = compute_frame_padding(
        original_num_frames,
        pipeline_frames_per_block(pipe),
        vae_temporal_scale,
    )
    pad_frames += compute_context_parallel_padding(
        original_num_frames + pad_frames,
        args.context_parallel_size,
        vae_temporal_scale,
    )
    target_num_frames = original_num_frames + pad_frames
    target_latent_num_frames = latent_num_frames_from_rgb_num_frames(target_num_frames, vae_temporal_scale)
    if pad_frames:
        rgb_rendered = pad_temporal(rgb_rendered, pad_frames, dim=1)

    opacity = item["opacity"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    if pad_frames:
        opacity = pad_temporal(opacity, pad_frames, dim=1)
    camera_rays = item["camera_rays"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    w2cs = item["w2cs"].unsqueeze(0).to(device)
    Ks = item["Ks"].unsqueeze(0).to(device)
    latent_pad_frames = target_latent_num_frames - camera_rays.shape[1]
    if latent_pad_frames:
        camera_rays = pad_temporal(camera_rays, latent_pad_frames, dim=1)
        w2cs = pad_temporal(w2cs, latent_pad_frames, dim=1)
        Ks = pad_temporal(Ks, latent_pad_frames, dim=1)

    neighbors = item["rgb_neighbors"].unsqueeze(0).to(device)
    neighbor_w2cs = item["neighbor_w2cs"].unsqueeze(0).to(device)
    neighbor_Ks = item["neighbor_Ks"].unsqueeze(0).to(device)
    prompt_embeds = item["encoded_prompt"].unsqueeze(0).to(device)

    condition = pipe.encode_video_frames(rgb_rendered)
    neighbors_condition = pipe.encode_neighbors(neighbors, args.max_neighbors_per_encode)
    noise_ids = None
    if args.deterministic_overlap_noise:
        if source_indices is None:
            raise ValueError("deterministic overlap noise requires explicit source_indices")
        noise_ids = temporal_noise_ids(
            source_indices,
            padded_num_frames=target_num_frames,
            vae_temporal_scale=vae_temporal_scale,
        ).to(device)
        if noise_ids.numel() != condition.shape[2]:
            raise AssertionError(
                f"derived {noise_ids.numel()} noise ids for {condition.shape[2]} latent frames"
            )
        # Different directions must not receive the same image-plane noise,
        # but the same direction and global time must be identical in every
        # overlapping temporal window.
        noise = pipe.deterministic_noise_like(
            condition,
            noise_ids,
            int(args.deterministic_noise_seed) + 10_000_019 * int(view_index),
        )
    else:
        noise = torch.randn_like(condition)
    latents = pipe.prepare_latents(condition, opacity, True, noise=noise)
    result = {
        "scene_id": str(item["scene_id"]),
        "frame_indices": [int(value) for value in item["frame_indices"]],
        "original_num_frames": original_num_frames,
        "padded_num_frames": target_num_frames,
        "condition": _cpu(condition[0]),
        "initial_noise": _cpu(noise[0]),
        "noise_ids": None if noise_ids is None else _cpu(noise_ids),
        "latents": _cpu(latents[0]),
        "neighbor_hidden_states": _cpu(neighbors_condition[0]),
        "opacity": _cpu(opacity[0]),
        "camera_rays": _cpu(camera_rays[0]),
        "w2cs": _cpu(w2cs[0]),
        "neighbor_w2cs": _cpu(neighbor_w2cs[0]),
        "Ks": _cpu(Ks[0]),
        "neighbor_Ks": _cpu(neighbor_Ks[0]),
        "prompt_embeds": _cpu(prompt_embeds[0]),
    }
    del rgb_rendered, opacity, camera_rays, w2cs, Ks, neighbors
    del neighbor_w2cs, neighbor_Ks, prompt_embeds, condition, neighbors_condition, noise, latents
    torch.cuda.empty_cache()
    return result


def build_peer_contexts(
    records: list[dict[str, Any]],
    rotations: list[list[list[float]]],
    *,
    fov_degrees: float = 110.0,
    max_peer_views: int | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Describe clean target-condition features from every other rig direction.

    Target camera tensors use a per-view reference coordinate system.  Peer
    poses are therefore rebuilt in the target record's reference system using
    the known camera-local rig rotations before PRoPE cross-attention.  The
    large hidden-state tensors remain referenced rather than concatenated here;
    a 77-frame, 14-view window would otherwise duplicate every encoded feature
    thirteen times in host memory.
    """
    if len(records) != len(rotations) or len(records) < 2:
        raise ValueError("records and rotations must describe the same multi-view rig")
    if max_peer_views is not None and max_peer_views <= 0:
        raise ValueError("max_peer_views must be positive when set")
    result = []
    rotation_tensors = [torch.as_tensor(value, dtype=torch.float32)[:3, :3] for value in rotations]
    for target_index, target_record in enumerate(records):
        target_w2cs = target_record["w2cs"].to(torch.float32)
        target_c2ws = torch.linalg.inv(target_w2cs)
        peer_source_indices = []
        peer_w2cs = []
        peer_Ks = []
        peer_candidates: list[tuple[float, int]] = []
        for source_index in range(len(records)):
            if source_index != target_index:
                _, valid = perspective_reprojection_grid_rotations(
                    32,
                    32,
                    fov_degrees,
                    source_rotation=rotation_tensors[source_index],
                    target_rotation=rotation_tensors[target_index],
                    device=torch.device("cpu"),
                )
                overlap = float(valid.to(torch.float32).mean().item())
                if overlap > 0.0:
                    peer_candidates.append((overlap, source_index))
        peer_candidates.sort(key=lambda item: (-item[0], item[1]))
        if max_peer_views is not None:
            peer_candidates = peer_candidates[:max_peer_views]

        peer_overlap_fractions = []
        for overlap, source_index in peer_candidates:
            source_record = records[source_index]
            if source_record["condition"].shape[1] != target_c2ws.shape[0]:
                raise ValueError("peer target conditions must have one shared latent timeline")
            relative = rotation_tensors[target_index].T @ rotation_tensors[source_index]
            source_c2ws = target_c2ws.clone()
            source_c2ws[:, :3, :3] = target_c2ws[:, :3, :3] @ relative
            source_w2cs = torch.linalg.inv(source_c2ws)
            peer_source_indices.append(source_index)
            peer_overlap_fractions.append(overlap)
            peer_w2cs.append(source_w2cs)
            peer_Ks.append(source_record["Ks"].to(torch.float32))

        if not peer_source_indices:
            raise ValueError(f"target view {target_index} has no geometrically overlapping peer")

        real_hidden = target_record["neighbor_hidden_states"]
        real_w2cs = target_record["neighbor_w2cs"].to(torch.float32)
        real_Ks = target_record["neighbor_Ks"].to(torch.float32)
        result.append(
            {
                "peer_source_indices": tuple(peer_source_indices),
                "peer_overlap_fractions": tuple(peer_overlap_fractions),
                "neighbor_w2cs": torch.cat([real_w2cs, *peer_w2cs], dim=0).contiguous(),
                "neighbor_Ks": torch.cat([real_Ks, *peer_Ks], dim=0).contiguous(),
                "real_neighbor_count": torch.tensor(real_hidden.shape[1]),
                "peer_neighbor_count": torch.tensor(
                    sum(records[index]["condition"].shape[1] for index in peer_source_indices)
                ),
            }
        )
    return result


def stack_peer_hidden_states(
    records: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    indices: range,
    device: torch.device,
) -> torch.Tensor:
    """Materialize only the peer feature batch needed by the current denoiser call."""
    batches = []
    for target_index in indices:
        tensors = [records[target_index]["neighbor_hidden_states"]]
        tensors.extend(
            records[source_index]["condition"]
            for source_index in contexts[target_index]["peer_source_indices"]
        )
        batches.append(torch.cat(tensors, dim=1))
    return torch.stack(batches).to(device, non_blocking=True)


def load_depth_window(
    depth_dir: Path,
    *,
    view_count: int,
    frames_per_view: int,
    source_indices: list[int],
    padded_num_frames: int,
) -> torch.Tensor:
    """Load one view-major radial-depth window and repeat-pad its final frame."""
    videos = []
    expected_shape = None
    for view_index in range(view_count):
        frames = []
        for source_index in source_indices:
            if not 0 <= source_index < frames_per_view:
                raise ValueError(f"source frame {source_index} is outside [0, {frames_per_view})")
            path = depth_dir / f"{view_index * frames_per_view + source_index:05d}.npy"
            if not path.is_file():
                raise FileNotFoundError(path)
            value = np.load(path, allow_pickle=False)
            if value.ndim == 3 and value.shape[-1] == 1:
                value = value[..., 0]
            if value.ndim != 2 or not np.issubdtype(value.dtype, np.floating):
                raise ValueError(f"invalid radial depth {path}: shape={value.shape} dtype={value.dtype}")
            if not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"radial depth contains invalid values: {path}")
            expected_shape = expected_shape or value.shape
            if value.shape != expected_shape:
                raise ValueError(f"radial depth shape drift: {path} has {value.shape}, expected {expected_shape}")
            frames.append(torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)))
        video = torch.stack(frames)
        if padded_num_frames < video.shape[0]:
            raise ValueError("padded_num_frames cannot be smaller than the source window")
        if padded_num_frames > video.shape[0]:
            video = torch.cat(
                [video, video[-1:].expand(padded_num_frames - video.shape[0], -1, -1)],
                dim=0,
            )
        videos.append(video)
    return torch.stack(videos)


def stack_records(
    records: list[dict[str, Any]],
    indices: range,
    key: str,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack([records[index][key] for index in indices]).to(device, non_blocking=True)


def save_outputs(
    pipe,
    records: list[dict[str, Any]],
    latents: torch.Tensor,
    view_names: list[str],
    output_root: Path,
    rank: int,
    *,
    source_indices: list[int],
    output_source_indices: list[int],
    frames_per_view: int,
    window_index: int,
) -> None:
    if rank != 0:
        return
    window_root = output_root / "window_predictions" / f"window_{window_index:03d}"
    if window_root.exists():
        raise FileExistsError(f"refusing to overwrite window output: {window_root}")
    for view_index, (record, view_name) in enumerate(zip(records, view_names)):
        decoded = pipe.video_processor.postprocess_video(
            pipe.latents_to_rgb(latents[view_index : view_index + 1]),
            output_type="pt",
        )[:, : record["original_num_frames"]].cpu().float().clamp(0, 1)
        view_dir = window_root / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        if len(source_indices) != record["original_num_frames"]:
            raise ValueError("source_indices do not match decoded window length")
        if len(output_source_indices) != len(source_indices):
            raise ValueError("output_source_indices do not match source_indices")
        for local_index, source_index in enumerate(source_indices):
            frame = decoded[0, local_index]
            view_path = view_dir / f"{source_index:05d}.png"
            if view_path.exists():
                raise FileExistsError(f"refusing duplicate window output: {view_path}")
            save_image(frame, view_path)
        del decoded
        torch.cuda.empty_cache()


def temporal_window_weights(
    windows: list[dict[str, Any]],
    source_index: int,
) -> dict[int, float]:
    """Raised-cosine partition of unity for one temporal source frame.

    The registered windows may overlap pairwise but not three-way.  A shared
    frame therefore receives evidence from both complete 77-frame denoising
    contexts instead of being hard-switched at the old core boundary.
    """
    memberships = [
        window_index
        for window_index, window in enumerate(windows)
        if source_index in window["source_indices"]
    ]
    if not memberships:
        raise ValueError(f"source frame {source_index} is absent from every window")
    if len(memberships) == 1:
        return {memberships[0]: 1.0}
    if len(memberships) != 2:
        raise ValueError("temporal blending supports pairwise overlaps only")
    first_index, second_index = sorted(
        memberships,
        key=lambda index: windows[index]["source_indices"][0],
    )
    overlap = sorted(
        set(windows[first_index]["source_indices"])
        & set(windows[second_index]["source_indices"])
    )
    if not overlap or overlap != list(range(overlap[0], overlap[-1] + 1)):
        raise ValueError("window overlap must be non-empty and temporally contiguous")
    if len(overlap) == 1:
        return {first_index: 0.5, second_index: 0.5}
    if source_index == overlap[0]:
        return {first_index: 1.0, second_index: 0.0}
    if source_index == overlap[-1]:
        return {first_index: 0.0, second_index: 1.0}
    alpha = (source_index - overlap[0]) / (overlap[-1] - overlap[0])
    first_weight = float(np.cos(0.5 * np.pi * alpha) ** 2)
    second_weight = float(np.sin(0.5 * np.pi * alpha) ** 2)
    return {first_index: first_weight, second_index: second_weight}


def merge_window_outputs(
    output_root: Path,
    windows: list[dict[str, Any]],
    view_names: list[str],
    frames_per_view: int,
) -> dict[str, Any]:
    """Fuse every decoded overlap rather than selecting a hard window core."""
    merged = output_root / "merged_predictions"
    if merged.exists():
        raise FileExistsError(f"refusing to overwrite merged output: {merged}")
    merged.mkdir(parents=True)
    for window_index, window in enumerate(windows):
        expected = {
            f"{source_index:05d}.png" for source_index in window["source_indices"]
        }
        for view_name in view_names:
            directory = output_root / "window_predictions" / f"window_{window_index:03d}" / view_name
            actual = {path.name for path in directory.glob("*.png")}
            if actual != expected:
                raise ValueError(
                    f"window {window_index} view {view_name} decoded {len(actual)} frames; "
                    f"expected {len(expected)}"
                )

    windows_with_explicit_outputs = ["output_source_indices" in window for window in windows]
    if any(windows_with_explicit_outputs) and not all(windows_with_explicit_outputs):
        raise ValueError("window output_source_indices must be provided consistently")
    if all(windows_with_explicit_outputs):
        published_source_indices = [
            value
            for window in windows
            for value in window["output_source_indices"]
            if value != -1
        ]
    else:
        published_source_indices = sorted(
            {value for window in windows for value in window["source_indices"]}
        )
    if not published_source_indices or len(set(published_source_indices)) != len(
        published_source_indices
    ):
        raise ValueError("published window outputs must be non-empty and unique")
    if published_source_indices != sorted(published_source_indices):
        raise ValueError("published window outputs must be temporally ordered")
    if any(not 0 <= value < frames_per_view for value in published_source_indices):
        raise ValueError("published window output is outside the full clip")

    overlap_frame_count = 0
    maximum_contributors = 0
    for source_index in published_source_indices:
        weights = temporal_window_weights(windows, source_index)
        maximum_contributors = max(maximum_contributors, len(weights))
        overlap_frame_count += int(len(weights) > 1)
        for view_index, view_name in enumerate(view_names):
            accumulator = None
            expected_shape = None
            for window_index, weight in weights.items():
                path = (
                    output_root
                    / "window_predictions"
                    / f"window_{window_index:03d}"
                    / view_name
                    / f"{source_index:05d}.png"
                )
                with Image.open(path) as image:
                    array = np.asarray(image.convert("RGB"), dtype=np.float32)
                if expected_shape is None:
                    expected_shape = array.shape
                    accumulator = np.zeros_like(array)
                elif array.shape != expected_shape:
                    raise ValueError(f"inconsistent decoded frame shape at {path}")
                accumulator += array * weight
            output = np.rint(np.clip(accumulator, 0, 255)).astype(np.uint8)
            view_dir = output_root / view_name
            view_dir.mkdir(exist_ok=True)
            view_path = view_dir / f"{source_index:05d}.png"
            merged_path = merged / f"{view_index * frames_per_view + source_index:05d}.png"
            if view_path.exists() or merged_path.exists():
                raise FileExistsError(f"refusing duplicate merged output: {view_path} / {merged_path}")
            Image.fromarray(output, mode="RGB").save(view_path)
            Image.fromarray(output, mode="RGB").save(merged_path)
    return {
        "method": "pairwise_raised_cosine_rgb_partition_of_unity",
        "overlap_frame_count": overlap_frame_count,
        "maximum_window_contributors": maximum_contributors,
        "all_window_predictions_retained_until_merge": True,
    }


def load_window_manifest(
    path: Path | None,
    *,
    view_names: list[str],
    frames_per_view: int,
    dataset_size: int,
) -> list[dict[str, Any]]:
    if path is None:
        if dataset_size != len(view_names):
            raise ValueError(f"dataset has {dataset_size} items but rig has {len(view_names)} views")
        return [{"window_index": 0, "scene_ids": None, "source_indices": None, "output_source_indices": None}]
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != 1 or raw.get("view_order") != view_names:
        raise ValueError("window manifest schema or view order does not match the rig")
    if int(raw.get("frames_per_view", -1)) != frames_per_view:
        raise ValueError("window manifest frames_per_view does not match the rig")
    windows = raw.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("window manifest must contain a non-empty windows list")
    if dataset_size != len(windows) * len(view_names):
        raise ValueError(
            f"dataset has {dataset_size} items; expected {len(windows) * len(view_names)} for window manifest"
        )
    published = []
    for expected_index, window in enumerate(windows):
        if int(window.get("window_index", -1)) != expected_index:
            raise ValueError("window indices must be contiguous and ordered")
        scene_ids = window.get("scene_ids")
        sources = window.get("source_indices")
        outputs = window.get("output_source_indices")
        if not isinstance(scene_ids, list) or len(scene_ids) != len(view_names):
            raise ValueError(f"window {expected_index} must provide one scene id per view")
        if not isinstance(sources, list) or not sources or len(set(sources)) != len(sources):
            raise ValueError(f"window {expected_index} source indices must be non-empty and unique")
        if not isinstance(outputs, list) or len(outputs) != len(sources):
            raise ValueError(f"window {expected_index} output indices do not match its source indices")
        if any(not isinstance(value, int) or not 0 <= value < frames_per_view for value in sources):
            raise ValueError(f"window {expected_index} source index is outside the full clip")
        for source, output in zip(sources, outputs):
            if output not in (-1, source):
                raise ValueError(f"window {expected_index} output must be -1 or its source index")
            if output != -1:
                published.append(output)
    expected_published = raw.get("published_source_indices", list(range(frames_per_view)))
    if (
        not isinstance(expected_published, list)
        or not expected_published
        or any(not isinstance(value, int) for value in expected_published)
    ):
        raise ValueError("published_source_indices must be a non-empty integer list")
    if published != expected_published:
        raise ValueError("window cores do not publish the registered source frames exactly once in order")
    return windows


@torch.inference_mode()
def run_multiview_group(
    pipe,
    dataset,
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
    *,
    view_names: list[str],
    rotations: list[list[list[float]]],
    fov_degrees: float,
    frames_per_view: int,
    dataset_offset: int,
    window: dict[str, Any],
) -> dict[str, Any]:
    window_index = int(window["window_index"])
    requested_source_indices = window.get("source_indices")
    if requested_source_indices is not None:
        requested_source_indices = [int(value) for value in requested_source_indices]

    records: list[dict[str, Any]] = []
    for view_index, view_name in enumerate(view_names):
        item = dataset[dataset_offset + view_index]
        scene_id = str(item["scene_id"])
        expected_scene_ids = window.get("scene_ids")
        if expected_scene_ids is not None:
            if scene_id != expected_scene_ids[view_index]:
                raise ValueError(
                    f"dataset item {dataset_offset + view_index} is {scene_id!r}, "
                    f"expected {expected_scene_ids[view_index]!r}"
                )
        elif not scene_id.endswith(f"_{view_name}"):
            raise ValueError(f"dataset item {view_index} is {scene_id!r}, expected suffix _{view_name}")
        records.append(
            prepare_item(
                pipe,
                item,
                args,
                device,
                source_indices=requested_source_indices,
                view_index=view_index,
            )
        )
        if rank == 0:
            print(
                f"SYNC_PREPARED window={window_index} {view_index + 1}/{len(view_names)} view={view_name}",
                flush=True,
            )

    frame_counts = {record["original_num_frames"] for record in records}
    padded_counts = {record["padded_num_frames"] for record in records}
    latent_shapes = {tuple(record["latents"].shape) for record in records}
    if len(frame_counts) != 1 or len(padded_counts) != 1 or len(latent_shapes) != 1:
        raise ValueError(
            f"views must share frame/padded/latent shapes, got frames={frame_counts}, "
            f"padded={padded_counts}, latents={latent_shapes}"
        )
    source_indices = requested_source_indices
    if source_indices is None:
        source_indices = records[0]["frame_indices"]
    source_indices = [int(value) for value in source_indices]
    if any(record["original_num_frames"] != len(source_indices) for record in records):
        raise ValueError("window source indices do not match every view record")
    output_source_indices = window.get("output_source_indices")
    if output_source_indices is None:
        output_source_indices = source_indices
    output_source_indices = [int(value) for value in output_source_indices]

    latents = torch.stack([record["latents"] for record in records]).to(device)
    graph = None
    if args.peer_features_only:
        graph_stats = {
            "policy": "internal_prope_peer_features_only",
            "external_reprojection_edges": 0,
        }
    elif args.depth_dir is not None:
        depths = load_depth_window(
            args.depth_dir,
            view_count=len(view_names),
            frames_per_view=frames_per_view,
            source_indices=source_indices,
            padded_num_frames=next(iter(padded_counts)),
        )
        opacity = torch.stack([record["opacity"].to(torch.float32) for record in records])
        if opacity.shape[-2:] != depths.shape[-2:]:
            opacity = torch.nn.functional.interpolate(
                opacity.flatten(0, 1).unsqueeze(1),
                size=depths.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).view(*opacity.shape[:2], *depths.shape[-2:])
        graph = build_depth_aware_latent_reprojection_graph(
            rotations,
            depths,
            latents.shape[-2],
            latents.shape[-1],
            fov_degrees,
            device=device,
            dtype=latents.dtype,
            opacity=opacity,
            vae_temporal_scale=pipe.vae.config.scale_factor_temporal,
            minimum_overlap_fraction=args.minimum_overlap_fraction,
            minimum_depth_overlap_fraction=args.minimum_depth_overlap_fraction,
            depth_relative_tolerance=args.depth_relative_tolerance,
            spatial_valid_fraction=args.depth_spatial_valid_fraction,
            temporal_valid_fraction=args.depth_temporal_valid_fraction,
        )
        del depths, opacity
    elif not args.peer_features_only:
        graph = build_latent_reprojection_graph(
            rotations,
            latents.shape[-2],
            latents.shape[-1],
            fov_degrees,
            device=device,
            dtype=latents.dtype,
            minimum_overlap_fraction=args.minimum_overlap_fraction,
        )
    if graph is not None:
        required_pairs = [tuple(value.split(":")) for value in args.required_loop_pair]
        graph_stats = validate_reprojection_graph(
            graph,
            view_names,
            required_bidirectional_pairs=required_pairs,
        )
    peer_contexts = (
        build_peer_contexts(
            records,
            rotations,
            fov_degrees=fov_degrees,
            max_peer_views=args.max_peer_views,
        )
        if args.peer_conditioning
        else None
    )
    if rank == 0:
        print(
            f"SYNC_CONTEXT window={window_index} "
            f"real_neighbors={int(records[0]['neighbor_hidden_states'].shape[1])} "
            f"peer_neighbors={int(peer_contexts[0]['peer_neighbor_count']) if peer_contexts else 0} "
            f"peer_views={len(peer_contexts[0]['peer_source_indices']) if peer_contexts else 0}",
            flush=True,
        )
    initial_noise_stats = []
    if args.initial_noise_consensus_iterations:
        initial_noise = torch.stack([record["initial_noise"] for record in records]).to(device)
        for _ in range(args.initial_noise_consensus_iterations):
            initial_noise, stats = synchronize_latent_views(initial_noise, graph, blend=1.0)
            initial_noise_stats.append(stats)
        conditions = torch.stack([record["condition"] for record in records]).to(device)
        opacities = torch.stack([record["opacity"] for record in records]).to(device)
        latents = pipe.prepare_latents(conditions, opacities, True, noise=initial_noise)
        del initial_noise, conditions, opacities
        if rank == 0:
            print(
                f"SYNC_INITIAL_NOISE window={window_index} "
                f"iterations={args.initial_noise_consensus_iterations} "
                f"{json.dumps(initial_noise_stats[-1], sort_keys=True)}",
                flush=True,
            )
    if args.peer_features_only:
        initial_stats = {
            "external_latent_averaging": 0.0,
            "external_noise_averaging": 0.0,
            "internal_peer_cross_attention": 1.0,
        }
    elif args.synchronization_target == "latents":
        latents, initial_stats = synchronize_latent_views(
            latents,
            graph,
            blend=args.synchronization_blend,
        )
    else:
        # The condition latents come from one shared 3DGS and are already the
        # sharpest geometry-consistent signal available.  Averaging them here
        # irreversibly blurs view-specific detail before denoising begins.
        initial_stats = {
            "directed_edges": float(len(graph)),
            "mean_overlap_fraction": float(sum(edge.overlap_fraction for edge in graph) / len(graph)),
            "latent_content_averaging": 0.0,
        }
    if rank == 0:
        print(
            f"SYNC_GRAPH window={window_index} views={len(view_names)} "
            f"directed_edges={len(graph) if graph is not None else 0} "
            f"depth_aware={args.depth_dir is not None} peer_features_only={args.peer_features_only}",
            flush=True,
        )
        print(f"SYNC_INITIAL {json.dumps(initial_stats, sort_keys=True)}", flush=True)

    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    step_stats = []
    for step_index, timestep_value in enumerate(
        tqdm(pipe.scheduler.timesteps, disable=(rank != 0), desc="closed-loop denoising")
    ):
        noise_predictions = torch.empty_like(latents)
        for start in range(0, len(records), args.view_microbatch_size):
            stop = min(start + args.view_microbatch_size, len(records))
            indices = range(start, stop)
            hidden_states = latents[start:stop]
            prompt_embeds = stack_records(records, indices, "prompt_embeds", device)
            context_records = peer_contexts if peer_contexts is not None else records
            neighbor_hidden_states = (
                stack_peer_hidden_states(records, peer_contexts, indices, device)
                if peer_contexts is not None
                else stack_records(records, indices, "neighbor_hidden_states", device)
            )
            common_kwargs = {
                "hidden_states": hidden_states,
                "timestep": timestep_value.expand(stop - start).to(hidden_states.dtype),
                "neighbor_hidden_states": neighbor_hidden_states,
                "opacity": stack_records(records, indices, "opacity", device),
                "camera_rays": stack_records(records, indices, "camera_rays", device),
                "w2cs": stack_records(records, indices, "w2cs", device),
                "neighbor_w2cs": stack_records(context_records, indices, "neighbor_w2cs", device),
                "Ks": stack_records(records, indices, "Ks", device),
                "neighbor_Ks": stack_records(context_records, indices, "neighbor_Ks", device),
                "neighbor_prope_chunk_cameras": args.neighbor_prope_chunk_cameras,
                "return_dict": False,
            }
            noise_cond = pipe.transformer(encoder_hidden_states=prompt_embeds, **common_kwargs)[0]
            negative_prompt = pipe.default_negative_prompt.expand(stop - start, -1, -1)
            noise_uncond = pipe.transformer(encoder_hidden_states=negative_prompt, **common_kwargs)[0]
            noise_predictions[start:stop] = noise_uncond + 5.0 * (noise_cond - noise_uncond)
            del hidden_states, prompt_embeds, neighbor_hidden_states
            del common_kwargs, noise_cond, noise_uncond, negative_prompt
        if args.peer_features_only:
            stats = {
                "external_latent_averaging": 0.0,
                "external_noise_averaging": 0.0,
                "internal_peer_cross_attention": 1.0,
            }
        elif args.synchronization_target == "noise_predictions":
            noise_predictions, stats = synchronize_latent_views(
                noise_predictions,
                graph,
                blend=args.synchronization_blend,
            )
            stats = {f"noise_{key}": value for key, value in stats.items()}
        latents = pipe.scheduler.step(noise_predictions, timestep_value, latents, return_dict=False)[0]
        del noise_predictions
        if args.peer_features_only:
            pass
        elif args.synchronization_target == "latents":
            latents, stats = synchronize_latent_views(
                latents,
                graph,
                blend=args.synchronization_blend,
            )
        elif args.post_step_latent_blend:
            latents, latent_stats = synchronize_latent_views(
                latents,
                graph,
                blend=args.post_step_latent_blend,
            )
            stats = {
                **stats,
                **{f"post_latent_{key}": value for key, value in latent_stats.items()},
            }
        step_stats.append(stats)
        if rank == 0:
            print(
                f"SYNC_STEP window={window_index} index={step_index} "
                f"{json.dumps(stats, sort_keys=True)}",
                flush=True,
            )

    final_latent_stats = []
    for final_index in range(args.final_latent_consensus_iterations):
        latents, stats = synchronize_latent_views(latents, graph, blend=1.0)
        final_latent_stats.append(stats)
        if rank == 0:
            print(
                f"SYNC_FINAL_LATENT window={window_index} index={final_index} "
                f"{json.dumps(stats, sort_keys=True)}",
                flush=True,
            )

    final_low_frequency_stats = []
    for final_index in range(args.final_low_frequency_consensus_iterations):
        latents, stats = synchronize_low_frequency_latent_views(
            latents,
            graph,
            kernel_size=args.final_low_frequency_kernel_size,
            blend=1.0,
        )
        final_low_frequency_stats.append(stats)
        if rank == 0:
            print(
                f"SYNC_FINAL_LOW_FREQUENCY window={window_index} index={final_index} "
                f"{json.dumps(stats, sort_keys=True)}",
                flush=True,
            )

    save_outputs(
        pipe,
        records,
        latents,
        view_names,
        args.save_dir,
        rank,
        source_indices=source_indices,
        output_source_indices=output_source_indices,
        frames_per_view=frames_per_view,
        window_index=window_index,
    )
    peer_count = int(peer_contexts[0]["peer_neighbor_count"]) if peer_contexts else 0
    real_count = int(records[0]["neighbor_hidden_states"].shape[1])
    result = {
        "window_index": window_index,
        "source_indices": source_indices,
        "output_source_indices": output_source_indices,
        "original_frame_count": next(iter(frame_counts)),
        "padded_frame_count": next(iter(padded_counts)),
        "latent_frame_count": int(latents.shape[2]),
        "peer_neighbor_count": peer_count,
        "real_neighbor_count": real_count,
        "deterministic_overlap_noise": bool(args.deterministic_overlap_noise),
        "noise_ids": (
            records[0]["noise_ids"].tolist() if records[0]["noise_ids"] is not None else None
        ),
        "graph": graph_stats,
        "initial_noise_stats": initial_noise_stats,
        "initial_stats": initial_stats,
        "step_stats": step_stats,
        "final_latent_stats": final_latent_stats,
        "final_low_frequency_stats": final_low_frequency_stats,
    }
    del records, latents, graph, peer_contexts
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def run_multiview(pipe, dataset, args: argparse.Namespace, rank: int, device: torch.device) -> None:
    view_names, rotations, fov_degrees, frames_per_view = load_rig(args.manifest)
    windows = load_window_manifest(
        args.window_manifest,
        view_names=view_names,
        frames_per_view=frames_per_view,
        dataset_size=len(dataset),
    )
    window_results = []
    for window_index, window in enumerate(windows):
        result = run_multiview_group(
            pipe,
            dataset,
            args,
            rank,
            device,
            view_names=view_names,
            rotations=rotations,
            fov_degrees=fov_degrees,
            frames_per_view=frames_per_view,
            dataset_offset=window_index * len(view_names),
            window=window,
        )
        window_results.append(result)
        if rank == 0:
            print(f"SYNC_WINDOW_DONE {window_index + 1}/{len(windows)}", flush=True)

    if rank == 0:
        window_blending = merge_window_outputs(
            args.save_dir,
            windows,
            view_names,
            frames_per_view,
        )
        print(f"SYNC_WINDOW_BLEND {json.dumps(window_blending, sort_keys=True)}", flush=True)
        published_per_view = sum(
            1
            for window in windows
            for value in (window.get("output_source_indices") or [])
            if value != -1
        )
        expected = len(view_names) * (published_per_view or frames_per_view)
        actual = len(list((args.save_dir / "merged_predictions").glob("*.png")))
        if actual != expected:
            raise RuntimeError(f"full output has {actual} frames; expected {expected}")
        metadata = {
            "schema_version": 2,
            "view_order": view_names,
            "fov_degrees": fov_degrees,
            "frames_per_view": frames_per_view,
            "window_count": len(windows),
            "num_inference_steps": args.num_inference_steps,
            "synchronization_blend": args.synchronization_blend,
            "synchronization_target": args.synchronization_target,
            "initial_noise_consensus_iterations": args.initial_noise_consensus_iterations,
            "post_step_latent_blend": args.post_step_latent_blend,
            "final_latent_consensus_iterations": args.final_latent_consensus_iterations,
            "final_low_frequency_consensus_iterations": args.final_low_frequency_consensus_iterations,
            "final_low_frequency_kernel_size": args.final_low_frequency_kernel_size,
            "peer_conditioning": args.peer_conditioning,
            "deterministic_overlap_noise": args.deterministic_overlap_noise,
            "deterministic_noise_seed": (
                args.deterministic_noise_seed if args.deterministic_overlap_noise else None
            ),
            "depth_aware": args.depth_dir is not None,
            "depth_relative_tolerance": args.depth_relative_tolerance if args.depth_dir else None,
            "required_loop_pairs": args.required_loop_pair,
            "minimum_overlap_fraction": args.minimum_overlap_fraction,
            "view_microbatch_size": args.view_microbatch_size,
            "neighbor_prope_chunk_cameras": args.neighbor_prope_chunk_cameras,
            "window_blending": window_blending,
            "windows": window_results,
        }
        args.save_dir.mkdir(parents=True, exist_ok=True)
        (args.save_dir / "synchronization_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"SYNC_OUTPUT_DIR={args.save_dir}", flush=True)


def main(args: argparse.Namespace) -> None:
    rank, world_size, local_rank = init_distributed(args.distributed_timeout_minutes)
    device = torch.device(f"cuda:{local_rank}")
    if world_size != args.context_parallel_size:
        raise ValueError(
            "closed-loop inference requires exactly one CP group: "
            f"world_size={world_size}, context_parallel_size={args.context_parallel_size}"
        )
    try:
        pipe = get_eval_pipe(args, device)
        barrier_if_distributed()
        if rank == 0:
            print("Initialized closed-loop multiview pipeline", flush=True)
        load_transformer_checkpoint(pipe.transformer, args)
        pipe.transformer.eval()
        if rank == 0:
            print("Loaded transformer checkpoint", flush=True)
        dataset = create_dataset(args, rank)

        mesh, num_groups, _, _ = create_context_parallel_meshes(
            rank, world_size, args.context_parallel_size, device.type
        )
        if num_groups != 1 or mesh is None:
            raise RuntimeError(f"expected one non-empty CP mesh, got num_groups={num_groups}")
        pipe.transformer.enable_context_parallel(mesh)
        try:
            run_multiview(pipe, dataset, args, rank, device)
        finally:
            pipe.transformer.disable_context_parallel()
        barrier_if_distributed()
        if rank == 0:
            print("Done!", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
