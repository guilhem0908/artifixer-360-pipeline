# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental joint ArtiFixer inference for one overlapping view pair.

Unlike the regular data-parallel evaluator, both views are held in one batch.
Their exact spherical overlap is fused after latent initialization and after
every scheduler step.  The script intentionally supports a single CP group so
that every rank participates in both view predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from model_eval.synchronized_multiview import synchronize_latent_pair
from model_training.utils.video_io import save_video


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--view-names", nargs=2, required=True)
    parser.add_argument("--view-yaws", nargs=2, type=float, required=True)
    parser.add_argument("--view-start-indices", nargs=2, type=int, required=True)
    parser.add_argument("--relative-frame-start", type=int, default=0)
    parser.add_argument("--fov-degrees", type=float, required=True)
    parser.add_argument("--synchronization-blend", type=float, default=0.5)
    parser.add_argument("--anchor-strength", type=float, default=0.25)
    args = parser.parse_args()
    validate_checkpoint_args(parser, args)
    validate_evalset_args(parser, args)
    if args.inference_pipeline != "bidirectional":
        parser.error("synchronized pair inference requires --inference_pipeline bidirectional")
    if not 0.0 <= args.synchronization_blend <= 1.0:
        parser.error("--synchronization-blend must be in [0, 1]")
    if not 0.0 <= args.anchor_strength <= 1.0:
        parser.error("--anchor-strength must be in [0, 1]")
    return args


def _stack_pair(items: list[dict], key: str, device: torch.device, dtype=None) -> torch.Tensor:
    value = torch.stack([item[key] for item in items]).to(device)
    return value.to(dtype=dtype) if dtype is not None else value


def _save_rgb_tensor(frame: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(frame, path)


@torch.inference_mode()
def run_pair(pipe, items: list[dict], args: argparse.Namespace, rank: int, device: torch.device) -> None:
    if len(items) != 2:
        raise ValueError(f"expected exactly two dataset items, got {len(items)}")
    frame_counts = [int(item["rgb_rendered"].shape[0]) for item in items]
    if frame_counts[0] != frame_counts[1]:
        raise ValueError(f"paired views must have equal frame counts, got {frame_counts}")

    original_num_frames = frame_counts[0]
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

    rgb_rendered = _stack_pair(items, "rgb_rendered", device)
    if pad_frames:
        rgb_rendered = pad_temporal(rgb_rendered, pad_frames, dim=1)
    opacity = _stack_pair(items, "opacity", device, torch.bfloat16)
    if pad_frames:
        opacity = pad_temporal(opacity, pad_frames, dim=1)
    camera_rays = _stack_pair(items, "camera_rays", device, torch.bfloat16)
    w2cs = _stack_pair(items, "w2cs", device)
    Ks = _stack_pair(items, "Ks", device)
    latent_pad_frames = target_latent_num_frames - camera_rays.shape[1]
    if latent_pad_frames:
        camera_rays = pad_temporal(camera_rays, latent_pad_frames, dim=1)
        w2cs = pad_temporal(w2cs, latent_pad_frames, dim=1)
        Ks = pad_temporal(Ks, latent_pad_frames, dim=1)

    neighbors = _stack_pair(items, "rgb_neighbors", device)
    neighbor_w2cs = _stack_pair(items, "neighbor_w2cs", device)
    neighbor_Ks = _stack_pair(items, "neighbor_Ks", device)
    prompt_embeds = _stack_pair(items, "encoded_prompt", device)

    condition = pipe.encode_video_frames(rgb_rendered)
    neighbors_condition = pipe.encode_neighbors(neighbors, args.max_neighbors_per_encode)
    latents = pipe.prepare_latents(condition, opacity, True)
    latents, initial_stats = synchronize_latent_pair(
        latents,
        fov_degrees=args.fov_degrees,
        yaw_degrees=tuple(args.view_yaws),
        blend=args.synchronization_blend,
    )
    if rank == 0:
        print(f"SYNC_INITIAL {json.dumps(initial_stats, sort_keys=True)}", flush=True)

    negative_prompt_embeds = pipe.default_negative_prompt.expand(2, -1, -1)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    step_stats = []
    for step_index, timestep_value in enumerate(
        tqdm(pipe.scheduler.timesteps, disable=(rank != 0), desc="synchronized denoising")
    ):
        timestep = timestep_value.expand(latents.shape[0])
        common_kwargs = dict(
            hidden_states=latents,
            timestep=timestep.to(latents.dtype),
            neighbor_hidden_states=neighbors_condition,
            opacity=opacity,
            camera_rays=camera_rays,
            w2cs=w2cs,
            neighbor_w2cs=neighbor_w2cs,
            Ks=Ks,
            neighbor_Ks=neighbor_Ks,
            return_dict=False,
        )
        noise_pred = pipe.transformer(encoder_hidden_states=prompt_embeds, **common_kwargs)[0]
        noise_uncond = pipe.transformer(encoder_hidden_states=negative_prompt_embeds, **common_kwargs)[0]
        noise_pred = noise_uncond + 5.0 * (noise_pred - noise_uncond)
        latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
        latents, stats = synchronize_latent_pair(
            latents,
            fov_degrees=args.fov_degrees,
            yaw_degrees=tuple(args.view_yaws),
            blend=args.synchronization_blend,
        )
        step_stats.append(stats)
        if rank == 0:
            print(f"SYNC_STEP index={step_index} {json.dumps(stats, sort_keys=True)}", flush=True)

    if rank != 0:
        return

    # Decode both the fully generated result and a geometry-anchored variant.
    # The latter preserves a configurable fraction of the input 3DGS latent;
    # producing both costs no additional transformer inference.
    synchronized_video = pipe.video_processor.postprocess_video(
        pipe.latents_to_rgb(latents), output_type="pt"
    )[:, :original_num_frames].cpu().float().clamp(0, 1)
    anchored_latents = torch.lerp(latents, condition, args.anchor_strength)
    anchored_video = pipe.video_processor.postprocess_video(
        pipe.latents_to_rgb(anchored_latents), output_type="pt"
    )[:, :original_num_frames].cpu().float().clamp(0, 1)
    rendered_video = rgb_rendered[:, :original_num_frames].cpu().float().clamp(0, 1)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "view_names": args.view_names,
        "view_yaws": args.view_yaws,
        "view_start_indices": args.view_start_indices,
        "fov_degrees": args.fov_degrees,
        "frame_count": original_num_frames,
        "padded_frame_count": target_num_frames,
        "num_inference_steps": args.num_inference_steps,
        "synchronization_blend": args.synchronization_blend,
        "anchor_strength": args.anchor_strength,
        "initial_stats": initial_stats,
        "step_stats": step_stats,
    }
    (args.save_dir / "synchronization_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for view_index, (view_name, view_start) in enumerate(zip(args.view_names, args.view_start_indices)):
        global_indices = [int(value) for value in items[view_index]["frame_indices"]]
        relative_indices = [value - view_start for value in global_indices]
        for local_index, (global_index, relative_index) in enumerate(zip(global_indices, relative_indices)):
            name = f"{relative_index:05d}.png"
            _save_rgb_tensor(synchronized_video[view_index, local_index], args.save_dir / view_name / "sync" / name)
            _save_rgb_tensor(anchored_video[view_index, local_index], args.save_dir / view_name / "anchor" / name)
            _save_rgb_tensor(rendered_video[view_index, local_index], args.save_dir / view_name / "rendered" / name)
            global_name = f"{global_index:05d}.png"
            _save_rgb_tensor(
                synchronized_video[view_index, local_index], args.save_dir / "global" / "sync" / global_name
            )
            _save_rgb_tensor(
                anchored_video[view_index, local_index], args.save_dir / "global" / "anchor" / global_name
            )
        save_video(
            synchronized_video[view_index],
            args.save_dir / view_name / "sync.mp4",
            fps=args.output_fps,
        )
        save_video(
            anchored_video[view_index],
            args.save_dir / view_name / "anchor.mp4",
            fps=args.output_fps,
        )
        save_video(
            rendered_video[view_index],
            args.save_dir / view_name / "rendered.mp4",
            fps=args.output_fps,
        )
    print(f"SYNC_OUTPUT_DIR={args.save_dir}", flush=True)


def main(args: argparse.Namespace) -> None:
    rank, world_size, local_rank = init_distributed(args.distributed_timeout_minutes)
    device = torch.device(f"cuda:{local_rank}")
    if world_size != args.context_parallel_size:
        raise ValueError(
            "the synchronized prototype requires exactly one CP group: "
            f"world_size={world_size}, context_parallel_size={args.context_parallel_size}"
        )
    try:
        pipe = get_eval_pipe(args, device)
        barrier_if_distributed()
        if rank == 0:
            print("Initialized synchronized pair pipeline", flush=True)
        load_transformer_checkpoint(pipe.transformer, args)
        pipe.transformer.eval()
        if rank == 0:
            print("Loaded transformer checkpoint", flush=True)

        dataset = create_dataset(args, rank)
        if len(dataset) != 2:
            raise ValueError(f"synchronized pair split must produce exactly two items, got {len(dataset)}")
        items = [dataset[0], dataset[1]]
        starts = [int(item["frame_indices"][0]) for item in items]
        expected_starts = [value + args.relative_frame_start for value in args.view_start_indices]
        if starts != expected_starts:
            raise ValueError(f"dataset item starts {starts}, expected {expected_starts}")

        mesh, num_groups, _, _ = create_context_parallel_meshes(
            rank, world_size, args.context_parallel_size, device.type
        )
        if num_groups != 1 or mesh is None:
            raise RuntimeError(f"expected one non-empty CP mesh, got num_groups={num_groups}")
        pipe.transformer.enable_context_parallel(mesh)
        try:
            run_pair(pipe, items, args, rank, device)
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
