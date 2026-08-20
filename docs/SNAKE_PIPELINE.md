# Continuous perspective snake pipeline

This branch presents ArtiFixer with one ordered perspective video whose camera
direction winds around the sphere using small rotations. It is not a native ERP
model.

All commands assume:

```bash
source configs/artifixer360.env
cd "$ARTIFIXER_REPO"
mkdir -p "$TRAJECTORY_ROOT" "$OUTPUT_ROOT"
```

## 1. Generate the winding trajectory and the render rig

The default order covers the equator, upper ring and lower ring. Consecutive
rotations are limited to five degrees, and alternate lanes run in opposite
temporal directions to avoid a large reset at every lane boundary.

```bash
python scripts/generate_spherical_snake_trajectory.py \
  "$BASE_TRANSFORMS" "$TRAJECTORY_ROOT/snake.json" \
  --manifest "$TRAJECTORY_ROOT/snake_manifest.json" \
  --reference-frame "$REFERENCE_FRAME" \
  --start-frame 0 --frame-count "$FRAME_COUNT" \
  --orientation 0:0    --orientation-name H000 \
  --orientation 0:45   --orientation-name U000 \
  --orientation 90:45  --orientation-name U090 \
  --orientation 60:0   --orientation-name H060 \
  --orientation 120:0  --orientation-name H120 \
  --orientation 90:-45 --orientation-name D090 \
  --orientation 180:-45 --orientation-name D180 \
  --orientation 180:0   --orientation-name H180 \
  --orientation 180:45  --orientation-name U180 \
  --orientation 270:45  --orientation-name U270 \
  --orientation 240:0   --orientation-name H240 \
  --orientation 300:0   --orientation-name H300 \
  --orientation 270:-45 --orientation-name D270 \
  --orientation 360:-45 --orientation-name D000 \
  --max-turn-degrees "$MAX_TURN_DEGREES" \
  --size "$VIEW_SIZE" --fov-degrees "$FOV_DEGREES"

python scripts/generate_world_locked14_trajectories.py \
  "$BASE_TRANSFORMS" "$TRAJECTORY_ROOT/frustum_rig" \
  --size "$VIEW_SIZE" --fov-degrees "$FOV_DEGREES"
```

For 154 source positions, this recipe produces 2,156 lane frames plus the
small-angle connectors. Read `snake_manifest.json`; never hard-code the target
count for a different video.

## 2. Render the initial 3D scene along the snake

Reuse the frozen base checkpoint instead of retraining geometry for the target
trajectory:

```bash
python -m data_processing.prepare_colmap_artifixer_inputs \
  --colmap_dir "$COLMAP_DIR" \
  --output_root "$TARGET_SCENE_ROOT" \
  --selected_image_names_file "$BASE_SELECTED_IMAGES" \
  --trajectory_path "$TRAJECTORY_ROOT/snake.json" \
  --phases prepare,render,scale \
  --reconstruction_checkpoint "$BASE_CHECKPOINT" \
  --reconstruction_steps 30000 \
  --metric_scale "$METRIC_SCALE" \
  --replace
```

The rendered trajectory must contain RGB, opacity and radial depth for every
snake target.

## 3. Build bounded context windows

```bash
python scripts/generate_spherical_snake_inference_manifest.py \
  "$TARGET_SCENE_ROOT/trajectory/target_indices.json" \
  "$TARGET_SCENE_ROOT/recon_results/$SCENE_ID/reconstruction/$SCENE_ID/ours_30000/trajectory/selected_indices.json" \
  "$TARGET_SCENE_ROOT/inference_manifest.json" \
  --context-views 12 \
  --core-size 49 --halo-size 12 \
  --maximum-item-frames 77 \
  --trajectory "$TRAJECTORY_ROOT/snake.json" \
  --translation-break-ratio 5
```

The manifest guarantees complete, non-duplicated publication of target indices
while keeping each model call within the configured context limit.

## 4. Run ArtiFixer2D

```bash
torchrun --standalone --nproc_per_node="$GPU_COUNT" \
  -m model_eval.run_inference \
  --evalset reconstructed_colmap \
  --checkpoint_pt "$ARTIFIXER_CHECKPOINT" \
  --save_dir "$OUTPUT_ROOT/artifixer2d" \
  --split_path "$TARGET_SCENE_ROOT/split.json" \
  --inference_manifest_path "$TARGET_SCENE_ROOT/inference_manifest.json" \
  --render_trajectory trajectory \
  --num_views 12 --max_neighbors_per_encode 1 \
  --inference_pipeline bidirectional \
  --context_parallel_size "$GPU_COUNT" \
  --bidirectional_chunk_size 77 \
  --save_frame_outputs_only
```

Materialize the nested outputs into one contiguous stream:

```bash
TARGET_COUNT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["target_count"])' \
  "$TRAJECTORY_ROOT/snake_manifest.json")

python -m scripts.materialize_spherical_snake_predictions \
  --input-root "$OUTPUT_ROOT/artifixer2d" \
  --output-dir "$OUTPUT_ROOT/artifixer2d_all" \
  --expected-count "$TARGET_COUNT"
```

## 5. Inspect an ArtiFixer2D ERP before distillation

Extract the 14 lanes and project them to ERP:

```bash
python -m scripts.extract_spherical_snake_lanes \
  --prediction-root "$OUTPUT_ROOT/artifixer2d" \
  --snake-manifest "$TRAJECTORY_ROOT/snake_manifest.json" \
  --rig-manifest "$TRAJECTORY_ROOT/frustum_rig/manifest.json" \
  --output-dir "$OUTPUT_ROOT/artifixer2d_view_major"

python scripts/stitch_artifixer_frustums360.py \
  --input-dir "$OUTPUT_ROOT/artifixer2d_view_major" \
  --manifest "$TRAJECTORY_ROOT/frustum_rig/manifest.json" \
  --output-dir "$OUTPUT_ROOT/artifixer2d_erp" \
  --output-width "$ERP_WIDTH" --output-height "$ERP_HEIGHT" \
  --blend-mode feather --ownership-mode angular \
  --fps "$FPS" --crf 17
```

## 6. Build canonical depth and loop targets

```bash
RAW_TRAJECTORY="$TARGET_SCENE_ROOT/recon_results/$SCENE_ID/reconstruction/$SCENE_ID/ours_30000/trajectory"

python scripts/build_depth_canonical_targets.py \
  --transforms "$TRAJECTORY_ROOT/snake.json" \
  --depth-dir "$RAW_TRAJECTORY/depth" \
  --opacity-dir "$RAW_TRAJECTORY/opacity" \
  --render-dir "$RAW_TRAJECTORY/renders" \
  --prediction-dir "$OUTPUT_ROOT/artifixer2d_all" \
  --output-dir "$OUTPUT_ROOT/canonical_targets" \
  --sample-stride 4 --voxel-pixels 2.5 \
  --depth-percentile 99.5 \
  --low-opacity 0.78 --high-opacity 0.98 \
  --singleton-weight 0.25 --support-for-full-weight 4
```

For every target, attach its radial depth and expose the canonical image/mask
as color-loop sidecars:

```bash
for ((i=0; i<TARGET_COUNT; i++)); do
  index=$(printf '%05d' "$i")
  ln -s "$RAW_TRAJECTORY/depth/$index.npy" \
    "$OUTPUT_ROOT/canonical_targets/frames/${index}_depth.npy"
  ln -s "$OUTPUT_ROOT/canonical_targets/frames/$index.png" \
    "$OUTPUT_ROOT/canonical_targets/frames/${index}_color_loop.png"
  ln -s "$OUTPUT_ROOT/canonical_targets/frames/${index}_mask.png" \
    "$OUTPUT_ROOT/canonical_targets/frames/${index}_color_loop_mask.png"
done
```

## 7. Geometry-locked ArtiFixer3D distillation

```bash
python -m data_processing.run_artifixer3d \
  --scene_root "$TARGET_SCENE_ROOT" \
  --artifixer_frames_dir "$OUTPUT_ROOT/canonical_targets/frames" \
  --output_root "$RUN_ROOT/artifixer3d" \
  --render_trajectory_path "$TRAJECTORY_ROOT/snake.json" \
  --base_checkpoint "$BASE_CHECKPOINT" \
  --fresh_optimizer_on_resume --geometry_locked \
  --depth_guided --depth_loss_weight 0.5 \
  --color_loop_guided --color_loop_loss_weight 0.02 \
  --balanced_real_override_sampling --real_samples_per_override 3 \
  --override_reconstruction_weight 0.5 \
  --override_lpips_weight 0.05 \
  --artifixer3d_steps 35000 \
  --checkpoint_iterations 31000 33000 35000 \
  --config_name apps/colmap_3dgut_sparse_mcmc_lpips \
  --phases distill,render,prepare_artifixer3d_plus \
  --replace --no-use_wandb
```

Run `audit_artifixer3d_checkpoint_lock.py` between the base and final
checkpoints. Positions, rotations, scales and densities must remain bitwise
equal when `--geometry_locked` is claimed.

## 8. Render and stitch the distilled scene

Render the combined 14-view rig from the final ArtiFixer3D checkpoint, then:

```bash
python scripts/stitch_artifixer_frustums360.py \
  --input-dir /path/to/distilled/combined/renders \
  --manifest "$TRAJECTORY_ROOT/frustum_rig/manifest.json" \
  --output-dir "$OUTPUT_ROOT/distilled_erp" \
  --output-width "$ERP_WIDTH" --output-height "$ERP_HEIGHT" \
  --blend-mode feather --ownership-mode angular \
  --fps "$FPS" --crf 17
```

## 9. Optional ArtiFixer3D+

Use the generated `split_artifixer3d_plus.json` with the same inference
manifest, materialize the predictions, extract the snake lanes and stitch them
exactly as in steps 4–5. ArtiFixer3D+ is another video-restoration pass; it is
not a geometric optimizer.

## 10. Required QC

```bash
python scripts/evaluate_spherical_snake_turn.py --help
python scripts/evaluate_temporal_panorama.py --help
python scripts/audit_artifixer3d_checkpoint_lock.py --help
```

Do not encode or publish a final MP4 until target counts, loop/turn QC and the
geometry-lock audit pass.
