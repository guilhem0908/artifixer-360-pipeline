# Synchronized multi-view depth/loop pipeline

This branch processes the 14 perspective directions jointly at each temporal
window. It shares features/noise between geometrically overlapping views and
requires an explicit horizontal closure pair.

## 1. Generate the world-locked rig

```bash
source configs/artifixer360.env
cd "$ARTIFIXER_REPO"

python scripts/generate_world_locked14_trajectories.py \
  "$BASE_TRANSFORMS" "$TRAJECTORY_ROOT/frustum_rig" \
  --size "$VIEW_SIZE" --fov-degrees "$FOV_DEGREES"
```

Prepare/render the rig's `combined.json` with the frozen base 3DGRUT checkpoint
using `data_processing.prepare_colmap_artifixer_inputs`. The resulting split
must contain all targets first and the selected real anchors after them.

## 2. Prepare overlapping temporal windows

```bash
python scripts/prepare_depth_synchronized_multiview_windows.py \
  --source-split "$TARGET_SCENE_ROOT/split.json" \
  --rig-manifest "$TRAJECTORY_ROOT/frustum_rig/manifest.json" \
  --output-root "$RUN_ROOT/depth_windows" \
  --scene-prefix "$SCENE_ID-depth-loop" \
  --frame-start 0 --frame-stop "$FRAME_COUNT" \
  --window-size 13 --window-stride 8
```

The window manifest publishes every source position once while overlapping
neighboring calls. Window size and stride must remain compatible with the VAE
temporal phase.

## 3. Run synchronized ArtiFixer

```bash
DEPTH_DIR=/path/to/world_locked_14view/depth

torchrun --standalone --nproc_per_node="$GPU_COUNT" \
  -m model_eval.run_synchronized_multiview_inference \
  --evalset reconstructed_colmap \
  --checkpoint_pt "$ARTIFIXER_CHECKPOINT" \
  --save_dir "$OUTPUT_ROOT/depth_loop" \
  --split_path "$RUN_ROOT/depth_windows/split.json" \
  --manifest "$TRAJECTORY_ROOT/frustum_rig/manifest.json" \
  --window-manifest "$RUN_ROOT/depth_windows/window_manifest.json" \
  --depth-dir "$DEPTH_DIR" \
  --render_trajectory trajectory \
  --num_views 12 --max_neighbors_per_encode 1 \
  --inference_pipeline bidirectional \
  --context_parallel_size "$GPU_COUNT" \
  --bidirectional_chunk_size 13 \
  --num_inference_steps 8 \
  --view-microbatch-size 1 \
  --peer-conditioning \
  --initial-noise-consensus-iterations 3 \
  --synchronization-target noise_predictions \
  --synchronization-blend 0.75 \
  --post-step-latent-blend 0.0 \
  --final-latent-consensus-iterations 0 \
  --final-low-frequency-consensus-iterations 1 \
  --final-low-frequency-kernel-size 5 \
  --minimum-overlap-fraction 0.02 \
  --depth-relative-tolerance 0.08 \
  --minimum-depth-overlap-fraction 0.002 \
  --depth-spatial-valid-fraction 0.5 \
  --depth-temporal-valid-fraction 0.5 \
  --required-loop-pair H000:H300
```

The output `synchronization_metadata.json` records the actual graph, windows,
loop pairs and consensus settings.

## 4. Evaluate before distillation

```bash
python scripts/evaluate_depth_synchronized_multiview.py \
  --manifest "$TRAJECTORY_ROOT/frustum_rig/manifest.json" \
  --window-manifest "$RUN_ROOT/depth_windows/window_manifest.json" \
  --input-dir /path/to/world_locked_14view/renders \
  --prediction-dir "$OUTPUT_ROOT/depth_loop/merged_predictions" \
  --depth-dir "$DEPTH_DIR" \
  --output "$OUTPUT_ROOT/depth_loop/depth_overlap_qc.json" \
  --qc-size 256 --depth-relative-tolerance 0.08 \
  --max-prediction-mae 0.10 --max-prediction-p95 0.30 \
  --max-regression-ratio 1.15 --regression-slack 0.005 \
  --minimum-coverage 0.002
```

The report must confirm `horizontal_loop_present`. Depth-overlap MAE is the
primary geometric metric; ERP seam and temporal warp are complementary output
metrics.

## 5. Stitch and optionally distill

`merged_predictions` is view-major and can be passed directly to
`stitch_artifixer_frustums360.py` with the same rig manifest.

For distillation, use the same canonical target and geometry-lock procedure as
the snake branch. Keep the pre-distillation depth-overlap report: a cleaner
pseudo-view set does not prove that the following 3D optimization retained the
gain.

## Interpretation boundary

Joint feature communication can improve appearance and overlapping-view
agreement. It still does not make the perspective video transformer understand
that two pixels are the same persistent 3D surface. Depth and poses constrain
the exchange; the distillation stage must preserve those constraints.
