# ABCI execution

The code is cluster-agnostic. ABCI-specific queue names, group IDs, usernames
and storage roots are deliberately excluded from the public repository.

## Four-GPU launch

Inside the ArtiFixer CUDA container:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m model_eval.run_inference \
  --evalset reconstructed_colmap \
  --checkpoint_pt "$ARTIFIXER_CHECKPOINT" \
  --save_dir "$OUTPUT_ROOT/artifixer2d" \
  --split_path "$TARGET_SCENE_ROOT/split.json" \
  --inference_manifest_path "$TARGET_SCENE_ROOT/inference_manifest.json" \
  --render_trajectory trajectory \
  --num_views 12 \
  --max_neighbors_per_encode 1 \
  --inference_pipeline bidirectional \
  --context_parallel_size 4 \
  --bidirectional_chunk_size 77 \
  --save_frame_outputs_only
```

For joint multi-view inference, use
`model_eval.run_synchronized_multiview_inference` as shown in
[DEPTH_LOOP_PIPELINE.md](DEPTH_LOOP_PIPELINE.md).

## PBS template requirements

A local PBS wrapper should:

1. request one four-GPU node;
2. use `set -euo pipefail` and an error trap;
3. validate every input before creating the run directory;
4. refuse to overwrite an existing run;
5. bind repository, data, checkpoint and cache roots explicitly;
6. write one log and one status file per run;
7. verify frame counts and QC reports before encoding an MP4.

No public script assumes a particular ABCI allocation. Set the queue,
`group_list`, wall time and absolute paths in your private submission wrapper.
