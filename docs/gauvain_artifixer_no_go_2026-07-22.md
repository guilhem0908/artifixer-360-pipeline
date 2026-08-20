# Gauvain panoramic restoration: final NO-GO report

Date: 2026-07-22 (Asia/Tokyo)

Status: **CLOSED — NO-GO with the current observations, weights, and pipeline family**

## Decision

The Gauvain campaign is closed. No additional sweep, prompt, seed, blending mode,
projection, or zero-shot image editor should be run on this branch.

The requested result was a panoramic video satisfying all three properties at the
same time:

1. locally clean, without 3DGS floaters, needles, black streaks, holes, or duplicated
   surfaces;
2. spatially continuous over the full equirectangular sphere, without face or window
   boundaries;
3. temporally stable, so that surfaces and objects do not change between frames.

No tested result satisfies all three. This is not a claim that panoramic video
restoration is impossible in general. It is a NO-GO under the available Gauvain
observations and the current pretrained-model approach.

At closure time, the local machine had no matching compute process, and `qstat` for
the ABCI account returned no active or queued job. A local VLC viewer displaying an
existing result was left untouched. No job had to be cancelled. Existing logs,
checkpoints, scripts, images, and videos are intentionally preserved.

## Decisive evidence

### Seam QC

The seam validator marks an adjacency as failed when its failure fraction is greater
than `0.1`. The table reports the mean and maximum failure fractions over the 12
cubemap adjacencies.

| Result | Mean | Maximum | Failed adjacencies | Finding |
|---|---:|---:|---:|---|
| Rules 35k + feather | 0.2467 | 0.5200 | 11/12 | Cleaner locally, still discontinuous |
| Rules 60k + feather | 0.5067 | 0.7600 | 12/12 | More optimization regressed |
| Raw 120k cubemap | 0.3420 | 0.7597 | 11/12 | Dirty but less contradictory than ArtiFixer |
| ArtiFixer 120k cubemap | 0.8328 | 0.9675 | 12/12 | Cleaning made seams much worse |
| Geometry-locked 3D | 0.0000 | 0.0000 | 0/12 | Continuous because geometry cannot move; geometric artifacts remain |
| Sequential-15 ArtiFixer | 0.7776 | 0.9545 | 12/12 | Shared sequence did not reconcile the views |
| FOV110 feather | 0.6699 | 0.9610 | 12/12 | Feather hides boundaries with ghosts |
| FOV110 loop-closed | 0.4610 | 0.8571 | 12/12 | Better number, still fails every adjacency |
| ERP 480k raw | 0.5379 | 0.9481 | 12/12 | Existing geometric contradiction |
| ERP labelled “seamless” | 0.5379 | 0.9481 | 12/12 | Identical QC: post-blending did not repair geometry |

The source files for these measurements remain on ABCI under:

- `/path/to/artifixer/outputs/gauvain_base30k_cubemap25_fov110_rules_retry1/`
- `/path/to/artifixer/outputs/gauvain_cubemap_1216_fov100_120k/`
- `/path/to/artifixer/outputs/gauvain_joint14_a3d_continuous_locked_qc/`
- `/path/to/artifixer/outputs/gauvain_seq15_480k_cubefaces_artifixer/`
- `/path/to/artifixer/outputs/gauvain_seq15_480k_cubefaces_artifixer_fov110/`
- `/path/to/artifixer/outputs/gauvain_seq15_clean_20260719/`

The zero seam score of the geometry-locked variant is not a clean-video success. It
is the empirical lower bound “coherent but dirty”: positions, rotations, scales, and
densities were frozen, so only appearance could change and geometric floaters could
not be removed.

### Temporal and canonical reprojection experiments

- The corrected return-loop experiment covered only `3.0865%` of the target pixels on
  average over 76 return frames. It cannot constrain most of the video.
- The world-space depth atlas found correspondences broadly, but changed `67.7630%`
  of generated pixels on average. Its apparent agreement comes from replacing most of
  the output, not from the denoiser learning a stable solution.
- The second depth-loop target construction still changed `58.7719%` of pixels on
  average.
- Independent overlap predictions had an overlap MAE of `19.13`; conservative
  composition reduced it to `7.89`. The `0.071` ownership result was obtained by
  explicitly selecting one view and reprojecting it into the other, not by making the
  two generations agree.
- Qwen completed all 154 frames, but the correction was effectively regenerated at
  every frame. The audited inter-frame residual variation was `1.021x` the mean edit
  amplitude, and the horizontal seam ratio worsened from about `1.06x` to `2.15x`.

Metric files:

- `/path/to/artifixer/outputs/gauvain_ns14_base30k_pilot_h000_h060/temporal_loop/metrics.json`
- `/path/to/artifixer/outputs/gauvain_worldlocked14_base30k_guided/depth_canonical/metrics.json`
- `/path/to/artifixer/outputs/gauvain_worldlocked14_clean154_depth/depth_loop_targets/metrics.json`

### Successful jobs that did not produce an acceptable video

PBS state `F` below means finished. Every listed job has `Exit_status = 0`; none is a
failed computation being mistaken for a negative result.

| Job | Name | Walltime | Output log | Technical outcome |
|---|---|---:|---|---|
| `164437.qjcm` | `gauvain-closedloop14` | 00:18:07 | `/path/to/artifixer/src/ArtiFixer/gauvain-closedloop14.o164437` | Latent/closed-loop pilot completed; visual inconsistency remained |
| `164637.qjcm` | `gauvain-noisesync14` | 00:17:46 | `/path/to/artifixer/src/ArtiFixer/gauvain-noisesync14.o164637` | Noise synchronization completed; smoothing/zones remained |
| `164684.qjcm` | `g15-final-ply` | 00:02:08 | `/path/to/artifixer/logs/gauvain_seq15_clean_20260719/final_export_ply.log` | The 15-stage chain reached the final 480k PLY; geometric artifacts remained |
| `172792.qjcm` | `qwen-edit360-full` | 00:36:43 | `/path/to/artifixer/src/ArtiFixer/qwen-edit360-full.o172792` | 154/154 edited frames; temporal hallucination remained |

This distinction is important: the campaign's `success.env` files and many previous
status reports checked exit codes, frame counts, or tensor locking. Those are useful
pipeline checks, but they are not evidence of clean, spatially continuous, temporally
stable video.

## What was tested, 14–22 July

| Date | Experiment families | Outcome |
|---|---|---|
| July 14 | Native ArtiFixer/ArtiFixer3D, corrected COLMAP, yaw rotations, ArtiFixer3D+ | Cleaning was possible in individual views; geometric and directional contradictions persisted |
| July 15 | Six cubemap faces, direct ERP | Cubemap introduced face disagreements; direct ERP was outside the trained projection distribution |
| July 16 | Fourteen Nerfstudio frustums, graph-cut, pair synchronization | Graph-cut moved sharp boundaries; synchronization averaged incompatible content |
| July 17 | Overlap consistency, temporal loop, world-locked views, depth canonicalization, geometry lock | Reprojection coverage was insufficient; locking exposed the coherent-but-dirty bound |
| July 18 | Closed-loop latent synchronization, joint snake, deterministic noise, depth loop, cubemap/direct/fisheye diagnostics | Shared noise did not provide shared surface identity; joint snake used an invalid 1D ordering for sphere × time |
| July 19 | Geometry regularization, angular ownership, noise-prediction synchronization, sequential 15-stage chain to 480k | The strongest common-3D attempt was temporally steadier but retained large geometric artifacts |
| July 20 | Direct ERP rendering, Fixer→ArtiFixer, raw→ArtiFixer, cubeface ArtiFixer, DiT360, pruning/refit | Stronger generation replaced the scene; pruning could not isolate the artifacts |
| July 21 | Eighteen overlapping views + Fixer, FOV110 loop closing, ArtiFixer3D rule variants | Feathering produced blur/ghosts; loop closing and longer fitting still failed all seams |
| July 22 | OmniGS, 360PanT, Qwen-Edit-360 | Raw artifacts remained or were replaced by temporally unstable hallucinations |

The pruning experiment removed only `18,595 / 1,000,000` Gaussians (`1.86%`), then
only `970 / 981,405` more after refitting. The unwanted structures were not a small,
separable population recoverable by another threshold sweep.

## Why the two objectives fight each other

The source camera has resolution `1972×1087` and focal length `924.0169 px`, giving a
field of view of approximately `93.72° × 60.93°`. Its rectangular solid angle covers
only `12.06%` of the sphere at one pose. A complete ERP therefore asks the system to
recover many directions that are not directly observed from that pose.

The current model family resolves missing or low-opacity content generatively:
[pipeline_base.py](../model_training/pipeline/pipeline_base.py) blends the condition
with random noise where evidence is weak. Local cleaning therefore invents plausible
content. Coherence, however, requires one persistent identity for every surface over
all directions, frames, occlusions, and revisits.

That persistent spherical/3D memory does not exist in the tested pipelines:

- historical Fixer and Qwen edit each image independently;
- ArtiFixer models a perspective video sequence, not a periodic ERP video;
- direct ERP disables the perspective-neighbor path because a single pinhole camera
  cannot represent an ERP ([run_inference.py](../model_eval/run_inference.py));
- the joint-snake manifest linearizes direction and time into one temporal axis, which
  is not the topology seen during training;
- latent or noise synchronization averages or correlates samples, but does not attach
  generated content to a canonical surface
  ([synchronized_multiview.py](../model_eval/synchronized_multiview.py));
- hard cuts expose incompatible geometry, while feathering/multiband blending creates
  blur or double surfaces and graph-cut moves the discontinuity
  ([stitch_artifixer_frustums360.py](../scripts/stitch_artifixer_frustums360.py));
- ArtiFixer3D can keep a single scene coherent, but unlocked geometry fits
  contradictory pseudo-targets; locked geometry cannot remove geometric artifacts
  ([artifixer3d.py](../data_processing/artifixer3d.py)).

Consequently, the observed frontier is structural rather than a hyperparameter issue:

- preserve the original geometry → coherent but dirty;
- let a generator replace unreliable areas → cleaner locally but inconsistent;
- merge multiple replacements → fewer visible cuts but blur, ghosts, or displaced
  seams.

Where no reliable observation exists, a result cannot simultaneously be faithful,
clean, and fully determined. Any completion must either be explicitly plausible rather
than faithful, or be supported by additional ground truth.

## Representative preserved outputs

These files were verified present locally at closure time:

- Raw/seamless ERP: `/path/to/local/data/gauvain_480k_360_equirectangular_seamless_4096x2048_15fps.mp4`
- Cubeface ArtiFixer feather: `/path/to/local/data/gauvain_480k_cubefaces_artifixer_fov110_feather180_4096x2048_15fps.mp4`
- Loop-closed cubeface: `/path/to/local/data/gauvain_480k_cubefaces_fov110_loopclosed_feather180_4096x2048_15fps.mp4`
- Rules 35k: `/path/to/local/data/gauvain_base30k_cubemap25_a3d_rules35k_feather180_4096x2048_15fps.mp4`
- Rules 90k: `/path/to/local/data/gauvain_ns14_25_rules90k_4096x2048_15fps.mp4`
- Overlap18 Fixer: `/path/to/local/data/gauvain_481k_overlap18_fixer_feather_full_1024x512_15fps.mp4`
- OmniGS: `/path/to/local/data/gauvain_481k_omnigs_erp_full154_1024x512_15fps.mp4`
- Qwen-Edit-360: `/path/to/local/data/gauvain_481k_overlap18_qwen_edit360/abci_q_output/gauvain_481k_overlap18_qwen_edit360_1024x512_15fps.mp4`
- Synchronization comparison: `/path/to/local/data/erp_fixer_models_test_20260722/videos/comparison_artifixer_sync_modes.mp4`
- Feather/multiband still: `/path/to/local/data/gauvain_481k_overlap18_fov120_qc/qc_compare_frame3.png`
- Graph-cut still: `/path/to/local/data/gauvain_481k_overlap18_fov120_qc/qc_compare_graphcut_frame3.png`

## Do not repeat

Without new ground truth or a purpose-trained model, do not reopen this campaign for:

- prompt, seed, denoising-strength, or guidance sweeps;
- more cubemap faces, frustums, overlap, feathering, multiband, or graph-cut;
- independent frame editors, including generalist 360 image editors;
- latent/noise averaging or deterministic-noise variants;
- longer ArtiFixer3D fitting, geometry-lock blends, or heuristic Gaussian pruning;
- another five-frame pilot presented as evidence for a full coherent video.

## Conditions required to reopen

Reopening is justified only by at least one material change:

1. additional real observations that cover the missing directions;
2. paired artifact/clean panoramic video data and a budget to train a model with
   spherical, temporal, identity, reprojection, and occlusion constraints;
3. a released checkpoint specifically trained for temporally coherent ERP restoration,
   validated on the entire 154-frame clip rather than isolated frames;
4. an explicit relaxation from faithful restoration to plausible scene completion.

Any reopened project must use whole-clip acceptance gates for local artifact removal,
horizontal and inter-view continuity, temporal warp/revisit consistency, and identity
outside the repair mask. Improvement on only one axis is not a pass.
