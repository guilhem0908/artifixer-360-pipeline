# ArtiFixer handover: ABCI workflow and off-trajectory experiments

Date: 2026-08-20 (Asia/Tokyo)
Audience: Raphael, Oka-san, and future ArtiFixer maintainers

## Executive status

The recommended reference result is the 117-frame `ytb_900` run using a
smoothed forward-facing 14-view rig, two overlapping 77-frame temporal
windows, synchronized peer conditioning, depth-aware reprojection, ERP
stitching, and fail-closed depth/detail/temporal quality gates. The final job
was `189387.qjcm`; it finished with `Exit_status=0`, walltime `00:20:21`, and
verdict `PASS_FORWARD14_DEPTHPEER77_PILOT`.

The source code and the local result videos are preserved. The ABCI runtime is
**not currently ready to reproduce that run**. A cleanup on 2026-08-19 removed
the repository, COLMAP runtime, Hugging Face cache, run directories, and the
original ArtiFixer container. A generic NGC image for another project was then
placed at the old ArtiFixer container path. Its SHA-256 is different and it has
not been validated for ArtiFixer.

Do not submit an ArtiFixer job until
`scripts/check_artifixer_abci_handover.py` returns `PASS` and a GPU smoke test
has passed. Exact bitwise reproduction additionally requires recovering the
two original pinned SIF files. Rebuilding from the current Dockerfile is a
valid recovery path, but it creates a new runtime that must be revalidated and
must not be described as the historical runtime.

No token, password, or private credential is included in this handover. Use
`hf auth whoami` to check authentication; never copy a Hugging Face token into
a script, log, document, or chat.

## What I worked on

The work extended the public ArtiFixer/3DGRUT workflow into an experimental
360-degree room-tour pipeline:

1. ingest ordered perspective observations and seal their hashes;
2. reconstruct all camera poses with COLMAP 4.1.1 and one shared camera;
3. train/render a 3DGRUT base reconstruction;
4. render novel perspective views along several trajectory organizations;
5. run ArtiFixer with temporal context and, in the latest variant, synchronized
   multiview/depth peer features;
6. stitch the corrected perspective views into ERP frames;
7. optionally distill pseudo-views into ArtiFixer3D and run ArtiFixer3D+;
8. evaluate depth overlap, retained detail, ERP seams, temporal warp/flicker,
   loop closure, and selected window boundaries;
9. publish outputs atomically only after mandatory gates pass.

The current reference path is:

```text
ordered 1920x1080 perspective frames
  -> immutable source manifest
  -> all-frame COLMAP 4.1.1 poses
  -> 30k 3DGRUT reconstruction
  -> smoothed forward-facing 14-view renders (1,638 targets)
  -> two 77-frame windows, stride 40
  -> synchronized ArtiFixer inference
  -> 14-view ERP stitch
  -> depth + detail + temporal gates
  -> atomic publication
```

This reference output is direct synchronized ArtiFixer followed by ERP
stitching. It is not the same as the separate historical
ArtiFixer -> ArtiFixer3D -> ArtiFixer3D+ distillation chain.

## Authoritative results and artifacts

### `ytb_900` reference run

The source was the first 117 images after chronological/numeric sorting of:

```text
/path/to/local/data/ytb_90/ytb_900/images
```

The complete raw archive is still local:

```text
/path/to/local/data/ytb_90.zip
SHA256 5fa37973a4b744d8530718d4d5ef10bf91e912825b5d24dbcc82ed47757444b0
```

Historical ABCI records, now removed by the cleanup:

- run root:
  `/path/to/artifixer/data/room_tour_pipeline_v2/runs/ytb900_pilot117_v1`;
- pose job `189007.qjcm`: 117/117 registered images, 24,933 points,
  median reprojection error approximately 0.952 px;
- final job `189387.qjcm`: `PASS_FORWARD14_DEPTHPEER77_PILOT`;
- historical published directory:
  `jobs/forward14_depthpeer77/189387.qjcm/published`;
- historical persistent inference cache: 1,638 validated predictions.

Preserved local result directory:

```text
/path/to/local/data/ytb_90_artifixer3d_pilot117/
```

It contains:

- `ytb900_forward_input_vs_depthpeer77_2048x512_15fps.mp4`;
- `ytb900_forward_depthpeer77_117_1024x512_15fps.mp4`;
- `ytb900_forward_input117_1024x512_15fps.mp4`;
- the depth, detail, and temporal JSON reports.

All videos contain 117 frames at 15 fps (7.8 seconds). The final QC values
were:

- temporal warp MAE median: `0.02016` output vs `0.03732` raw input;
- edge flicker median: `0.01627` output vs `0.03422` raw input;
- mandatory boundary 57 -> 58: warp MAE `0.02012`, P95 `0.06144`, valid
  fraction `0.9916`;
- high-confidence edge strength ratio median `0.4859`, per-view minimum
  `0.3653`, orientation cosine `0.9723`.

Visual conclusion: the result is much more coherent and readable than the raw
fragmented 3DGRUT render, but weakly observed lateral areas remain soft or
smeared. It is an appearance/coherence improvement, not proof of correct
geometry in unobserved regions.

### Gauvain comparison assets

- raw 3DGRUT ERP vs ArtiFixer3D+ 451k:
  <https://drive.google.com/file/d/1Lpva8kRQ4O4lVtV2GwzPpgV4b4B-DzUF/view?usp=drivesdk>;
- position-first vs direction-first ArtiFixer2D:
  <https://drive.google.com/file/d/1l80NMnOy1NTCkRCsMYD7D1IP5fnmSlmM/view?usp=drivesdk>;
- supervisor deck:
  <https://docs.google.com/presentation/d/1ZyuGAkQU--Mgiz9EyFKQZAIocbSFKFQhGaECbtd9P-w/edit>.

The Gauvain campaign is a scientific `NO-GO` for a faithful, clean,
spatially continuous, temporally stable 360-degree video under the available
observations and model weights. A job finishing successfully is not the same
as a visual/scientific pass. See
`docs/gauvain_artifixer_no_go_2026-07-22.md`.

## Canonical public source snapshot

The complete public handover is split across these two repositories:

```text
ArtiFixer 360 base release: 7e271037bd00a9091f8f2572ad3e06b69f40a205
3DGRUT-ArtiFixer-360: decee83591594d5812c30e01fa445f15560d21f3
main repository: https://github.com/guilhem0908/artifixer-360-pipeline
3DGRUT fork: https://github.com/guilhem0908/3DGRUT-ArtiFixer-360
```

Clone the main repository with `--recurse-submodules`; its submodule is pinned
to the canonical 3DGRUT fork commit above. The historical overlay inventory is
kept in `docs/artifixer_handover_overlay_files.txt` for audit purposes, but a
fresh public clone does not require extracting that overlay.

The prepared local transfer artifacts are:

```text
/path/to/local/data/ArtiFixer_handover_overlay_2026-08-20.tar.gz
/path/to/local/data/ArtiFixer_handover_overlay_2026-08-20.tar.gz.sha256
/path/to/local/data/ArtiFixer_handover_overlay_2026-08-20.files.sha256
```

The same handover, archive, checksums, preflight program, and captured ABCI
audit are installed at:

```text
/path/to/artifixer/handover/2026-08-20/
```

All `/path/to/...` values in this public document are placeholders. Set
`AFROOT` to the receiving researcher's ABCI root and `LOCAL_DATA_ROOT` to the
local input/output directory before using the commands.

## ABCI runtime: required layout and pins

Expected root:

```text
/path/to/artifixer/
  src/ArtiFixer/
  models/artifixer-cuda12.sif
  models/colmap-4.1.1-a0d785f.sif
  venvs/gauvain-colmap411/bin/python
  cache/huggingface/
  data/room_tour_pipeline_v2/runs/
  logs/
```

Pinned assets:

| Asset | Source/version | Required SHA-256 |
|---|---|---|
| ArtiFixer SIF | historical validated CUDA12 image | `226b1e00a26f16442be67157be6acb7f94ace5a698ea9c720fb7dcf89e0af132` |
| COLMAP SIF | COLMAP 4.1.1, commit `a0d785fba74b2664f31edc4a29026a8b27c00f67` | `eeb2d997f811208f5098c961ccf18e41b6cc46c2f195159a67db4a63991b38aa` |
| ArtiFixer checkpoint | `nvidia/ArtiFixer`, `artifixer-14b.pt` | `c1a6d31fb849211d4c682a28b40980549cd8f807ee309e7bc0141a336ffcd16b` |
| MoGe checkpoint | `Ruicheng/moge-2-vitl-normal`, `model.pt` | `280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01` |
| Wan caption model | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | revision `38ec498cb3208fb688890f8cc7e94ede2cbd7f68` |

Current ABCI audit on 2026-08-20:

| Component | Current state |
|---|---|
| `src/ArtiFixer` | missing |
| original ArtiFixer SIF | missing |
| substitute at the same path | present, SHA-256 `733ceadfccd75d15abc008a05911eba129d19c0f403b6d8df68ccc84cdc28a6d`; reject |
| COLMAP SIF and Python venv | missing |
| Hugging Face cache | missing |
| historical run/publication directories | missing |

The substitute is `nvcr.io/nvidia/pytorch:24.07-py3` plus patches for a
different VOD/SAM3 project. Matching the old filename does not make it an
ArtiFixer runtime.

Run the read-only audit after restoring files:

```bash
cd /path/to/artifixer/src/ArtiFixer
python3 scripts/check_artifixer_abci_handover.py \
  --afroot /path/to/artifixer
```

For a prepared run, add:

```bash
--run-dir /path/to/artifixer/data/room_tour_pipeline_v2/runs/ytb900_pilot117_v1
```

The audit hashes both large containers and exits with code 2 on any missing or
drifted dependency. A `PASS` validates the file layout and hashes only. Before
production, submit a small GPU smoke test that imports Torch, ArtiFixer,
3DGRUT, FlashAttention, MoGe, and performs one minimal forward pass.

### Recovery options

Preferred: recover the two original SIF files and the COLMAP venv/wheels from a
colleague or archive, then verify the hashes above.

Fallback: build a new runtime from the current `Dockerfile.cuda12`, export the
OCI image with `docker save`, transfer it to ABCI, and convert it with
`singularity build --fakeroot ... docker-archive://...`. The current Dockerfile
SHA-256 is
`5cb00206c89ae2d1cdfbd178743626c687cfa0b502d273844a51e97bc4073e3d`.
This is a new experimental runtime: update pins only after a GPU smoke and a
small end-to-end reference comparison. Do not use
`scripts/abci_build_colmap_image.pbs` as a reproducible recovery command: it
pulls `colmap/colmap:latest`.

## Reproducing the 117-frame reference workflow

The commands below describe the protocol. They will intentionally fail at the
preflight until the pinned ABCI runtime has been restored.

### 1. Stage the exact input selection locally

The historical selection rule was “the first 117 numerically sorted JPEGs in
`ytb_900/images`”. Materialize the PNG sequence expected by the room-tour
initializer:

```bash
export RAW_IMAGES=/path/to/local/data/ytb_90/ytb_900/images
export STAGED_FRAMES=/path/to/local/data/ytb900_pilot117_frames
mkdir -p "$STAGED_FRAMES"
python3 - "$RAW_IMAGES" "$STAGED_FRAMES" <<'PY'
import sys
from pathlib import Path
from PIL import Image

source = Path(sys.argv[1])
output = Path(sys.argv[2])
paths = sorted(source.glob("*.jpg"), key=lambda path: int(path.stem))[:117]
assert len(paths) == 117
for index, path in enumerate(paths, start=1):
    with Image.open(path) as image:
        assert image.size == (1920, 1080)
        image.convert("RGB").save(output / f"frame_{index:04d}.png")
PY
ffmpeg -nostdin -hide_banner -loglevel error -y \
  -framerate 15 -start_number 1 -i "$STAGED_FRAMES/frame_%04d.png" \
  -frames:v 117 -c:v libx264 -crf 17 -pix_fmt yuv420p \
  /path/to/local/data/ytb900_pilot117_source.mp4
```

Copy the staged frames, source video, repository overlay, and raw ZIP (or its
immutable hash record) to ABCI. Preserve the first-117 selection; do not
randomly resample it.

### 2. Initialize the immutable run on ABCI

```bash
export AFROOT=/path/to/artifixer
export REPO="$AFROOT/src/ArtiFixer"
export RUN_DIR="$AFROOT/data/room_tour_pipeline_v2/runs/ytb900_pilot117_v1"
cd "$REPO"
python3 scripts/prepare_roomtour_v2_run.py \
  --frames-dir "$AFROOT/staging/ytb900_pilot117/frames" \
  --source-video "$AFROOT/staging/ytb900_pilot117/ytb900_pilot117_source.mp4" \
  --run-dir "$RUN_DIR" \
  --source-url "local-dataset:ytb_90.zip#ytb_900" \
  --source-start-seconds 0 \
  --source-end-seconds 7.8 \
  --sampling-fps 15 \
  --expected-count 117 \
  --scene-id ytb900_pilot117
python3 scripts/check_artifixer_abci_handover.py \
  --afroot "$AFROOT" --run-dir "$RUN_DIR"
```

`prepare_roomtour_v2_run.py` refuses to overwrite an existing run. The source
contract currently records the timing label
`UNIFORM_SAMPLES_FROM_NATIVE_24_FPS` as a fixed string. For another dataset,
correct and test that provenance field instead of silently reusing it.

### 3. Submit all-frame COLMAP pose reconstruction

```bash
export DRIVER_SHA256=$(sha256sum "$REPO/scripts/run_roomtour_standard_colmap.py" | awk '{print $1}')
export SOURCE_MANIFEST_SHA256=$(sha256sum "$RUN_DIR/contracts/source_manifest_v2.json" | awk '{print $1}')
export POSE_JOB=$(qsub \
  -v ROOMTOUR_RUN_DIR="$RUN_DIR",DRIVER_SHA256="$DRIVER_SHA256",SOURCE_MANIFEST_SHA256="$SOURCE_MANIFEST_SHA256" \
  "$REPO/scripts/abci_roomtour_standard_colmap.pbs")
printf 'POSE_JOB=%s\n' "$POSE_JOB"
```

The pose job uses one `rt_QG` node, ALIKED/LightGlue exhaustive matching, a
single `SIMPLE_RADIAL` camera, and requires every frame to be registered. Its
candidate is published only after the checks pass:

```text
$RUN_DIR/jobs/pose_standard/$POSE_JOB/published/candidate.json
```

Immediately after a real submission, provide the user with an exact local
`ssh abciq` monitoring command containing the returned job ID and the exact
per-job `job.log`/`status.env` paths, refreshing every 10 seconds. `Ctrl+C`
must stop monitoring. Follow `AGENTS.md`; do not reuse an example job ID.

### 4. Chain the synchronized forward14/depthpeer77 job

The inference job can be submitted immediately with an `afterok` dependency;
the future candidate path is deterministic:

```bash
export POSE_CANDIDATE="$RUN_DIR/jobs/pose_standard/$POSE_JOB/published/candidate.json"
export PIPELINE_SCRIPT_SHA256=$(sha256sum "$REPO/scripts/abci_roomtour_forward14_depthpeer77_pilot_4gpu.pbs" | awk '{print $1}')
export INFERENCE_JOB=$(qsub \
  -W depend=afterok:"$POSE_JOB" \
  -v ROOMTOUR_RUN_DIR="$RUN_DIR",POSE_CANDIDATE="$POSE_CANDIDATE",PIPELINE_SCRIPT_SHA256="$PIPELINE_SCRIPT_SHA256",SOURCE_COUNT=117 \
  "$REPO/scripts/abci_roomtour_forward14_depthpeer77_pilot_4gpu.pbs")
printf 'INFERENCE_JOB=%s\n' "$INFERENCE_JOB"
```

This job uses one `rt_QF` node and four processes. The validated parameters are:

| Parameter | Value |
|---|---:|
| source/output frames | 117 |
| forward-facing views | 14 |
| perspective resolution/FOV | 768 x 768, 110 degrees |
| trajectory smoothing | 56 iterations, heading span 3 |
| windows | 77 frames, stride 40, overlap 37 |
| real context views | 12 |
| inference steps | 8 |
| deterministic noise seed | 0 |
| initial noise consensus | 3 iterations |
| synchronization | noise predictions, blend 0.75 |
| depth relative tolerance | 0.08 |
| required loop pair | `H000:H300` |

The job is currently specialized to 117 frames: counts, two-window assertions,
video names, QC boundary, and publication metadata are explicit. Adapting it
to another length requires code and test changes; changing only
`SOURCE_COUNT` will fail by design.

### 5. Validate and collect

After the PBS state is finished with exit status 0:

```bash
export PUBLISHED="$RUN_DIR/jobs/forward14_depthpeer77/$INFERENCE_JOB/published"
python3 - "$PUBLISHED/result.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
assert result["status"] == "PASS"
assert result["verdict"] == "PASS_FORWARD14_DEPTHPEER77_PILOT"
assert result["source_image_count"] == 117
print(json.dumps(result, indent=2, sort_keys=True))
PY
(cd "$PUBLISHED" && sha256sum -c sha256sums.txt)
```

Publication contains the input/prediction frames, three videos, the three QC
reports, trajectory/window/synchronization manifests, pose/source provenance,
and the 30k base checkpoint. Publication is an atomic directory rename and is
not created on a failed gate.

## Novel off-trajectory views: what was tried and learned

### Direction-first, position-first, and all-directions synchronization

These are distinct organizations:

- **direction-first**: process one angular lane along the full camera
  trajectory, then move to the next direction. It provides temporal continuity
  within a direction but weak coupling among directions at the same position;
- **position-first**: at each position, keep all 14 directions consecutive.
  The implemented manifest alternates the angular order at successive
  positions, uses three core positions plus one halo position on each side,
  never splits a 14-view group, and stays at 70 frames maximum under the
  77-frame model horizon. Twelve real observations are retained as context;
- **synchronized forward14**: 14 view streams share temporal windows and
  exchange reprojected peer features/noise predictions. This is the method in
  the `ytb_900` reference run; it is not merely concatenating all directions
  into one temporal sequence.

The controlled position-first vs direction-first video was more convincing for
same-position directional coherence. Position-first is therefore the better
ordering when using the unmodified 1D temporal model. Synchronized forward14
is preferred when the depth/peer extension and its QC gates are available.

### Spherical snake / winding trajectory

The snake implementation keeps world-locked angular lanes, reverses the
translation direction on alternating lanes, and inserts same-center angular
turn frames so that it does not jump abruptly from the end of one lane to the
start of the next. Relevant files are:

- `scripts/generate_spherical_snake_trajectory.py`;
- `scripts/generate_spherical_snake_inference_manifest.py`;
- `scripts/evaluate_spherical_snake_turn.py`;
- `scripts/extract_spherical_snake_lanes.py`.

This is useful as a deterministic traversal and diagnostic, and it directly
addresses the requested “perspective winding trajectory”. Its limitation is
fundamental: flattening sphere x time into one sequence gives the temporal
model a 1D neighborhood that is not its training topology. Smooth snake turns
do not create a persistent spherical surface identity. Use snake as one
trajectory variant to compare, not as evidence that every direction is jointly
consistent.

### Depth and loop conditioning

Depth-aware peer reprojection improved the synchronized pseudo-views. Before
distillation, depth-overlap MAE decreased from `0.03396` to `0.02471`
(`27.25%`) and included the horizontal closure pair `H300 <-> H000`.

The improvement did not survive clearly in the historical final distilled ERP:
the 451k baseline had temporal warp MAE `0.02329` and seam MAE `0.01143`, while
the depth+loop final had `0.02582` and `0.01174`. The correct conclusion is
“depth+loop improves pseudo-view agreement, but the present distillation can
lose that benefit”, not “depth+loop fixes geometry”.

### Main experiment matrix

| Variant | Technical status | Main finding |
|---|---|---|
| native ArtiFixer on source/novel perspective views | operational | cleans individual views; unseen regions remain generative |
| 6 cubemap faces | completed | face contradictions and visible seams |
| 14 direction-first lanes | completed | good within-lane temporal context, weak same-position coupling |
| 14 position-first groups | completed | more convincing directional coherence; best ordering without synchronized peer extension |
| spherical snake/winding | implemented and tested | smooth traversal, but invalid as a complete sphere x time topology model |
| deterministic/latent/noise synchronization | completed | reduces some local variation; no canonical surface memory |
| depth + horizontal loop peers | completed | improves pseudo-view overlap; final distillation can erase the gain |
| geometry-locked ArtiFixer3D | completed | continuous but preserves dirty geometric artifacts |
| unlocked or longer ArtiFixer3D fitting | completed | can clean locally but fits contradictory pseudo-targets |
| synchronized forward14/depthpeer77 on `ytb_900` | full 117-frame PASS | current recommended reference protocol |
| direct ERP through perspective ArtiFixer | completed | outside the model's pinhole training distribution |
| Qwen/other independent 360 editors | completed | local cleanup with temporal hallucination/identity drift |
| Rein3D integration | contract/preflight only | closest high-level design, but official compatible runtime/checkpoint assets were not available during the experiment |

The Gauvain source camera covered approximately `12.06%` of the sphere at one
position. Most off-trajectory directions therefore have no direct same-position
observation. No ordering trick can replace the missing evidence.

## Failures and workarounds worth keeping

| Problem | Symptom | Workaround / rule |
|---|---|---|
| Hugging Face quota or transient transfer | model materialization jobs fail before inference | pre-download pinned files, verify SHA-256, use a persistent cache; do not expose the token |
| model or MoGe absent on node scratch | early missing-file/import failure | materialize both assets before the expensive stages and verify their hashes |
| trajectory spike | trajectory QC rejects a large step | repair isolated pose spikes, resample arc length, then use 56 smoothing iterations and heading span 3 for the reference |
| stale peer-neighbor assertion | inference completes but post-check expects 260 instead of actual 160 | keep metadata assertions synchronized with the current neighbor graph; covered by PBS contract tests |
| one reciprocal pair dominates global depth maximum | valid run rejected by a single extreme pair | gate P95 on the edge-pair 95% quantile while retaining absolute, coverage, loop, and non-regression checks |
| good global detail hides a weak direction | aggregate detail passes while a lateral view is blurred | require minimum per-view strength ratio (`0.35`) in addition to the global median |
| scratch disappears after a QC failure | expensive 1,638 predictions are lost | publish a hash-validated immutable inference retry cache atomically before QC |
| seam hidden by feather/multiband | boundary looks softer but becomes a ghost/double surface | always evaluate hard geometric consistency; blending is presentation, not repair |
| job `Exit_status=0` mistaken for scientific success | technically complete but visually invalid outputs | require explicit QC verdicts and whole-clip visual review |
| ABCI container replaced at same filename | scripts find a SIF but run in the wrong environment | hash every runtime; filename-only checks are insufficient |

## Optional VEnhancer post-processing

VEnhancer v2 is not part of the core ABCI reference workflow. A local
post-process was tested on the 117-frame ArtiFixer output with native
1024 x 512 resolution, 15 fps, `noise_aug=35`, CFG 2, 15 fast steps, and no
frame interpolation.

A single native-yaw VEnhancer pass improves temporal stability but can destroy
the ERP longitude seam. The successful method uses two independent passes:
native yaw and a 512-pixel pre-encode yaw shift, followed by circular
raised-cosine ownership fusion, unrolling, and a 48-pixel-per-side source seam
guard. The upstream wrapper's prefix-drop behavior must be disabled; otherwise
it can remove `4 - (N mod 4)` frames.

Preserved result:

```text
/path/to/local/data/ytb_90_artifixer3d_venhancer_v2_117/
```

It achieved temporal warp MAE `0.02020 -> 0.01339` (`-33.7%`) and edge
flicker `0.01627 -> 0.01009` (`-38.0%`) while passing the guarded ERP seam
checks. It is visibly smoother and can reinterpret weak texture/geometry, so it
is a perceptual enhancement, not geometric validation. The reproducible fusion
and QC code is `scripts/build_venhancer_erp_yaw_ensemble.py`.

## Closest related approaches

- [ArtiFixer](https://arxiv.org/abs/2603.00492) is the direct basis: an
  opacity-conditioned bidirectional generator distilled into a causal
  autoregressive model, with pseudo-views reusable for 3D refinement.
- [3DiM](https://arxiv.org/abs/2210.04628) is relevant to the ordering question:
  it uses pose-conditioned diffusion and stochastic conditioning across
  available views to improve approximate 3D consistency.
- [PanoWan](https://arxiv.org/abs/2505.22016) directly addresses panoramic
  video representations with latitude-aware sampling, rotated semantic
  denoising, and longitude-aware decoding; this is closer to ERP topology than
  applying a pinhole model directly to ERP frames.
- [Imagine360](https://arxiv.org/abs/2412.03552) lifts perspective video anchors
  into 360-degree video using perspective and panorama denoising branches plus
  antipodal constraints. It is relevant when the objective is plausible 360
  completion rather than faithful reconstruction.
- [Rein3D](https://arxiv.org/abs/2604.10578) is the closest high-level pipeline:
  panoramic video restoration along exploration trajectories followed by
  refinement of a persistent 3D Gaussian field. Its paired panoramic
  degradation/restoration training is the missing ingredient in the current
  zero-shot experiments.
- [VidSplat](https://arxiv.org/abs/2605.11424) is also close conceptually: it
  iterates trajectory sampling, geometry-guided video diffusion, and
  confidence-weighted Gaussian refinement for sparse-view reconstruction.

The main gap between these approaches and the present implementation is not
just the trajectory. It is the combination of spherical training data,
geometry-aware multiview conditioning, confidence/occlusion handling, and a
persistent 3D representation that is updated only from verified evidence.

## Relevant file index

| Purpose | Main files |
|---|---|
| public setup and standard inference | `README.md`, `Dockerfile.cuda12` |
| source-run materialization | `scripts/prepare_roomtour_v2_run.py` |
| all-frame pose reconstruction | `scripts/run_roomtour_standard_colmap.py`, `scripts/abci_roomtour_standard_colmap.pbs` |
| reference 117-frame job | `scripts/abci_roomtour_forward14_depthpeer77_pilot_4gpu.pbs` |
| forward-facing 14-view rig | `scripts/generate_forward_facing14_trajectories.py` |
| synchronized windows | `scripts/prepare_depth_synchronized_multiview_windows.py` |
| synchronized inference | `model_eval/run_synchronized_multiview_inference.py`, `model_eval/synchronized_multiview.py` |
| QC and stitching | `scripts/evaluate_depth_synchronized_multiview.py`, `scripts/evaluate_high_confidence_detail.py`, `scripts/evaluate_temporal_panorama.py`, `scripts/stitch_artifixer_frustums360.py` |
| position-first ordering | `scripts/generate_position_first_artifixer_manifest.py`, `scripts/generate_viewmajor_position_first_manifest.py` |
| spherical snake | `scripts/generate_spherical_snake_trajectory.py`, `scripts/generate_spherical_snake_inference_manifest.py`, `scripts/extract_spherical_snake_lanes.py`, `scripts/evaluate_spherical_snake_turn.py` |
| full research chain including distillation | `scripts/abci_roomtour_standard_full_pipeline_4gpu.pbs` |
| seam-safe VEnhancer fusion | `scripts/build_venhancer_erp_yaw_ensemble.py` |
| runtime audit | `scripts/check_artifixer_abci_handover.py` |
| scientific closure report | `docs/gauvain_artifixer_no_go_2026-07-22.md` |

The matching tests are under `tests/`. The focused handover suite is:

```bash
python -m pytest -q \
  tests/test_check_artifixer_abci_handover.py \
  tests/test_abci_roomtour_forward14_depthpeer77_pilot.py \
  tests/test_evaluate_depth_synchronized_multiview.py \
  tests/test_evaluate_high_confidence_detail.py \
  tests/test_build_venhancer_erp_yaw_ensemble.py
```

## Suggested knowledge-transfer session

1. inspect the raw 3DGRUT vs ArtiFixer and position-first vs direction-first
   videos;
2. walk through the pose candidate, forward14 manifest, two windows, and
   synchronization metadata;
3. run the handover preflight and explain the current ABCI recovery blocker;
4. prepare a two- or 17-frame smoke run after the runtime is restored;
5. show how to monitor an actual PBS job with its exact job ID and per-job log;
6. inspect all QC JSON and distinguish technical completion from a scientific
   pass;
7. explain why new frame counts and new trajectory topology require explicit
   contract/test updates.

The minimum acceptance criterion for the transfer is that Raphael or Oka-san
can independently prepare a run, explain which inputs are real vs generated,
submit and monitor a smoke job, locate the publication, and interpret every QC
verdict without relying on undocumented paths.
