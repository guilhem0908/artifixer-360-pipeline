import subprocess
from pathlib import Path


PBS = Path(__file__).parents[1] / "scripts" / "abci_roomtour_standard_full_pipeline_4gpu.pbs"


def script_text() -> str:
    return PBS.read_text()


def test_full_pipeline_pbs_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(PBS)], check=True)


def test_requires_the_exact_standard_all_frame_colmap_candidate():
    text = script_text()
    assert 'artifixer.roomtour_standard_colmap_candidate_v1' in text
    assert 'PASS_STANDARD_COLMAP_ALL_FRAMES' in text
    assert 'candidate.get("all_frames_used_for_geometry") is not True' in text
    assert 'candidate.get("custom_holdout_rejection") is not False' in text
    assert 'candidate.get("registered_images", -1)) != expected_count' in text
    assert ': "${SOURCE_COUNT:?SOURCE_COUNT is required}"' in text


def test_pipeline_is_scratch_first_and_publishes_only_compact_artifacts():
    text = script_text()
    assert 'MOGE_CHECKPOINT=${MOGE_CHECKPOINT:-$AFROOT/models/moge-2-vitl-normal/model.pt}' in text
    assert 'MOGE_CHECKPOINT=' in text
    assert '--bind "$MOGE_CHECKPOINT:/models/moge.pt:ro"' in text
    assert '--env MOGE_MODEL_PATH=/models/moge.pt' in text
    assert 'test -s "$MOGE_CHECKPOINT"' in text
    assert 'MOGE_SNAPSHOT=' not in text
    assert '--env HF_XET_CACHE=/work/hf_xet_cache' in text
    assert '"$SCRATCH/hf_xet_cache"' in text
    assert 'SCRATCH="$PBS_LOCALDIR/roomtour_full_$JOB_TOKEN"' in text
    assert '--bind "$SCRATCH:/work"' in text
    assert '--output_root /work/data/$BASE_ID' in text
    assert '--output_root /work/data/$SNAKE_ID/distill_locked35k' in text
    assert 'mv "$PUBLISH_STAGE" "$PUBLISHED"' in text
    assert 'cp -p "$SCRATCH/panoramas/artifixer3d_plus/frames/"*.png' in text
    assert 'cp -p "$BASE_CHECKPOINT"' in text
    assert 'cp -p "$DISTILLED_CHECKPOINT"' in text


def test_pipeline_keeps_the_intended_geometry_and_spherical_snake_contract():
    text = script_text()
    assert '--reconstruction_steps 30000' in text
    assert '--selected_image_names_file /work/all_images.txt' in text
    assert '--frame-count "$SOURCE_COUNT"' in text
    assert 'SNAKE_TARGET_COUNT=$((RIG_TARGET_COUNT + SNAKE_TURN_COUNT))' in text
    assert 'assert snake["target_count"] == target_count' in text
    assert 'assert rig["total_frames"] == rig_count' in text
    assert '--context-views 12 --core-size 49 --halo-size 12 --vae-phase 4' in text
    assert '--maximum-item-frames 77' in text
    assert '--trajectory /work/trajectories/$SNAKE_ID/snake.json --translation-break-ratio 5' in text
    assert '--trajectory "$SNAKE_TRAJECTORY"' not in text
    assert '--inference_pipeline bidirectional --context_parallel_size 4' in text
    assert '--geometry_locked' in text
    assert '--depth_guided --depth_loss_weight 0.5' in text
    assert '--color_loop_guided --color_loop_loss_weight 0.02' in text
    assert '--real_samples_per_override 3' in text
    assert '--artifixer3d_steps 35000' in text


def test_distillation_and_render_use_separate_cuda_processes():
    text = script_text()
    assert '--phases distill --replace --no-use_wandb' in text
    assert 'stage geometry_locked_distillation_gpu_reset_and_snake_render' in text
    assert '--phases render,prepare_artifixer3d_plus --replace --no-use_wandb' in text
    assert '--phases distill,render,prepare_artifixer3d_plus' not in text


def test_caption_uses_the_exact_wan_t2v_revision_and_frozen_repo_materializer():
    text = script_text()
    assert "Wan-AI/Wan2.1-T2V-14B-Diffusers" in text
    assert "Wan2.1-I2V" not in text
    assert "38ec498cb3208fb688890f8cc7e94ede2cbd7f68" in text
    assert 'PYTHONPATH="$RUN_REPO" python3 -m scripts.materialize_spherical_snake_predictions' in text
    assert '--expected-count "$SNAKE_TARGET_COUNT" --hardlink' in text
    assert text.count('--output-dir "$AF2D/view_major" --hardlink') == 1
    assert text.count('--output-dir "$AFPLUS/view_major" --hardlink') == 1
    assert 'ln -s "$SNAKE_RENDER' not in text


def test_qc_is_recorded_but_does_not_reintroduce_a_custom_rejection_gate():
    text = script_text()
    function = text.split("qc_snake_nonblocking() {", 1)[1].split("\n}\n", 1)[0]
    assert "set +e" in function
    assert "set -e" in function
    assert "QC_EXIT" in function
    assert 'exit "$QC_EXIT"' not in text
    assert '"qc_is_diagnostic_not_a_holdout_gate": True' in text


def test_publishes_four_dynamic_stage_videos_and_one_comparison():
    text = script_text()
    for name in (
        "raw_3dgrut_1024x512_15fps.mp4",
        "artifixer2d_1024x512_15fps.mp4",
        "distilled_locked35k_1024x512_15fps.mp4",
        "artifixer3d_plus_1024x512_15fps.mp4",
        "comparison_raw_2d_distilled_plus_4096x512_15fps.mp4",
    ):
        assert name in text
    assert 'assert int(stream["nb_read_frames"]) == expected_frames' in text
    assert '-frames:v "$SOURCE_COUNT"' in text
    assert 'test "$(find "$PUBLISH_STAGE/videos" -maxdepth 1 -type f -name \'*.mp4\' | wc -l)" -eq 5' in text


def test_container_stitching_is_encoded_by_host_ffmpeg():
    text = script_text()
    assert "encode_erp_video()" in text
    assert "-framerate 15 -start_number 0" in text
    assert text.count("encode_erp_video \"") == 4
    assert "--output-video /work/" not in text


def test_each_world_locked_erp_is_reoriented_to_the_source_camera():
    text = script_text()
    assert text.count(
        "container_cpu python /workspace/artifixer/scripts/reorient_erp_to_source_camera.py"
    ) == 4
    assert text.count("--reference-frame 0") == 5  # snake generation plus four ERP passes
    assert text.count("nerfstudio/transforms.json") >= 6
    for stage in ("raw", "artifixer2d", "distilled", "artifixer3d_plus"):
        assert f"/work/panoramas/{stage}_world/frames" in text
        assert f"/work/panoramas/{stage}/frames" in text
        assert f'{stage}_source_camera_alignment.json' in text
    assert '"erp_orientation": "per_frame_source_camera_optical_axis_and_up"' in text
