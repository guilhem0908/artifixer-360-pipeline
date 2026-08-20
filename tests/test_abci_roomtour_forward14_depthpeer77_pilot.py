import subprocess
from pathlib import Path


PBS = (
    Path(__file__).parents[1]
    / "scripts"
    / "abci_roomtour_forward14_depthpeer77_pilot_4gpu.pbs"
)


def script_text() -> str:
    return PBS.read_text(encoding="utf-8")


def test_pbs_has_valid_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(PBS)], check=True)


def test_requires_117_frame_all_image_pose_candidate() -> None:
    text = script_text()
    assert 'test "$SOURCE_COUNT" -eq 117' in text
    assert "artifixer.roomtour_standard_colmap_candidate_v1" in text
    assert "PASS_STANDARD_COLMAP_ALL_FRAMES" in text
    assert 'candidate.get("registered_images") != 117' in text
    assert 'candidate.get("all_frames_used_for_geometry") is not True' in text


def test_matches_validated_forward14_depthpeer77_protocol() -> None:
    text = script_text()
    assert "--output-frame-count 117" in text
    assert "--smooth-iterations 56 --heading-span 3" in text
    assert 'assert manifest["total_frames"] == 1638' in text
    assert "--window-size 77 --window-stride 40" in text
    assert "--num_inference_steps 8" in text
    assert "--peer-conditioning" in text
    assert "--deterministic-overlap-noise --deterministic_noise_seed 0" in text
    assert "--initial-noise-consensus-iterations 3" in text
    assert "--synchronization-target noise_predictions --synchronization-blend 0.75" in text
    assert "--required-loop-pair H000:H300" in text
    assert 'first["peer_neighbor_count"] == second["peer_neighbor_count"] == 160' in text
    assert 'first["real_neighbor_count"] == second["real_neighbor_count"] == 12' in text


def test_missing_pinned_models_are_recovered_and_hashed() -> None:
    text = script_text()
    assert "stage materialize_and_hash_pinned_model_assets" in text
    assert 'repo_id="nvidia/ArtiFixer", filename="artifixer-14b.pt"' in text
    assert 'repo_id="Ruicheng/moge-2-vitl-normal", filename="model.pt"' in text
    assert "HF_HUB_DISABLE_XET" not in text
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in text
    assert text.count("--env HF_HUB_DOWNLOAD_TIMEOUT=600") == 2
    assert 'ARTIFIXER_CHECKPOINT="$SCRATCH/models/artifixer-checkpoint/artifixer-14b.pt"' in text
    assert 'MOGE_CHECKPOINT="$SCRATCH/models/moge-2-vitl-normal/model.pt"' in text
    assert text.count('--bind "$SCRATCH:/work"') >= 3
    assert "--env TORCH_HOME=/work/torch_cache" in text
    assert "--env TORCH_EXTENSIONS_DIR=/work/torch_extensions" in text


def test_base_checkpoint_is_cached_before_later_failure_points() -> None:
    text = script_text()
    assert 'BASE_CACHE="$ROOMTOUR_RUN_DIR/artifacts/base_3dgrut/$BASE_ID"' in text
    assert 'echo "BASE_CACHE=REUSE path=$BASE_CACHE"' in text
    assert "stage publish_immutable_base_3dgrut_retry_cache" in text
    cache_stage = text.index("stage publish_immutable_base_3dgrut_retry_cache")
    trajectory_stage = text.index("stage generate_smoothed_forward_facing_14_view_trajectory")
    assert cache_stage < trajectory_stage
    assert 'ln "$BASE_CACHE_CHECKPOINT" "$PUBLISH_STAGE/checkpoints/base_3dgrut_30000.pt"' in text
    assert "c1a6d31fb849211d4c682a28b40980549cd8f807ee309e7bc0141a336ffcd16b" in text
    assert "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01" in text
    assert 'sha256sum "$ARTIFIXER_CHECKPOINT"' in text
    assert 'sha256sum "$MOGE_CHECKPOINT"' in text


def test_synchronized_predictions_are_cached_before_qc() -> None:
    text = script_text()
    assert 'INFERENCE_CACHE="$ROOMTOUR_RUN_DIR/artifacts/synchronized_inference/' in text
    assert 'echo "INFERENCE_CACHE=REUSE path=$INFERENCE_CACHE"' in text
    assert "stage publish_immutable_synchronized_inference_retry_cache" in text
    cache_stage = text.index("stage publish_immutable_synchronized_inference_retry_cache")
    qc_stage = text.index("stage depth_detail_and_temporal_qc")
    assert cache_stage < qc_stage
    assert 'assert manifest["prediction_count"] == 1638' in text
    assert '--max-prediction-p95 0.50' in text
    assert '--max-prediction-p95-q95 0.30' in text
    assert '--minimum-strength-ratio 0.45' in text
    assert '--minimum-per-view-strength-ratio 0.35' in text


def test_publishes_qc_and_comparison_video_atomically() -> None:
    text = script_text()
    for report in (
        "depth_overlap_qc.json",
        "high_confidence_detail_qc.json",
        "temporal_qc.json",
    ):
        assert report in text
    assert "ytb900_forward_input_vs_depthpeer77_2048x512_15fps.mp4" in text
    assert 'mv "$PUBLISH_STAGE" "$PUBLISHED"' in text
    assert "PASS_FORWARD14_DEPTHPEER77_PILOT" in text
