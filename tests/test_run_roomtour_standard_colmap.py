from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_roomtour_standard_colmap as standard


def _option(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def test_all_frame_production_commands_use_shared_camera_and_exhaustive_pairs(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    images = tmp_path / "images"
    sparse = tmp_path / "sparse"

    feature = standard.feature_extractor_arguments(
        database,
        images,
        camera_model="SIMPLE_RADIAL",
        camera_params=(960.0, 960.0, 540.0, 0.0),
    )
    matching = standard.exhaustive_matcher_arguments(database)
    mapper = standard.mapper_arguments(database, images, sparse)

    assert feature[0] == "feature_extractor"
    assert _option(feature, "--ImageReader.single_camera") == "1"
    assert _option(feature, "--ImageReader.camera_model") == "SIMPLE_RADIAL"
    assert _option(feature, "--FeatureExtraction.type") == "ALIKED_N16ROT"
    assert matching[0] == "exhaustive_matcher"
    assert _option(matching, "--FeatureMatching.type") == "ALIKED_LIGHTGLUE"
    assert "--FeatureMatching.guided_matching" not in matching
    assert mapper[0] == "mapper"
    assert "--Mapper.image_list_path" not in mapper
    assert _option(mapper, "--Mapper.multiple_models") == "0"
    assert _option(mapper, "--Mapper.ba_refine_focal_length") == "1"
    assert _option(mapper, "--Mapper.ba_refine_extra_params") == "1"


def test_optional_sequential_profile_does_not_claim_unobserved_loop(tmp_path: Path) -> None:
    arguments = standard.sequential_matcher_arguments(tmp_path / "database.db", overlap=15)
    assert arguments[0] == "sequential_matcher"
    assert _option(arguments, "--SequentialMatching.overlap") == "15"
    assert _option(arguments, "--SequentialMatching.quadratic_overlap") == "1"
    assert _option(arguments, "--SequentialMatching.loop_detection") == "0"


def test_source_loader_uses_all_frames_and_ignores_old_holdout_labels(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    records = []
    for index in range(150):
        name = f"frame_{index:06d}.png"
        payload = f"pixel-{index}".encode()
        (images / name).write_bytes(payload)
        records.append(
            {
                "index": index,
                "name": name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "appearance_holdout": index % 10 == 4,
            }
        )
    manifest = {
        "schema": "artifixer.roomtour_source_manifest_v2",
        "status": "PASS",
        "verdict": "PASS_CONTINUOUS_PERSPECTIVE_OBSERVATIONS",
        "observation_count": 150,
        "source": {
            "no_people": True,
            "no_scene_cuts": True,
            "sampling_fps": 5.0,
        },
        "records": records,
    }
    path = tmp_path / "source_manifest_v2.json"
    path.write_text(json.dumps(manifest))

    names, loaded = standard.load_source_frames(images, path)

    assert len(names) == 150
    assert names[4] == "frame_000004.png"
    assert loaded["records"][4]["appearance_holdout"] is True


def test_source_loader_rejects_unmanifested_images(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    records = []
    for index in range(150):
        name = f"frame_{index:06d}.png"
        payload = b"x"
        (images / name).write_bytes(payload)
        records.append(
            {
                "index": index,
                "name": name,
                "size_bytes": 1,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (images / "unexpected.png").write_bytes(b"x")
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "artifixer.roomtour_source_manifest_v2",
                "status": "PASS",
                "verdict": "PASS_CONTINUOUS_PERSPECTIVE_OBSERVATIONS",
                "observation_count": 150,
                "source": {
                    "no_people": True,
                    "no_scene_cuts": True,
                    "sampling_fps": 5.0,
                },
                "records": records,
            }
        )
    )

    with pytest.raises(standard.GeometryError, match="unexpected"):
        standard.load_source_frames(images, path)


def test_source_loader_accepts_a_dense_210_frame_sequence(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    records = []
    for index in range(210):
        name = f"frame_{index:06d}.png"
        payload = f"dense-{index}".encode()
        (images / name).write_bytes(payload)
        records.append(
            {
                "index": index,
                "name": name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "artifixer.roomtour_source_manifest_v2",
                "status": "PASS",
                "verdict": "PASS_CONTINUOUS_PERSPECTIVE_OBSERVATIONS",
                "observation_count": 210,
                "source": {"no_people": True, "no_scene_cuts": True, "sampling_fps": 10.0},
                "records": records,
            }
        )
    )

    names, _ = standard.load_source_frames(images, path)

    assert len(names) == 210
    assert names[-1] == "frame_000209.png"


def test_abci_job_is_isolated_and_publishes_regular_colmap_scene() -> None:
    driver = Path("scripts/run_roomtour_standard_colmap.py").read_text()
    body = Path("scripts/abci_roomtour_standard_colmap.pbs").read_text()
    assert 'solver_id = f"colmap_incremental_all_frames_{args.matching_mode}"' in driver
    assert '"solver_id": solver_id' in driver
    assert "jobs/pose_standard/$JOB_TOKEN" in body
    assert "--matching-mode exhaustive" in body
    assert "PASS_STANDARD_COLMAP_ALL_FRAMES" in body
    assert 'metrics.get("registered_images") != int(sys.argv[2])' in body
    assert '"registered_images": int(sys.argv[3])' in body
    assert 'cp -al "$IMAGES/." "$PUBLISH_STAGE/colmap_scene/images/"' in body
    assert 'ln -s "$IMAGES"' not in body
    assert 'cp -p "$SCRATCH/output/final_sparse/"* "$PUBLISH_STAGE/colmap_scene/sparse/0/"' in body
    assert "custom_holdout_rejection" in body
    assert "holdout_metrics" not in body
    assert "jobs/pose/$JOB_TOKEN" not in body
