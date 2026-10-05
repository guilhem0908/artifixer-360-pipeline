# SPDX-FileCopyrightText: Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Keep the README results table tied to the dated experiment records."""

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTED = json.loads((ROOT / "docs" / "results" / "reported_metrics.json").read_text(encoding="utf-8"))
METRICS = {metric["id"]: metric for metric in REPORTED["metrics"]}
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _number(text: str) -> float:
    return float(text.replace(",", "").replace("%", ""))


@pytest.mark.parametrize("metric_id", sorted(METRICS))
def test_every_reported_value_is_in_its_source_record(metric_id):
    metric = METRICS[metric_id]
    source = (ROOT / metric["source"]).read_text(encoding="utf-8")

    for name, value in metric["values"].items():
        assert value in source, f"{metric_id}.{name}={value!r} is not in {metric['source']}"


@pytest.mark.parametrize("metric_id", sorted(METRICS))
def test_every_reported_value_is_quoted_in_the_readme(metric_id):
    metric = METRICS[metric_id]

    for name, value in metric["values"].items():
        assert value in README, f"{metric_id}.{name}={value!r} is not in README.md"
    assert metric["source"] in README, f"README.md does not link {metric['source']}"


def test_metric_ids_are_unique_and_sources_exist():
    assert len(METRICS) == len(REPORTED["metrics"])
    for metric in METRICS.values():
        assert (ROOT / metric["source"]).is_file()
        assert metric["meaning"].strip()


def test_recorded_relative_changes_agree_with_the_recorded_values():
    depth = METRICS["depth_overlap_before_distillation"]["values"]
    change = 100 * (1 - _number(depth["with_depth_sync"]) / _number(depth["without_depth_sync"]))
    # The record gives 27.25 %, computed before the two values were rounded.
    assert change == pytest.approx(_number(depth["relative_change_recorded"]), abs=0.05)

    venhancer = METRICS["venhancer_two_yaw_postprocess"]["values"]
    warp = 100 * (_number(venhancer["temporal_warp_mae_after"]) / _number(venhancer["temporal_warp_mae_before"]) - 1)
    flicker = 100 * (_number(venhancer["edge_flicker_after"]) / _number(venhancer["edge_flicker_before"]) - 1)
    assert warp == pytest.approx(_number(venhancer["temporal_warp_relative_change"]), abs=0.05)
    assert flicker == pytest.approx(_number(venhancer["edge_flicker_relative_change"]), abs=0.05)


def test_negative_results_are_reported_as_negative():
    distilled = METRICS["depth_gain_after_distillation"]["values"]
    assert _number(distilled["depth_loop_temporal_warp_mae"]) > _number(distilled["baseline_451k_temporal_warp_mae"])
    assert _number(distilled["depth_loop_seam_mae"]) > _number(distilled["baseline_451k_seam_mae"])

    cubemap = METRICS["per_face_cubemap_repair"]["values"]
    assert _number(cubemap["per_face_artifixer_mean"]) > _number(cubemap["raw_cubemap_mean"])

    full_run = METRICS["full_run_154_frames"]["values"]
    assert _number(full_run["max_temporal_warp_mae"]) > _number(full_run["max_temporal_warp_mae_threshold"])
    assert _number(full_run["max_temporal_p95"]) > _number(full_run["max_temporal_p95_threshold"])
    assert full_run["final_verdict"].startswith("QC_FAIL")


def test_source_camera_sphere_coverage_is_recomputed_from_the_intrinsics():
    values = METRICS["source_camera_sphere_coverage"]["values"]
    width, height, focal = 1972, 1087, _number(values["focal_length_px"])

    half_h = math.atan((width / 2) / focal)
    half_v = math.atan((height / 2) / focal)
    fraction = 4 * math.asin(math.sin(half_h) * math.sin(half_v)) / (4 * math.pi)

    assert f"{100 * fraction:.2f}%" == values["sphere_fraction"]
    assert (
        f"{math.degrees(2 * half_h):.2f}° × {math.degrees(2 * half_v):.2f}°" == values["field_of_view_degrees"]
    )
