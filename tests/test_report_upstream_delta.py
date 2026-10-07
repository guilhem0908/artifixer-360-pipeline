# SPDX-FileCopyrightText: Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from scripts.report_upstream_delta import (
    AREAS,
    INTERNSHIP_HEAD,
    UPSTREAM_BASE,
    area_of,
    commit_exists,
    measure,
    to_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "docs" / "results" / "upstream_delta.json"


def test_paths_are_assigned_to_the_expected_area():
    assert area_of("scripts/stitch_artifixer_frustums360.py") == "pipeline_scripts"
    assert area_of("scripts/abci_roomtour_standard_colmap.pbs") == "cluster_jobs"
    assert area_of("model_eval/synchronized_multiview.py") == "model_and_data_code"
    assert area_of("model_training/net/transformer.py") == "model_and_data_code"
    assert area_of("data_processing/artifixer3d.py") == "model_and_data_code"
    assert area_of("tests/test_panorama_metrics.py") == "tests"
    assert area_of("docs/HANDOVER.md") == "documentation"
    assert area_of("presentations/clip_a_seams/validate_framework_pipeline.py") == "diagram_tooling"
    assert area_of("README.md") == "top_level"
    assert area_of("configs/artifixer360.example.env") == "top_level"


def test_area_keys_are_unique_and_end_with_a_catch_all():
    keys = [key for key, _, _ in AREAS]

    assert len(keys) == len(set(keys))
    assert AREAS[-1][2] == ""


def test_committed_report_is_internally_consistent():
    report = json.loads(COMMITTED.read_text(encoding="utf-8"))
    totals = report["totals"]
    areas = report["areas"].values()

    assert report["base_commit"] == UPSTREAM_BASE
    assert report["head_commit"] == INTERNSHIP_HEAD
    assert totals["files_changed"] == totals["files_added"] + totals["files_modified"] + totals["files_deleted"]
    assert totals["lines_added"] == sum(area["lines_added"] for area in areas)
    assert totals["lines_deleted"] == sum(area["lines_deleted"] for area in areas)
    assert totals["files_added"] == sum(area["files_added"] for area in areas)
    assert len(report["modified_upstream_files"]) == totals["files_modified"]


def test_committed_report_matches_the_git_history():
    if not (commit_exists(UPSTREAM_BASE) and commit_exists(INTERNSHIP_HEAD)):
        pytest.skip("full Git history is not available (shallow clone)")

    assert measure(UPSTREAM_BASE, INTERNSHIP_HEAD) == json.loads(COMMITTED.read_text(encoding="utf-8"))


def test_readme_table_is_the_generated_table():
    report = json.loads(COMMITTED.read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for line in to_markdown(report).splitlines():
        assert line in readme, line
    assert report["base_commit"][:7] in readme
    assert f"{report['new_test_modules']} new test modules" in readme
    assert f"{report['test_functions_in_new_modules']} test functions" in readme
