# SPDX-FileCopyrightText: Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from scripts.render_rig_coverage_figure import (
    DEFAULT_FIGURE,
    DEFAULT_REPORT,
    analytic_source_fraction,
    build_projector,
    measure,
    solid_angle_weights,
    source_footprint,
)


@pytest.fixture(scope="module")
def measured():
    projector = build_projector(size=768, fov_degrees=110.0, erp_width=1024)
    report, coverage = measure(projector, 768, 110.0)
    return projector, report, coverage


def test_solid_angle_weights_sum_to_one_and_vanish_at_the_poles():
    weights = solid_angle_weights(64, 32)

    assert weights.shape == (32, 64)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0, 0] < weights[16, 0]
    assert np.allclose(weights[:, 0], weights[:, 63])


def test_source_camera_share_matches_the_closed_form():
    weights = solid_angle_weights(1024, 512)
    on_grid = float((source_footprint(1024, 512) * weights).sum())

    assert analytic_source_fraction() == pytest.approx(0.1206, abs=5e-5)
    assert on_grid == pytest.approx(analytic_source_fraction(), abs=1e-3)


def test_every_direction_is_seen_by_at_least_two_rig_views(measured):
    _, report, coverage = measured

    assert coverage.shape == (512, 1024)
    assert coverage.min() == 2
    assert report["views_per_direction"]["min"] == 2
    assert report["sphere_share_seen_by_at_least"]["2"] == pytest.approx(1.0)
    assert sum(report["ownership_share"].values()) == pytest.approx(1.0, abs=1e-3)


def test_committed_report_matches_a_fresh_measurement(measured):
    _, report, _ = measured
    committed = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))

    assert committed["rig"] == report["rig"]
    assert committed["erp_grid"] == report["erp_grid"]
    assert committed["views_per_direction"]["min"] == report["views_per_direction"]["min"]
    assert committed["views_per_direction"]["max"] == report["views_per_direction"]["max"]
    assert committed["views_per_direction"]["solid_angle_weighted_mean"] == pytest.approx(
        report["views_per_direction"]["solid_angle_weighted_mean"], abs=1e-3
    )
    for count, share in report["sphere_share_seen_by_at_least"].items():
        assert committed["sphere_share_seen_by_at_least"][count] == pytest.approx(share, abs=1e-3)
    assert committed["overlapping_view_pairs"] == report["overlapping_view_pairs"]
    assert committed["source_camera"]["sphere_share_analytic"] == pytest.approx(
        report["source_camera"]["sphere_share_analytic"], abs=1e-4
    )


def test_figure_is_versioned():
    assert DEFAULT_FIGURE.is_file()
    assert DEFAULT_FIGURE.stat().st_size > 10_000
