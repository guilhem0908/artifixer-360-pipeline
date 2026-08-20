from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from scripts import run_erp_benchmark_colmap as driver


def _database(path: Path, bad_pair: int | None = None) -> list[str]:
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT);"
        "CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);"
        "CREATE TABLE matches(pair_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);"
        "CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);"
    )
    names = [f"frame_{index:06d}.jpg" for index in range(driver.EXPECTED_COUNT)]
    coordinates = np.asarray(
        [[column * 120 + 20, row * 190 + 20, 1, 0] for row in range(5) for column in range(8)],
        dtype="<f4",
    )
    coordinates = np.tile(coordinates, (3, 1))
    for image_id, name in enumerate(names, start=1):
        connection.execute("INSERT INTO images VALUES(?,?)", (image_id, name))
        connection.execute(
            "INSERT INTO keypoints VALUES(?,?,?,?)",
            (image_id, len(coordinates), 4, coordinates.tobytes()),
        )
    matches = np.column_stack((np.arange(120), np.arange(120))).astype("<u4")
    for index in range(len(names) - 1):
        identifier = driver.pair_id(index + 1, index + 2)
        raw = 120
        inliers = 80 if index == bad_pair else 120
        connection.execute(
            "INSERT INTO matches VALUES(?,?,?,?)",
            (identifier, raw, 2, matches.tobytes()),
        )
        connection.execute(
            "INSERT INTO two_view_geometries VALUES(?,?,?,?)",
            (identifier, inliers, 2, matches[:inliers].tobytes()),
        )
    connection.commit()
    connection.close()
    return names


def test_consecutive_link_report_uses_true_inliers_ratio_and_spatial_coverage(tmp_path: Path) -> None:
    database = tmp_path / "database.db"
    names = _database(database)
    report = driver.consecutive_link_report(
        database,
        names,
        width=1024,
        height=1024,
        minimum_inliers=100,
        minimum_ratio=0.70,
        minimum_cells=8,
    )
    assert report["pass"] is True
    assert report["pair_count"] == 189
    assert report["failures"] == []


def test_consecutive_link_report_rejects_one_weak_link(tmp_path: Path) -> None:
    database = tmp_path / "database.db"
    names = _database(database, bad_pair=73)
    report = driver.consecutive_link_report(
        database,
        names,
        width=1024,
        height=1024,
        minimum_inliers=100,
        minimum_ratio=0.70,
        minimum_cells=8,
    )
    assert report["pass"] is False
    assert len(report["failures"]) == 1
    assert report["failures"][0]["pair"] == [names[73], names[74]]
