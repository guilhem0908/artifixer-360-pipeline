# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from scripts.validate_artifixer360_install import REQUIRED_FILES, validate


def test_validator_requires_complete_release_tree(tmp_path: Path, monkeypatch) -> None:
    for relative in REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test\n")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr("importlib.util.find_spec", lambda _: object())
    report = validate(tmp_path)
    assert report["verdict"] == "PASS"


def test_validator_fails_when_submodule_is_missing(tmp_path: Path, monkeypatch) -> None:
    for relative in REQUIRED_FILES[:-1]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test\n")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr("importlib.util.find_spec", lambda _: object())
    report = validate(tmp_path)
    assert report["verdict"] == "FAIL"
    assert report["checks"]["files"][REQUIRED_FILES[-1]] is False
