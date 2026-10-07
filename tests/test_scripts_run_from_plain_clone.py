# SPDX-FileCopyrightText: Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Scripts that import repository packages must run as ``python scripts/<name>.py``.

The README and the pipeline guides show that form. On a plain clone nothing puts
the repository root on ``sys.path`` (only the cluster job scripts export
``PYTHONPATH``), so every script that imports ``scripts.*`` or ``model_eval.*``
has to add the root itself.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIRST_PARTY_IMPORT = re.compile(
    r"^(?:from|import) (?:scripts|model_eval|model_training|data_processing)\b", re.MULTILINE
)


def scripts_importing_repository_packages() -> list[Path]:
    return sorted(
        path
        for path in (REPO_ROOT / "scripts").glob("*.py")
        if FIRST_PARTY_IMPORT.search(path.read_text(encoding="utf-8"))
    )


def test_every_script_importing_repository_packages_runs_from_a_plain_clone(tmp_path):
    scripts = scripts_importing_repository_packages()
    assert {path.name for path in scripts} >= {
        "evaluate_depth_synchronized_multiview.py",
        "evaluate_temporal_panorama.py",
    }

    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    failures = []
    for script in scripts:
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if completed.returncode != 0 or "usage:" not in completed.stdout:
            failures.append(f"{script.name}: {completed.stderr.strip().splitlines()[-1:]}")

    assert not failures, "\n".join(failures)
