#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Create one contiguous link view of sharded spherical-snake predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make `python scripts/<name>.py` work on a plain clone: the repository root must
# be importable for the `scripts.*` / `model_eval.*` imports below.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.extract_spherical_snake_lanes import prediction_index  # noqa: E402


def materialize(
    input_root: Path,
    output_dir: Path,
    expected_count: int,
    *,
    symlink: bool = True,
) -> dict[str, object]:
    predictions = prediction_index(input_root)
    expected = list(range(expected_count))
    if sorted(predictions) != expected:
        missing = sorted(set(expected) - set(predictions))
        extra = sorted(set(predictions) - set(expected))
        raise ValueError(f"prediction indices mismatch: missing={missing[:10]} extra={extra[:10]}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to replace non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in expected:
        destination = output_dir / f"{index:05d}.png"
        source = predictions[index].resolve()
        if symlink:
            destination.symlink_to(source)
        else:
            os.link(source, destination)
    return {
        "schema_version": 1,
        "layout": "contiguous_spherical_snake_predictions",
        "input_root": str(input_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "count": expected_count,
        "link_mode": "symlink" if symlink else "hardlink",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--hardlink", dest="symlink", action="store_false")
    args = parser.parse_args()
    if args.expected_count <= 0:
        raise SystemExit("error: expected-count must be positive")
    result = materialize(
        args.input_root,
        args.output_dir,
        args.expected_count,
        symlink=args.symlink,
    )
    manifest = args.manifest or args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(f"predictions={result['count']} output={result['output_dir']}")


if __name__ == "__main__":
    main()
