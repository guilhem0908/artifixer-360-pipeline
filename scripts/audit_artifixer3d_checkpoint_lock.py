#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed tensor audit for two-stage spherical-loop distillation.

The density stage may only reduce resumed density. The appearance stage may
only change diffuse appearance. The audit compares checkpoint tensors directly
and emits a report only after every invariant passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


PARAMETERS = (
    "positions",
    "rotation",
    "scale",
    "density",
    "features_albedo",
    "features_specular",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    document = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(document, dict):
        raise ValueError(f"{path} is not a checkpoint mapping")
    missing = [name for name in PARAMETERS if not isinstance(document.get(name), torch.Tensor)]
    if missing:
        raise ValueError(f"{path} is missing tensor parameters: {missing}")
    return document


def tensor_equal(first: torch.Tensor, second: torch.Tensor) -> bool:
    return (
        first.shape == second.shape
        and first.dtype == second.dtype
        and torch.equal(first, second)
    )


def audit(base_path: Path, final_path: Path, mode: str) -> dict[str, Any]:
    base = load_checkpoint(base_path)
    final = load_checkpoint(final_path)
    shapes_equal = {name: base[name].shape == final[name].shape for name in PARAMETERS}
    if not all(shapes_equal.values()):
        raise ValueError(f"Gaussian tensor shapes changed: {shapes_equal}")

    equal = {name: tensor_equal(base[name], final[name]) for name in PARAMETERS}
    if mode == "density":
        locked = ("positions", "rotation", "scale", "features_albedo", "features_specular")
        violations = int(torch.count_nonzero(final["density"] > base["density"]).item())
        decreased = int(torch.count_nonzero(final["density"] < base["density"]).item())
        passed = all(equal[name] for name in locked) and violations == 0
        details = {
            "locked_tensors_bitwise_equal": {name: equal[name] for name in locked},
            "density_nonincrease_violations": violations,
            "density_decreased_elements": decreased,
            "density_unchanged_elements": int(
                torch.count_nonzero(final["density"] == base["density"]).item()
            ),
            "density_max_increase": float(
                (final["density"] - base["density"]).max().item()
            ),
        }
    elif mode == "appearance":
        locked = ("positions", "rotation", "scale", "density", "features_specular")
        passed = all(equal[name] for name in locked)
        details = {
            "locked_tensors_bitwise_equal": {name: equal[name] for name in locked},
            "features_albedo_changed": not equal["features_albedo"],
            "features_albedo_changed_elements": int(
                torch.count_nonzero(
                    final["features_albedo"] != base["features_albedo"]
                ).item()
            ),
        }
    else:
        raise ValueError(f"unsupported audit mode {mode!r}")

    report = {
        "schema": "artifixer.checkpoint_lock_audit_v1",
        "mode": mode,
        "verdict": "PASS" if passed else "FAIL",
        "base_checkpoint": str(base_path.resolve()),
        "base_sha256": sha256_file(base_path),
        "final_checkpoint": str(final_path.resolve()),
        "final_sha256": sha256_file(final_path),
        "base_global_step": int(base.get("global_step", -1)),
        "final_global_step": int(final.get("global_step", -1)),
        "gaussian_count": int(base["density"].shape[0]),
        **details,
    }
    if not passed:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--mode", choices=("density", "appearance"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.base.resolve(), args.final.resolve(), args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
