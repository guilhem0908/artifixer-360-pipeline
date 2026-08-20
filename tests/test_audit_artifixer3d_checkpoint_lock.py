from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.audit_artifixer3d_checkpoint_lock import audit


def checkpoint(
    path: Path, *, density_delta: float = 0.0, albedo_delta: float = 0.0
) -> None:
    torch.save(
        {
            "global_step": 30000,
            "positions": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "rotation": torch.zeros(4, 4),
            "scale": torch.ones(4, 3),
            "density": torch.ones(4, 1) + density_delta,
            "features_albedo": torch.zeros(4, 3) + albedo_delta,
            "features_specular": torch.zeros(4, 3),
        },
        path,
    )


def test_density_audit_accepts_only_nonincrease(tmp_path: Path) -> None:
    base, final = tmp_path / "base.pt", tmp_path / "final.pt"
    checkpoint(base)
    checkpoint(final, density_delta=-0.1)
    report = audit(base, final, "density")
    assert report["verdict"] == "PASS"
    assert report["density_decreased_elements"] == 4
    checkpoint(final, density_delta=0.1)
    with pytest.raises(RuntimeError, match="density_nonincrease_violations"):
        audit(base, final, "density")


def test_appearance_audit_locks_geometry_and_density(tmp_path: Path) -> None:
    base, final = tmp_path / "base.pt", tmp_path / "final.pt"
    checkpoint(base)
    checkpoint(final, albedo_delta=0.1)
    report = audit(base, final, "appearance")
    assert report["verdict"] == "PASS"
    assert report["features_albedo_changed"] is True
    checkpoint(final, density_delta=-0.1, albedo_delta=0.1)
    with pytest.raises(RuntimeError, match='"density": false'):
        audit(base, final, "appearance")
