# SPDX-FileCopyrightText: Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.build_readme_comparison import (
    CAPTION_BAND_COLOR,
    CAPTION_BAND_HEIGHT,
    DEFAULT_STILL,
    DEFAULT_WEBP,
    build_caption_band,
    relabel,
)


def test_caption_band_has_text_in_both_halves_and_dark_margins():
    band = build_caption_band(2048, CAPTION_BAND_HEIGHT)

    assert band.shape == (CAPTION_BAND_HEIGHT, 2048, 3)
    background = np.array(CAPTION_BAND_COLOR, dtype=np.uint8)
    is_text = (band != background).any(axis=-1)
    assert is_text[:, :1024].any()
    assert is_text[:, 1024:].any()
    # Captions are centred in each half: nothing may touch the outer borders
    # or cross the middle of the side-by-side frame.
    assert not is_text[:, :16].any()
    assert not is_text[:, -16:].any()
    assert not is_text[:, 1016:1032].any()


def test_relabel_only_replaces_the_caption_rows():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(512, 2048, 3), dtype=np.uint8)
    band = build_caption_band(2048, CAPTION_BAND_HEIGHT)

    output = relabel(frame, band)

    assert np.array_equal(output[:CAPTION_BAND_HEIGHT], band)
    assert np.array_equal(output[CAPTION_BAND_HEIGHT:], frame[CAPTION_BAND_HEIGHT:])
    assert output is not frame


def test_relabel_rejects_a_band_of_the_wrong_width():
    frame = np.zeros((512, 1024, 3), dtype=np.uint8)
    band = build_caption_band(2048, CAPTION_BAND_HEIGHT)

    with pytest.raises(ValueError, match="width"):
        relabel(frame, band)


def test_versioned_readme_preview_is_a_small_looping_animation():
    assert DEFAULT_WEBP.is_file()
    assert DEFAULT_WEBP.stat().st_size < 3 * 1024 * 1024
    with Image.open(DEFAULT_WEBP) as animation:
        assert animation.format == "WEBP"
        assert animation.n_frames == 154
        assert animation.info.get("loop") == 0
        assert animation.size[0] == 4 * animation.size[1]
    with Image.open(DEFAULT_STILL) as still:
        assert still.size == (1600, 400)


def test_default_assets_live_under_the_readme_asset_directory():
    root = Path(__file__).resolve().parents[1]
    assert DEFAULT_WEBP.parent == root / "docs" / "assets" / "readme"
    assert DEFAULT_STILL.parent == DEFAULT_WEBP.parent
