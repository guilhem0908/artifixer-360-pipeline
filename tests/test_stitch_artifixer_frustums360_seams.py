import numpy as np

from scripts.stitch_artifixer_frustums360 import temporal_majority_labels


def test_temporal_majority_removes_one_frame_label_flicker() -> None:
    labels = np.zeros((5, 3, 4), dtype=np.uint8)
    labels[2, 1, 2] = 3
    smoothed = temporal_majority_labels(labels, radius=1)
    assert np.array_equal(smoothed, np.zeros_like(labels))
