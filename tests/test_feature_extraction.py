
"""Tests for ECG time-domain feature extraction."""

import numpy as np
import pytest

from src.ecg.features.extract import (
    extract_ecg_features,
    get_feature_names,
)


def test_feature_extraction():
    rng = np.random.default_rng(42)

    x = rng.normal(
        size=(2, 12, 1000)
    ).astype(np.float32)

    original = x.copy()

    features = extract_ecg_features(x)

    assert features.shape == (2, 108)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()

    single = extract_ecg_features(x[0])

    assert single.shape == (108,)

    np.testing.assert_allclose(
        single,
        features[0],
        rtol=1e-6,
        atol=1e-6,
    )

    names = get_feature_names()

    assert len(names) == 108
    assert len(set(names)) == 108
    assert names[0] == "I_mean"
    assert names[8] == "I_rms_diff"
    assert names[9] == "II_mean"

    assert np.array_equal(x, original)


def test_invalid_waveform_shape():
    with pytest.raises(ValueError):
        extract_ecg_features(
            np.zeros((12, 500))
        )


def test_nonfinite_waveform():
    x = np.zeros((12, 1000), dtype=np.float32)
    x[0, 0] = np.nan

    with pytest.raises(ValueError):
        extract_ecg_features(x)


def test_empty_batch():
    with pytest.raises(ValueError):
        extract_ecg_features(
            np.empty((0, 12, 1000))
        )
