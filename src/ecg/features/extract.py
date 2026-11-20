
"""Time-domain feature extraction for 12-lead ECGs."""

import numpy as np


LEADS = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

FEATURES = (
    "mean",
    "std",
    "rms",
    "mean_abs",
    "peak_to_peak",
    "p05",
    "p95",
    "mean_abs_diff",
    "rms_diff",
)


def get_feature_names():
    """Return feature names in their extraction order."""
    return [
        f"{lead}_{feature}"
        for lead in LEADS
        for feature in FEATURES
    ]


def extract_ecg_features(waveforms):
    """Extract 108 time-domain features from 12-lead ECGs.

    Accepts (12, 1000) or (batch, 12, 1000).
    Returns (108,) or (batch, 108), respectively.
    """

    x = np.asarray(waveforms, dtype=np.float64)

    single_recording = x.ndim == 2

    if single_recording:
        x = x[np.newaxis, :, :]

    if x.ndim != 3 or x.shape[1:] != (12, 1000):
        raise ValueError(
            "Expected (12, 1000) or (batch, 12, 1000)."
        )

    if x.shape[0] == 0:
        raise ValueError("Empty ECG batch.")

    if not np.isfinite(x).all():
        raise ValueError("Non-finite waveform values.")

    differences = np.diff(x, axis=2)

    statistics = [
        x.mean(axis=2),
        x.std(axis=2, ddof=0),
        np.sqrt(np.mean(x ** 2, axis=2)),
        np.mean(np.abs(x), axis=2),
        np.ptp(x, axis=2),
        np.percentile(x, 5, axis=2),
        np.percentile(x, 95, axis=2),
        np.mean(np.abs(differences), axis=2),
        np.sqrt(np.mean(differences ** 2, axis=2)),
    ]

    features = np.stack(
        statistics,
        axis=2,
    ).reshape(x.shape[0], -1)

    features = features.astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError("Non-finite extracted features.")

    if features.shape[1] != 108:
        raise AssertionError("Unexpected feature dimension.")

    if single_recording:
        return features[0]

    return features
