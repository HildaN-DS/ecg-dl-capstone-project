"""Unit tests for frozen CNN inference, using synthetic ECGs only.

The final integration verification against PTB-XL is in
scripts/verify_frozen_inference.py, not in this unit test module.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.ecg.inference.predictor import (
    CLASS_ORDER,
    LEAD_ORDER,
    FrozenCNNPredictor,
    _sha256_file,
)
from src.ecg.models.cnn1d import ECG1DCNN


class IdentityNormalizer:
    training_ecgs = 17084
    samples_per_lead = 17084000

    def transform(self, x):
        return x


class ConstantModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("fixed", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.fixed.unsqueeze(0).expand(len(x), -1)


@pytest.fixture
def predictor():
    return FrozenCNNPredictor(
        model=ConstantModel([2.0, 0.0, -1.0, -2.0, -3.0]),
        normalizer=IdentityNormalizer(),
        device="cpu",
        checkpoint_sha256="synthetic_checkpoint",
        normalization_sha256="synthetic_normalizer",
    )


@pytest.fixture
def ecg():
    return np.zeros((12, 1000), dtype=np.float32)


def predict_one(predictor, ecg, **changes):
    meta = {"lead_order": LEAD_ORDER, "sampling_rate_hz": 100, "units": "mV"}
    meta.update(changes)
    return predictor.predict(ecg, **meta)


def test_output_class_order_threshold_and_probabilities(predictor, ecg):
    output = predict_one(predictor, ecg)
    labels = output["predictions"]
    assert [r["class"] for r in labels] == list(CLASS_ORDER)
    assert all(r["threshold"] == 0.5 for r in labels)
    assert labels[0]["probability"] > 0.5
    assert labels[1]["predicted"] is True  # sigmoid(0) is exactly 0.5
    assert output["predicted_classes"] == ["NORM", "MI"]
    assert output["model"] == "cnn1d_baseline_v1"
    assert "not for clinical" in output["research_notice"]


def test_no_predictions_does_not_imply_normal(ecg):
    predictor = FrozenCNNPredictor(
        model=ConstantModel([-3.0] * 5),
        normalizer=IdentityNormalizer(),
        device="cpu",
        checkpoint_sha256="test",
        normalization_sha256="test",
    )
    output = predict_one(predictor, ecg)
    assert output["predicted_classes"] == []
    assert output["status"] == "no_class_above_threshold"
    assert not output["predictions"][0]["predicted"]


@pytest.mark.parametrize("shape", [(1000, 12), (12, 999), (11, 1000), (1, 12, 1000)])
def test_invalid_single_waveform_shapes(predictor, shape):
    with pytest.raises(ValueError, match="Expected"):
        predict_one(predictor, np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_reject_nonfinite_inputs(predictor, ecg, bad_value):
    ecg[0, 0] = bad_value
    with pytest.raises(ValueError, match="NaN or infinity"):
        predict_one(predictor, ecg)


def test_reject_non_numeric_inputs(predictor):
    for array in [np.full((12, 1000), "x"), np.zeros((12, 1000), dtype=bool)]:
        with pytest.raises(ValueError, match="numeric"):
            predict_one(predictor, array)


def test_reject_invalid_or_missing_metadata(predictor, ecg):
    swapped = list(LEAD_ORDER)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="lead_order"):
        predict_one(predictor, ecg, lead_order=swapped)
    with pytest.raises(ValueError, match="sampling_rate_hz"):
        predict_one(predictor, ecg, sampling_rate_hz=500)
    with pytest.raises(ValueError, match="units"):
        predict_one(predictor, ecg, units="uV")
    with pytest.raises(TypeError):
        predictor.predict(ecg)


def test_batch_predictions_and_limit(predictor, ecg):
    output = predictor.predict_batch(
        np.stack([ecg, ecg]),
        lead_order=LEAD_ORDER,
        sampling_rate_hz=100,
        units="mV",
    )
    assert len(output) == 2
    assert output[0] == output[1]
    with pytest.raises(ValueError, match="between 1 and 64"):
        predictor.predict_batch(
            np.zeros((65, 12, 1000), dtype=np.float32),
            lead_order=LEAD_ORDER, sampling_rate_hz=100, units="mV",
        )


def test_reject_nonfinite_normalizer_output(predictor, ecg):
    class BrokenNormalizer:
        def transform(self, x):
            return x * float("nan")

    predictor.normalizer = BrokenNormalizer()
    with pytest.raises(ValueError, match="Normalization returned"):
        predict_one(predictor, ecg)


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def create_fake_artifacts(root):
    checkpoint_path = root / "cnn1d_baseline_v1" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = ECG1DCNN(num_classes=5, dropout=0.3)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": "ECG1DCNN",
        "epoch": 17,
        "class_order": list(CLASS_ORDER),
        "num_classes": 5,
        "dropout": 0.3,
    }, checkpoint_path)
    normalization_path = root / "ptbxl_normalization_v1.json"
    normalization_path.write_text('{"training_stub": true}', encoding="utf-8")
    selection_path = root / "final_model_selection_v1.json"
    _write_json(selection_path, {
        "status": "frozen_before_test_evaluation",
        "selected_experiment": "cnn1d_baseline_v1",
        "selected_checkpoint": str(checkpoint_path),
        "class_order": list(CLASS_ORDER),
        "thresholds": [0.5] * 5,
        "selection_metric": "validation_macro_ap",
        "checkpoint_sha256": _sha256_file(checkpoint_path),
    })
    provenance_path = root / "final_test_cnn1d_v1" / "test_metrics.json"
    _write_json(provenance_path, {
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "normalization_sha256": _sha256_file(normalization_path),
        "selection_record_sha256": _sha256_file(selection_path),
    })
    return checkpoint_path, normalization_path


def test_load_from_frozen_artifacts_and_reject_tampering(tmp_path, monkeypatch, ecg):
    import src.ecg.inference.predictor as predictor_module
    checkpoint_path, normalization_path = create_fake_artifacts(tmp_path)
    monkeypatch.setattr(
        predictor_module.ECGStandardizer,
        "load",
        lambda _path: IdentityNormalizer(),
    )
    loaded = FrozenCNNPredictor.from_artifacts(tmp_path, device="cpu")
    output = predict_one(loaded, ecg)
    assert len(output["predictions"]) == 5
    assert loaded.model.training is False
    assert not any(p.requires_grad for p in loaded.model.parameters())

    # A single-byte change to normalization invalidates inference.
    normalization_path.write_text('changed', encoding="utf-8")
    with pytest.raises(ValueError, match="Normalization SHA-256"):
        FrozenCNNPredictor.from_artifacts(tmp_path)


def test_reject_modified_checkpoint(tmp_path):
    checkpoint_path, _ = create_fake_artifacts(tmp_path)
    with checkpoint_path.open("ab") as handle:
        handle.write(b"unauthorized change")
    with pytest.raises(ValueError, match="Checkpoint SHA-256"):
        FrozenCNNPredictor.from_artifacts(tmp_path)
