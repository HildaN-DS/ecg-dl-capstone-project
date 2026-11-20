"""Inference for the frozen, patient-disjoint PTB-XL CNN Baseline v1.

Input contract
--------------
A single ECG has 12 leads in the exact PTB-XL order, 1,000 samples/lead,
100 Hz, and amplitude values in millivolts. This module intentionally does
not infer, reorder, resample, or convert input units.

Model/normalization/decision thresholds are loaded from the frozen experiment
and verified against the selection record and final audit fingerprints.
This is a research-only classifier, NOT a clinical diagnostic device.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.ecg.features.normalize import ECGStandardizer
from src.ecg.models.cnn1d import ECG1DCNN

CLASS_ORDER = ("NORM", "MI", "STTC", "CD", "HYP")
LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SAMPLE_RATE_HZ = 100
SAMPLES_PER_LEAD = 1000
THRESHOLDS = (0.5, 0.5, 0.5, 0.5, 0.5)
MODEL_NAME = "cnn1d_baseline_v1"
EXPECTED_EPOCH = 17
EXPECTED_TRAIN_ECGS = 17084
EXPECTED_TRAIN_SAMPLES_PER_LEAD = 17084000
RESEARCH_NOTICE = "Research prototype only; not for clinical diagnosis or treatment."


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        obj = json.load(stream)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return obj


def _validate_metadata(*, lead_order: Sequence[str], sampling_rate_hz: int, units: str) -> None:
    if tuple(lead_order) != LEAD_ORDER:
        raise ValueError(
            "lead_order must be explicitly provided in the exact sequence: "
            + ", ".join(LEAD_ORDER)
        )
    if type(sampling_rate_hz) is not int or sampling_rate_hz != SAMPLE_RATE_HZ:
        raise ValueError("sampling_rate_hz must be exactly 100 (no automatic resampling)")
    if units != "mV":
        raise ValueError("units must be 'mV' (no automatic unit conversion)")


def _waveform_array(waveforms: Any, *, batch: bool) -> np.ndarray:
    try:
        array = np.asarray(waveforms)
    except (ValueError, TypeError) as exc:
        raise ValueError("ECG waveform must be a rectangular numeric array") from exc
    shape = (None, 12, 1000) if batch else (12, 1000)
    if array.ndim != len(shape) or any(
        actual != expected for actual, expected in zip(array.shape, shape)
        if expected is not None
    ):
        raise ValueError(
            f"Expected {'(batch, 12, 1000)' if batch else '(12, 1000)'}; "
            f"received {tuple(array.shape)}"
        )
    if batch and (array.shape[0] < 1 or array.shape[0] > 64):
        raise ValueError("Batch must contain between 1 and 64 recordings")
    if array.dtype.kind not in "fiu":
        raise ValueError("ECG waveform values must be numeric, not strings or booleans")
    if not np.isfinite(array).all():
        raise ValueError("ECG waveform contains NaN or infinity")
    # float32 is the exact dtype used in the waveform cache and CNN training.
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("ECG waveform values are outside float32 range")
    return np.ascontiguousarray(array)


class FrozenCNNPredictor:
    """Immutable research inference wrapper around the selected epoch-17 CNN."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        normalizer: ECGStandardizer,
        device: str | torch.device,
        checkpoint_sha256: str,
        normalization_sha256: str,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.normalizer = normalizer
        self.checkpoint_sha256 = checkpoint_sha256
        self.normalization_sha256 = normalization_sha256

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "FrozenCNNPredictor":
        """Load only the selected CNN after verifying frozen artifact hashes.

        artifacts_dir is e.g. ROOT / 'artifacts'. The final test report is read
        only for immutable provenance SHA256 values; no test labels or model
        performance data are consumed by this inference path.
        """
        root = Path(artifacts_dir)
        selection_path = root / "final_model_selection_v1.json"
        provenance_path = root / "final_test_cnn1d_v1" / "test_metrics.json"
        checkpoint_path = root / MODEL_NAME / "best_model.pt"
        normalization_path = root / "ptbxl_normalization_v1.json"

        selection = _read_json(selection_path)
        provenance = _read_json(provenance_path)

        if selection.get("status") != "frozen_before_test_evaluation":
            raise ValueError("Model-selection record was not frozen")
        if selection.get("selected_experiment") != MODEL_NAME:
            raise ValueError("Frozen model is not CNN Baseline v1")
        if selection.get("class_order") != list(CLASS_ORDER):
            raise ValueError("Unexpected class order in selection record")
        if selection.get("thresholds") != list(THRESHOLDS):
            raise ValueError("Frozen thresholds do not match the evaluated configuration")
        if selection.get("selection_metric") != "validation_macro_ap":
            raise ValueError("Unexpected model-selection criterion")
        if Path(selection.get("selected_checkpoint", "")).parts[-2:] != (MODEL_NAME, "best_model.pt"):
            raise ValueError("Unexpected checkpoint path in selection record")

        expected_checkpoint_hash = selection.get("checkpoint_sha256")
        if _sha256_file(checkpoint_path) != expected_checkpoint_hash:
            raise ValueError("Checkpoint SHA-256 differs from frozen selection record")
        if provenance.get("checkpoint_sha256") != expected_checkpoint_hash:
            raise ValueError("Checkpoint SHA-256 differs from final evaluation provenance")
        if provenance.get("selection_record_sha256") != _sha256_file(selection_path):
            raise ValueError("Model-selection record differs from final evaluation provenance")
        expected_norm_hash = provenance.get("normalization_sha256")
        if not expected_norm_hash or _sha256_file(normalization_path) != expected_norm_hash:
            raise ValueError("Normalization SHA-256 differs from final evaluation provenance")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid CNN checkpoint")
        if checkpoint.get("model_name") != "ECG1DCNN":
            raise ValueError("Unexpected model architecture")
        if checkpoint.get("epoch") != EXPECTED_EPOCH:
            raise ValueError("Unexpected checkpoint epoch")
        if checkpoint.get("class_order") != list(CLASS_ORDER):
            raise ValueError("Unexpected checkpoint class order")
        if checkpoint.get("num_classes") != len(CLASS_ORDER):
            raise ValueError("Unexpected checkpoint number of classes")
        if checkpoint.get("dropout") != 0.3:
            raise ValueError("Unexpected checkpoint dropout configuration")

        model = ECG1DCNN(num_classes=5, dropout=0.3)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if sum(param.numel() for param in model.parameters()) != 38597:
            raise ValueError("Unexpected CNN parameter count")

        normalizer = ECGStandardizer.load(normalization_path)
        if (
            normalizer.training_ecgs != EXPECTED_TRAIN_ECGS
            or normalizer.samples_per_lead != EXPECTED_TRAIN_SAMPLES_PER_LEAD
        ):
            raise ValueError("Normalization was not fitted on the frozen training partition")

        return cls(
            model=model,
            normalizer=normalizer,
            device=device,
            checkpoint_sha256=expected_checkpoint_hash,
            normalization_sha256=expected_norm_hash,
        )

    def predict_batch(
        self,
        waveforms: Any,
        *,
        lead_order: Sequence[str],
        sampling_rate_hz: int,
        units: str,
    ) -> list[dict[str, Any]]:
        """Predict a batch shaped (B,12,1000); never infer input metadata."""
        _validate_metadata(
            lead_order=lead_order, sampling_rate_hz=sampling_rate_hz, units=units
        )
        array = _waveform_array(waveforms, batch=True)
        x = torch.from_numpy(array).to(self.device)
        with torch.inference_mode():
            normalized = self.normalizer.transform(x)
            if normalized.shape != x.shape or not torch.isfinite(normalized).all():
                raise ValueError("Normalization returned an invalid waveform")
            logits = self.model(normalized)
            if logits.shape != (array.shape[0], len(CLASS_ORDER)):
                raise RuntimeError("Model returned an unexpected output shape")
            probabilities = torch.sigmoid(logits)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Model returned non-finite probabilities")
            probabilities_np = probabilities.cpu().numpy()

        responses = []
        for row in probabilities_np:
            labels = [
                {
                    "class": label,
                    "probability": float(prob),
                    "threshold": float(threshold),
                    "predicted": bool(prob >= threshold),
                }
                for label, prob, threshold in zip(CLASS_ORDER, row, THRESHOLDS)
            ]
            positives = [item["class"] for item in labels if item["predicted"]]
            responses.append({
                "model": MODEL_NAME,
                "checkpoint_epoch": EXPECTED_EPOCH,
                "checkpoint_sha256": self.checkpoint_sha256,
                "normalization_sha256": self.normalization_sha256,
                "predictions": labels,
                "predicted_classes": positives,
                "status": (
                    "classifications_generated" if positives
                    else "no_class_above_threshold"
                ),
                "research_notice": RESEARCH_NOTICE,
            })
        return responses

    def predict(
        self,
        waveform: Any,
        *,
        lead_order: Sequence[str],
        sampling_rate_hz: int,
        units: str,
    ) -> dict[str, Any]:
        """Predict one ECG shaped (12,1000)."""
        array = _waveform_array(waveform, batch=False)
        return self.predict_batch(
            array[np.newaxis, ...],
            lead_order=lead_order,
            sampling_rate_hz=sampling_rate_hz,
            units=units,
        )[0]
