"""Synthetic-only API tests: do not require PTB-XL artifacts or patient data."""

import pytest
import torch
from fastapi.testclient import TestClient

from src.ecg.api.main import create_app
from src.ecg.inference.predictor import FrozenCNNPredictor, LEAD_ORDER


class IdentityNormalizer:
    def transform(self, x):
        return x


class ConstantModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("fixed", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.fixed.unsqueeze(0).expand(len(x), -1)


def fake_predictor(logits):
    return FrozenCNNPredictor(
        model=ConstantModel(logits),
        normalizer=IdentityNormalizer(),
        device="cpu",
        checkpoint_sha256="synthetic-checkpoint",
        normalization_sha256="synthetic-normalization",
    )


def request_payload():
    return {
        "waveform": [[0.0] * 1000 for _ in range(12)],
        "lead_order": list(LEAD_ORDER),
        "sampling_rate_hz": 100,
        "units": "mV",
    }


@pytest.fixture
def client():
    with TestClient(create_app(predictor=fake_predictor([2, 0, -1, -2, -3]))) as test_client:
        yield test_client


def test_health_ready_with_research_notice(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["model"] == "cnn1d_baseline_v1"
    assert "not for clinical" in response.json()["research_notice"]


def test_predict_schema_and_frozen_decisions(client):
    response = client.post("/v1/predict", json=request_payload())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["predicted_classes"] == ["NORM", "MI"]
    assert [p["class"] for p in result["predictions"]] == [
        "NORM", "MI", "STTC", "CD", "HYP"
    ]
    assert all(p["threshold"] == 0.5 for p in result["predictions"])
    assert result["checkpoint_sha256"] == "synthetic-checkpoint"
    assert result["checkpoint_epoch"] == 17
    assert result["status"] == "classifications_generated"


def test_no_predicted_class_is_not_normal():
    with TestClient(create_app(predictor=fake_predictor([-3.0] * 5))) as client:
        result = client.post("/v1/predict", json=request_payload()).json()
    assert result["predicted_classes"] == []
    assert result["status"] == "no_class_above_threshold"
    assert not result["predictions"][0]["predicted"]


@pytest.mark.parametrize("shape", ["missing-lead", "wrong-samples", "transposed"])
def test_reject_invalid_waveform_shape(client, shape):
    payload = request_payload()
    if shape == "missing-lead":
        payload["waveform"] = payload["waveform"][:-1]
    elif shape == "wrong-samples":
        payload["waveform"][0] = payload["waveform"][0][:-1]
    else:
        payload["waveform"] = [[0.0] * 12 for _ in range(1000)]
    result = client.post("/v1/predict", json=payload)
    assert result.status_code == 422


@pytest.mark.parametrize("bad_sample", [True, "1.0", None, 1e39])
def test_reject_invalid_numeric_values(client, bad_sample):
    payload = request_payload()
    payload["waveform"][0][0] = bad_sample
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422


def test_reject_wrong_lead_order(client):
    payload = request_payload()
    payload["lead_order"][0], payload["lead_order"][1] = (
        payload["lead_order"][1], payload["lead_order"][0]
    )
    assert client.post("/v1/predict", json=payload).status_code == 422


@pytest.mark.parametrize("key,value", [("sampling_rate_hz", 500), ("units", "uV")])
def test_reject_wrong_metadata(client, key, value):
    payload = request_payload()
    payload[key] = value
    assert client.post("/v1/predict", json=payload).status_code == 422


def test_reject_missing_required_metadata(client):
    payload = request_payload()
    del payload["lead_order"]
    assert client.post("/v1/predict", json=payload).status_code == 422


def test_api_routes_are_intentionally_minimal(client):
    assert client.get("/v1/predict").status_code == 405
    assert client.get("/v1/unknown").status_code == 404
