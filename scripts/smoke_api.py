"""Run a research-only in-process API smoke test with frozen artifacts.

Example on Colab with Drive mounted:
python scripts/smoke_api.py --artifacts-dir '/content/drive/MyDrive/ECG DL Capstone Project/artifacts'

Uses a *synthetic* all-zero waveform to validate request routing only. The
output probabilities are NOT meaningful and must never be reported as results.
No test/validation patient recordings are loaded or evaluated by this script.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from src.ecg.api.main import create_app
from src.ecg.inference.predictor import LEAD_ORDER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    app = create_app(artifacts_dir=args.artifacts_dir, device=args.device)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        response = client.post("/v1/predict", json={
            "waveform": [[0.0] * 1000 for _ in range(12)],
            "lead_order": list(LEAD_ORDER),
            "sampling_rate_hz": 100,
            "units": "mV",
        })
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["model"] == "cnn1d_baseline_v1"
        assert len(payload["predictions"]) == 5
        assert [p["class"] for p in payload["predictions"]] == [
            "NORM", "MI", "STTC", "CD", "HYP"
        ]
        assert "not for clinical" in payload["research_notice"]
    print("Frozen artifact integrity: PASS")
    print("API health endpoint: PASS")
    print("Synthetic ECG POST /v1/predict: PASS")
    print("Thresholds and output classes: PASS")
    print("No validation or test recordings used: YES")
    print("MODEL RETRAINING: NO")
    print("FASTAPI INTEGRATION SMOKE TEST: PASSED")


if __name__ == "__main__":
    main()
