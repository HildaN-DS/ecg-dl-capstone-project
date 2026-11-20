"""Exercise the locally running research API using synthetic ECG data only.

Run on host after `docker compose -f compose.api.yaml up --build -d`:
python scripts/smoke_docker_api.py
"""
import argparse
import httpx

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    with httpx.Client(timeout=30.0, follow_redirects=False) as client:
        health = client.get(url + "/health")
        assert health.status_code == 200, health.text
        assert health.json()["status"] == "ready"
        payload = {
            "waveform": [[0.0] * 1000 for _ in range(12)],
            "lead_order": LEADS,
            "sampling_rate_hz": 100,
            "units": "mV",
        }
        result = client.post(url + "/v1/predict", json=payload)
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["model"] == "cnn1d_baseline_v1"
        assert [p["class"] for p in body["predictions"]] == ["NORM", "MI", "STTC", "CD", "HYP"]
        assert all(p["threshold"] == 0.5 for p in body["predictions"])
        assert "not for clinical" in body["research_notice"]
        assert all(0 <= p["probability"] <= 1 for p in body["predictions"])
        bad = dict(payload, sampling_rate_hz=500)
        rejected = client.post(url + "/v1/predict", json=bad)
        assert rejected.status_code == 422, rejected.text
    print("Health endpoint: PASS")
    print("Synthetic ECG inference: PASS")
    print("Five classes / fixed thresholds: PASS")
    print("Invalid sample rate HTTP 422: PASS")
    print("Patient ECG data used: NO")
    print("LOCAL DOCKER API SMOKE TEST: PASSED")


if __name__ == "__main__":
    main()
