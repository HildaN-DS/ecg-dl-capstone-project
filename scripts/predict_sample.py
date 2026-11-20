"""Validate a bundled synthetic 12-lead ECG and submit it to the local frozen CNN API.

From the repository root (no frontend required):
  python scripts/predict_sample.py --validate-only
  python scripts/predict_sample.py
Requires only Python 3.10+ standard library. Not for clinical use.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
DEFAULT_SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "sample_ecg_synthetic.csv"
MAX_FLOAT32 = 3.4028234663852886e38


def read_sample(path: Path) -> dict:
    """Parse exactly one 10-second, 100 Hz, millivolt ECG (12 x 1,000)."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        content = file.read()

    lines = content.splitlines()
    metadata = {}
    data_lines = []
    for line in lines:
        if line.startswith("#"):
            if data_lines:
                raise ValueError("Metadata must precede CSV columns")
            if "=" not in line:
                raise ValueError("Invalid CSV metadata")
            key, value = line[1:].strip().split("=", 1)
            key, value = key.strip(), value.strip()
            if key in metadata:
                raise ValueError(f"Duplicate metadata: {key}")
            metadata[key] = value
        elif line.strip():
            data_lines.append(line)

    if metadata != {"sampling_rate_hz": "100", "units": "mV", "synthetic": "true"}:
        raise ValueError("Expected sample metadata: sampling_rate_hz=100, units=mV, synthetic=true")

    reader = csv.reader(io.StringIO("\n".join(data_lines)))
    header = next(reader, None)
    if header != list(LEADS):
        raise ValueError(f"Lead order must be: {', '.join(LEADS)}")

    waveforms = [[] for _ in LEADS]
    count = 0
    for count, row in enumerate(reader, 1):
        if count > 1000:
            raise ValueError("Expected exactly 1000 samples per lead (too many rows)")
        if len(row) != len(LEADS):
            raise ValueError(f"Sample row {count}: expected 12 values; got {len(row)}")
        for i, token in enumerate(row):
            try:
                value = float(token)
            except ValueError as e:
                raise ValueError(f"Sample {count}, lead {LEADS[i]} is not numeric") from e
            if not math.isfinite(value) or abs(value) > MAX_FLOAT32:
                raise ValueError(f"Sample {count}, lead {LEADS[i]} is not finite float32-compatible")
            waveforms[i].append(value)
    if count != 1000:
        raise ValueError(f"Expected exactly 1000 samples per lead; got {count}")

    return {
        "waveform": waveforms,
        "lead_order": list(LEADS),
        "sampling_rate_hz": 100,
        "units": "mV",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    try:
        payload = read_sample(args.sample)
        print(f"Synthetic sample verified: {args.sample.name} | 12 leads x 1000 samples | 100 Hz | mV")
        if args.validate_only:
            print("SAMPLE STRUCTURE: PASS (no API request sent)")
            return 0

        request = Request(
            args.url.rstrip("/") + "/v1/predict",
            data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            body = json.load(response)
    except HTTPError as e:
        print(f"API returned HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError, UnicodeError, URLError, TimeoutError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    predictions = body.get("predictions")
    if not isinstance(predictions, list) or [p.get("class") for p in predictions] != list(CLASSES):
        print("ERROR: unexpected prediction response (missing or reordered classes)", file=sys.stderr)
        return 1

    print(f"Model: {body.get('model')} | checkpoint epoch: {body.get('checkpoint_epoch')}")
    print("Class     Score       Threshold   Selected")
    for item in predictions:
        print(f"{item['class']:<8}  {float(item['probability']):>7.2%}     {float(item['threshold']):>7.2%}      {bool(item['predicted'])}")
    print("Above threshold:", ", ".join(body.get("predicted_classes", [])) or "none")
    print("Research-only synthetic test. Predictions have no clinical interpretation.")
    print("SAMPLE API INFERENCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
