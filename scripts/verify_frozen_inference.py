"""Check FrozenCNNPredictor against saved validation predictions, not test data.

Example Colab:
python -u scripts/verify_frozen_inference.py \
  --root '/content/drive/MyDrive/ECG DL Capstone Project' \
  --device cuda

Loads the verified cache (potentially all shards into CPU memory) but only
passes validation fold 9 through inference. Does not evaluate fold 10.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ecg.data.cached_dataset import CachedECGStore
from src.ecg.inference.predictor import FrozenCNNPredictor, LEAD_ORDER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--local-cache-dir", type=Path,
        default=Path("/content/ptbxl_cached_shards"),
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this runtime")

    print("FROZEN CNN INFERENCE INTEGRATION AUDIT", flush=True)
    print("Device:", args.device, flush=True)
    predictor = FrozenCNNPredictor.from_artifacts(
        args.root / "artifacts", device=args.device
    )
    with np.load(
        args.root / "artifacts/cnn1d_baseline_v1/best_val_predictions.npz",
        allow_pickle=False,
    ) as saved:
        saved_ids = saved["ecg_ids"].copy()
        saved_true = saved["y_true"].copy()
        saved_prob = saved["y_prob"].copy()

    manifest_path = args.root / "data/processed/ptbxl_labels.csv"
    store = CachedECGStore(
        manifest_path=manifest_path,
        cache_dir=args.root / "data/processed/ptbxl_waveform_cache_v1",
        local_dir=args.local_cache_dir,
    )
    val_subset = store.split("val")
    assert len(val_subset) == 2146
    manifest = pd.read_csv(manifest_path)
    np.testing.assert_array_equal(
        manifest.iloc[val_subset.indices]["ecg_id"].to_numpy(), saved_ids
    )
    loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0)
    observed_prob = []
    observed_true = []
    for x, y in loader:
        responses = predictor.predict_batch(
            x.numpy(), lead_order=LEAD_ORDER,
            sampling_rate_hz=100, units="mV",
        )
        observed_prob.append(np.asarray(
            [[item["probability"] for item in res["predictions"]] for res in responses],
            dtype=np.float64,
        ))
        observed_true.append(y.numpy())

    got_prob = np.concatenate(observed_prob)
    got_true = np.concatenate(observed_true)
    np.testing.assert_array_equal(got_true, saved_true)
    max_diff = float(np.max(np.abs(got_prob - saved_prob)))
    print("Validation records:", len(got_true))
    print("Maximum probability difference:", f"{max_diff:.10f}")
    np.testing.assert_allclose(got_prob, saved_prob, atol=1e-4, rtol=1e-4)
    print("Saved validation predictions reproduced: PASS")
    print("Frozen SHA-256 checks: PASS")
    print("No checkpoint or threshold changes: YES")
    print("Test fold evaluated: NO")
    print("FROZEN CNN INFERENCE INTEGRATION AUDIT: PASSED")


if __name__ == "__main__":
    main()
