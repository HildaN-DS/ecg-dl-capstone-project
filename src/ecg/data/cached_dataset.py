
"""RAM-backed PyTorch datasets from verified PTB-XL shards."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, TensorDataset


CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")

SPLIT_FOLDS = {
    "train": list(range(1, 9)),
    "val": [9],
    "test": [10],
}


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


class CachedECGStore:
    """Load verified waveform shards into shared CPU memory."""

    def __init__(
        self,
        manifest_path: str | Path,
        cache_dir: str | Path,
        local_dir: str | Path,
    ):
        cache_dir = Path(cache_dir)
        local_dir = Path(local_dir)

        if not (cache_dir / "READY.json").is_file():
            raise ValueError("Waveform cache is not marked ready.")

        manifest = pd.read_csv(manifest_path)

        required = {
            "ecg_id", "patient_id", "strat_fold",
            *CLASSES,
        }

        if not required.issubset(manifest.columns):
            raise ValueError("Manifest is missing required columns.")

        if manifest["ecg_id"].duplicated().any():
            raise ValueError("Duplicate ECG IDs in manifest.")

        if not manifest["strat_fold"].between(1, 10).all():
            raise ValueError("Invalid fold assignments.")

        n = len(manifest)

        # Allocate the final arrays only once.
        signals = np.empty(
            (n, 12, 1000), dtype=np.float32
        )

        targets = np.empty(
            (n, 5), dtype=np.float32
        )

        shard_paths = sorted(
            cache_dir.glob("shard_*.npz")
        )

        if not shard_paths:
            raise ValueError("No waveform shards found.")

        local_dir.mkdir(parents=True, exist_ok=True)

        offset = 0

        for i, source in enumerate(shard_paths):
            expected_name = f"shard_{i:04d}.npz"

            if source.name != expected_name:
                raise ValueError(
                    f"Missing or unordered shard: {expected_name}"
                )

            marker_path = source.with_suffix(".json")
            marker = json.loads(marker_path.read_text())

            start = int(marker["start"])
            stop = int(marker["stop"])

            if start != offset or stop > n or stop <= start:
                raise ValueError("Invalid shard boundaries.")

            destination = local_dir / source.name

            # Reuse a local shard only when its hash matches.
            if (
                not destination.is_file()
                or sha256_file(destination)
                != marker["file_sha256"]
            ):
                shutil.copy2(source, destination)

            if (
                sha256_file(destination)
                != marker["file_sha256"]
            ):
                raise ValueError(
                    f"Checksum mismatch: {source.name}"
                )

            expected = manifest.iloc[start:stop]
            count = stop - start

            with np.load(
                destination, allow_pickle=False
            ) as shard:

                x = shard["signals"]
                y = shard["targets"]

                if x.shape != (count, 12, 1000):
                    raise ValueError("Invalid waveform shape.")

                if y.shape != (count, 5):
                    raise ValueError("Invalid target shape.")

                if x.dtype != np.float32:
                    raise ValueError("Waveforms must be float32.")

                if y.dtype != np.float32:
                    raise ValueError("Targets must be float32.")

                if not np.isfinite(x).all():
                    raise ValueError("Non-finite waveform values.")

                checks = {
                    "ecg_ids": "ecg_id",
                    "patient_ids": "patient_id",
                    "folds": "strat_fold",
                }

                for array_name, column in checks.items():
                    if not np.array_equal(
                        shard[array_name],
                        expected[column].to_numpy(),
                    ):
                        raise ValueError(
                            f"{array_name} alignment failed."
                        )

                if not np.array_equal(
                    y,
                    expected[list(CLASSES)].to_numpy(
                        dtype=np.float32
                    ),
                ):
                    raise ValueError("Target alignment failed.")

                signals[start:stop] = x
                targets[start:stop] = y

            offset = stop

            if (i + 1) % 20 == 0 or i + 1 == len(shard_paths):
                print(
                    f"Loaded {i + 1}/{len(shard_paths)} "
                    f"shards; {offset} ECGs",
                    flush=True,
                )

        if offset != n:
            raise ValueError(
                f"Incomplete cache: {offset}/{n} ECGs."
            )

        # Both tensors share the allocated NumPy memory.
        self.dataset = TensorDataset(
            torch.from_numpy(signals),
            torch.from_numpy(targets),
        )

        self.folds = manifest[
            "strat_fold"
        ].to_numpy(dtype=np.int64)

        self.ecg_ids = manifest[
            "ecg_id"
        ].to_numpy(dtype=np.int64)

    def split(self, name: str) -> Subset:
        """Return a partition without duplicating waveforms."""
        if name not in SPLIT_FOLDS:
            raise ValueError(f"Unknown split: {name}")

        indices = np.flatnonzero(
            np.isin(
                self.folds,
                SPLIT_FOLDS[name],
            )
        )

        return Subset(
            self.dataset,
            indices.tolist(),
        )
