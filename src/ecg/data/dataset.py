
"""PyTorch Dataset for PTB-XL 100 Hz ECG recordings."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset


CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")

EXPECTED_LEADS = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

SPLIT_FOLDS = {
    "train": set(range(1, 9)),
    "val": {9},
    "test": {10},
}


class ECGDataset(Dataset):
    """Lazily load 12-lead ECGs and five-class targets."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_dir: str | Path,
        split: str,
    ):
        if split not in SPLIT_FOLDS:
            raise ValueError(
                f"Invalid split: {split}. "
                "Use 'train', 'val', or 'test'."
            )

        self.data_dir = Path(data_dir)
        manifest = pd.read_csv(manifest_path)

        required = {
            "ecg_id",
            "patient_id",
            "strat_fold",
            "filename_lr",
            *CLASSES,
        }

        missing = required - set(manifest.columns)

        if missing:
            raise ValueError(
                f"Missing manifest columns: {sorted(missing)}"
            )

        if manifest["ecg_id"].duplicated().any():
            raise ValueError("Duplicate ECG IDs in manifest.")

        if not manifest[list(CLASSES)].isin([0, 1]).all().all():
            raise ValueError("Targets must be binary.")

        if manifest[list(CLASSES)].sum(axis=1).lt(1).any():
            raise ValueError(
                "Found recordings without diagnostic targets."
            )

        self.records = (
            manifest.loc[
                manifest["strat_fold"].isin(
                    SPLIT_FOLDS[split]
                )
            ]
            .reset_index(drop=True)
        )

        if self.records.empty:
            raise ValueError(f"No recordings for split: {split}")

        self.split = split

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]

        record_path = (
            self.data_dir / row["filename_lr"]
        )

        signal, fields = wfdb.rdsamp(
            str(record_path)
        )

        if signal.shape != (1000, 12):
            raise ValueError(
                f"ECG {row['ecg_id']}: "
                f"Unexpected shape {signal.shape}"
            )

        if fields["fs"] != 100:
            raise ValueError(
                f"ECG {row['ecg_id']}: "
                "Unexpected sampling frequency."
            )

        actual_leads = [
            lead.casefold()
            for lead in fields["sig_name"]
        ]

        expected_leads = [
            lead.casefold()
            for lead in EXPECTED_LEADS
        ]

        if actual_leads != expected_leads:
            raise ValueError(
                f"ECG {row['ecg_id']}: "
                "Unexpected lead names or order."
            )

        if not np.isfinite(signal).all():
            raise ValueError(
                f"ECG {row['ecg_id']}: "
                "Non-finite waveform values."
            )

        # WFDB: (time, leads)
        # PyTorch Conv1d: (channels, time)
        x = torch.from_numpy(
            np.ascontiguousarray(
                signal.T,
                dtype=np.float32,
            )
        )

        y = torch.tensor(
            row[list(CLASSES)].to_numpy(
                dtype=np.float32
            )
        )

        return x, y
