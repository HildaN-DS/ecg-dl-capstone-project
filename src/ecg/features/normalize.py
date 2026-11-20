
"""Training-only standardization for PTB-XL ECG waveforms."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


LEADS = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

TRAIN_FOLDS = tuple(range(1, 9))


class ECGStandardizer:
    """Apply global per-lead training-derived standardization."""

    METHOD = "global_per_lead_zscore"

    def __init__(
        self,
        mean,
        std,
        training_ecgs,
        samples_per_lead,
    ):
        self.mean = torch.as_tensor(
            mean, dtype=torch.float64
        ).clone()

        self.std = torch.as_tensor(
            std, dtype=torch.float64
        ).clone()

        self.training_ecgs = int(training_ecgs)
        self.samples_per_lead = int(samples_per_lead)

        if self.mean.shape != (12,):
            raise ValueError("Expected 12 lead means.")

        if self.std.shape != (12,):
            raise ValueError("Expected 12 lead standard deviations.")

        if not torch.isfinite(self.mean).all():
            raise ValueError("Non-finite means.")

        if not torch.isfinite(self.std).all():
            raise ValueError("Non-finite standard deviations.")

        if not (self.std > 0).all():
            raise ValueError("Standard deviations must be positive.")

        if self.training_ecgs <= 0 or self.samples_per_lead <= 0:
            raise ValueError("Invalid training sample counts.")

    @classmethod
    def fit(cls, store, batch_size=128):
        """Fit exclusively from the store's training partition."""

        train = store.split("train")

        # Verify that only folds 1-8 contribute to fitting.
        indices = np.asarray(train.indices)
        folds = store.folds[indices]

        if not np.isin(folds, TRAIN_FOLDS).all():
            raise ValueError(
                "Non-training fold included during fitting."
            )

        if len(train) == 0:
            raise ValueError("Training partition is empty.")

        loader = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        lead_sum = torch.zeros(
            12, dtype=torch.float64
        )

        count = 0

        for x, _ in loader:
            x = x.to(torch.float64)

            if x.ndim != 3 or x.shape[1:] != (12, 1000):
                raise ValueError("Unexpected ECG dimensions.")

            lead_sum += x.sum(dim=(0, 2))
            count += x.shape[0] * x.shape[2]

        mean = lead_sum / count

        squared_sum = torch.zeros(
            12, dtype=torch.float64
        )

        for x, _ in loader:
            x = x.to(torch.float64)

            squared_sum += (
                (x - mean.view(1, 12, 1)) ** 2
            ).sum(dim=(0, 2))

        std = torch.sqrt(squared_sum / count)

        return cls(
            mean=mean,
            std=std,
            training_ecgs=len(train),
            samples_per_lead=count,
        )

    def _parameters(self, x):
        """Broadcast parameters to an ECG or a batch."""

        if x.ndim == 2 and x.shape[0] == 12:
            shape = (12, 1)

        elif x.ndim == 3 and x.shape[1] == 12:
            shape = (1, 12, 1)

        else:
            raise ValueError(
                "Expected (12, time) or (batch, 12, time)."
            )

        mean = self.mean.to(
            device=x.device, dtype=x.dtype
        ).reshape(shape)

        std = self.std.to(
            device=x.device, dtype=x.dtype
        ).reshape(shape)

        return mean, std

    def transform(self, x):
        """Standardize without modifying the input tensor."""

        if not x.is_floating_point():
            raise TypeError("ECG input must be floating-point.")

        mean, std = self._parameters(x)
        return (x - mean) / std

    def inverse_transform(self, x):
        """Reconstruct original amplitude values."""

        mean, std = self._parameters(x)
        return x * std + mean

    def save(self, path):
        """Save parameters and training provenance as JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "method": self.METHOD,
            "lead_order": list(LEADS),
            "training_folds": list(TRAIN_FOLDS),
            "training_ecgs": self.training_ecgs,
            "samples_per_lead": self.samples_per_lead,
            "mean_mV": self.mean.tolist(),
            "std_mV": self.std.tolist(),
        }

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path):
        """Load and validate fitted parameters."""

        with Path(path).open(
            "r", encoding="utf-8"
        ) as file:
            data = json.load(file)

        if data["method"] != cls.METHOD:
            raise ValueError("Unexpected normalization method.")

        if data["lead_order"] != list(LEADS):
            raise ValueError("Lead order mismatch.")

        if data["training_folds"] != list(TRAIN_FOLDS):
            raise ValueError("Training fold mismatch.")

        return cls(
            mean=data["mean_mV"],
            std=data["std_mV"],
            training_ecgs=data["training_ecgs"],
            samples_per_lead=data["samples_per_lead"],
        )
