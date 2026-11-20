"""Compact bidirectional LSTM for PTB-XL five-label ECG classification.

Input:  float32 ECG waveforms with shape (batch, 12 leads, time samples).
Output: raw logits with shape (batch, num_classes), suitable for
        torch.nn.BCEWithLogitsLoss (no sigmoid inside the model).

Waveforms must be normalized externally using the fitted training-only
ECGStandardizer. The model itself does not fit preprocessing statistics.
"""

import torch
from torch import nn


class ECGLSTM(nn.Module):
    """Two-layer bidirectional LSTM with a small multi-label classifier head."""

    def __init__(
        self,
        num_classes: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if num_classes < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("num_classes, hidden_size and num_layers must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.lstm = nn.LSTM(
            input_size=12,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw five-class logits for an ECG batch."""
        if x.ndim != 3 or x.shape[1] != 12 or x.shape[2] < 1:
            raise ValueError(
                "Expected ECG input with shape (batch, 12, time >= 1); "
                f"received {tuple(x.shape)}."
            )
        if not torch.is_floating_point(x):
            raise TypeError("ECG input must be a floating-point tensor.")

        # PTB-XL cache is channels-first; nn.LSTM expects sequence-first
        # within each batch when batch_first=True.
        sequence = x.transpose(1, 2).contiguous()  # (B, T, 12)
        _, (h_n, _) = self.lstm(sequence)

        # h_n contains [layer0 forward, layer0 backward, ...,
        # final-layer forward, final-layer backward]. These two states
        # summarize the entire sequence in each temporal direction.
        final_state = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.classifier(self.dropout(final_state))
