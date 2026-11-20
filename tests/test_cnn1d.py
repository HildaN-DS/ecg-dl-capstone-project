
"""Tests for the baseline PTB-XL 1D CNN."""

import pytest
import torch
from torch import nn

from src.ecg.models.cnn1d import ECG1DCNN


def test_forward_output_shape():
    torch.manual_seed(42)

    model = ECG1DCNN()
    x = torch.randn(4, 12, 1000)

    logits = model(x)

    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()


def test_backward_pass():
    torch.manual_seed(42)

    model = ECG1DCNN()
    x = torch.randn(4, 12, 1000)

    targets = torch.randint(
        0, 2, (4, 5)
    ).float()

    logits = model(x)

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, targets)

    assert torch.isfinite(loss)

    loss.backward()

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert all(
        parameter.grad is not None
        for parameter in trainable
    )

    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in trainable
    )


def test_probability_conversion():
    model = ECG1DCNN()
    model.eval()

    x = torch.randn(2, 12, 1000)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.sigmoid(logits)

    assert probabilities.shape == (2, 5)
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)


def test_invalid_input_shape():
    model = ECG1DCNN()

    with pytest.raises(ValueError):
        model(torch.randn(2, 1000, 12))


def test_invalid_configuration():
    with pytest.raises(ValueError):
        ECG1DCNN(num_classes=0)

    with pytest.raises(ValueError):
        ECG1DCNN(dropout=1.0)
