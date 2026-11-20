"""CPU-friendly unit tests for the PTB-XL bidirectional LSTM."""

import pytest
import torch
from torch import nn

from src.ecg.models.lstm import ECGLSTM


def test_forward_shape_for_ptbxl_waveforms():
    model = ECGLSTM(num_classes=5).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 12, 1000))
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()


def test_variable_batch_and_length():
    model = ECGLSTM(num_classes=5).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 12, 32)).shape == (1, 5)
        assert model(torch.randn(3, 12, 64)).shape == (3, 5)


def test_rejects_invalid_input_and_parameters():
    model = ECGLSTM()
    with pytest.raises(ValueError, match="Expected ECG input"):
        model(torch.randn(2, 1000, 12))
    with pytest.raises(ValueError, match="Expected ECG input"):
        model(torch.randn(2, 12))
    with pytest.raises(ValueError, match="Expected ECG input"):
        model(torch.empty(2, 12, 0))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.ones(2, 12, 32, dtype=torch.long))
    with pytest.raises(ValueError, match="dropout"):
        ECGLSTM(dropout=1.0)


def test_bidirectional_hidden_state_summary_matches_forward():
    model = ECGLSTM(dropout=0.0).eval()
    x = torch.randn(2, 12, 48)
    with torch.no_grad():
        _, (h_n, _) = model.lstm(x.transpose(1, 2).contiguous())
        expected = model.classifier(torch.cat((h_n[-2], h_n[-1]), dim=1))
        actual = model(x)
    torch.testing.assert_close(actual, expected)


def test_backward_pass_has_finite_gradients():
    model = ECGLSTM()
    x = torch.randn(2, 12, 48)
    y = torch.randint(0, 2, (2, 5)).float()
    logits = model(x)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )


def test_optimizer_step_changes_parameters():
    model = ECGLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(2, 12, 48)
    y = torch.randint(0, 2, (2, 5)).float()
    before = model.classifier.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = nn.BCEWithLogitsLoss()(model(x), y)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight.detach())
