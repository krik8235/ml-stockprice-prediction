import pytest
import datetime
import unittest
from unittest import mock


def test_instatiate():
    from src.model.instantiate import MLP, AdaptiveEnsemble, LSTM, GRU
    import torch
    from torch.optim import swa_utils

    mlp = MLP(input_size=10)
    total_params = sum(p.numel() for p in mlp.parameters())
    assert total_params != 0

    adaptive = AdaptiveEnsemble(num_experts=5, input_size=10)
    total_params = sum(p.numel() for p in adaptive.parameters())
    assert total_params != 0

    lstm = LSTM(input_size=10)
    total_params = sum(p.numel() for p in lstm.parameters())
    assert total_params != 0

    gru = GRU(input_size=10)
    total_params = sum(p.numel() for p in gru.parameters())
    assert total_params != 0

    gru_swa = swa_utils.AveragedModel(gru)
    assert gru_swa is not None
