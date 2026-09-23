"""Fixtures for your unit tests."""

from collections.abc import Callable
from functools import partial

import pytest
import torch

from src.models.components.simple_dense_net import SimpleDenseNet
from src.models.mnist_module import MNISTLitModule


@pytest.fixture
def mnist_module_factory() -> Callable[[], MNISTLitModule]:
    """Return a factory building a small `MNISTLitModule` (no trainer, no data needed)."""

    def build() -> MNISTLitModule:
        return MNISTLitModule(
            net=SimpleDenseNet(input_size=784, lin1_size=8, lin2_size=8, lin3_size=8, output_size=10),
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            scheduler=None,
        )

    return build
