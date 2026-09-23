"""Tests for the MNIST datamodule split."""

from collections.abc import Sized
from typing import Any, cast

import pytest
import torch
from torch.utils.data import Dataset

from src.data import mnist_datamodule
from src.data.mnist_datamodule import MNISTDataModule


class FakeMNIST(Dataset):
    """Stand-in for torchvision's MNIST: no download, stable lengths."""

    def __init__(self, root: str, train: bool = True, transform: Any = None, download: bool = False) -> None:
        """Mirror the torchvision MNIST constructor signature."""
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download

    def __len__(self) -> int:
        """Return the length of the real MNIST splits."""
        return 60_000 if self.train else 10_000

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return a dummy (image, label) pair."""
        return torch.zeros(1, 28, 28), 0


def test_setup_keeps_the_official_test_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Train/val are carved out of the training split; the test split is left untouched.

    Regression test: train and test used to be concatenated and re-split as a whole, so 8.6k of
    the 10k "test" images were actually training images and the reported test accuracy was
    meaningless.
    """
    monkeypatch.setattr(mnist_datamodule, "MNIST", FakeMNIST)

    datamodule = MNISTDataModule(data_dir="data/")
    datamodule.setup()

    assert datamodule.data_train is not None
    assert datamodule.data_val is not None
    assert datamodule.data_test is not None
    assert len(cast(Sized, datamodule.data_train)) == 55_000
    assert len(cast(Sized, datamodule.data_val)) == 5_000
    assert isinstance(datamodule.data_test, FakeMNIST)
    assert datamodule.data_test.train is False
    assert len(datamodule.data_test) == 10_000
