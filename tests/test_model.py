"""Tests for the LightningModule and the model itself."""

from collections.abc import Callable
from typing import Any

import lightning
import torch

from src.models.components.simple_dense_net import SimpleDenseNet
from src.models.mnist_module import MNISTLitModule


def test_forward_keeps_the_autograd_graph() -> None:
    """The net output must stay attached to the graph, otherwise backward() cannot run.

    Regression test: the net used to `return torch.tensor(self.model(x))`, which detaches the
    output and makes training fail with
    "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn".
    """
    net = SimpleDenseNet()
    out = net(torch.rand(2, 1, 28, 28))

    assert out.requires_grad
    assert out.grad_fn is not None
    out.sum().backward()  # must not raise


def test_forward_output_shape() -> None:
    """The net returns one logit vector per image."""
    net = SimpleDenseNet(output_size=10)
    assert net(torch.rand(3, 1, 28, 28)).shape == (3, 10)


def test_model_step_returns_batch_shaped_predictions(mnist_module_factory: Callable[[], MNISTLitModule]) -> None:
    """`model_step` returns a scalar loss and one prediction/target per sample.

    Regression test: the return type was annotated with `TensorType[()]` for the predictions and
    the targets, so typeguard raised "the return value[1] must be of type TensorType[()]" on the
    very first validation step.
    """
    module = mnist_module_factory()
    x, y = torch.rand(4, 1, 28, 28), torch.randint(0, 10, (4,))

    loss, preds, targets = module.model_step(x, y)

    assert loss.ndim == 0
    assert preds.shape == (4,)
    assert targets.shape == (4,)
    loss.backward()  # must not raise


def test_module_can_be_loaded_from_a_checkpoint(mnist_module_factory: Callable[[], MNISTLitModule]) -> None:
    """The module must be reconstructible from its own checkpoint (that is what serve.py does)."""
    module = mnist_module_factory()
    state: dict[str, Any] = {
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
        "pytorch-lightning_version": lightning.__version__,
    }

    recreated = MNISTLitModule(**state["hyper_parameters"])
    recreated.load_state_dict(state["state_dict"])
    recreated.eval()  # BatchNorm needs a batch of more than one sample in train mode

    assert isinstance(recreated.net, SimpleDenseNet)
    assert recreated(torch.rand(2, 1, 28, 28)).shape == (2, 10)
