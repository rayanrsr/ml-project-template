"""Tests for the LitServe API."""

from collections.abc import Callable
from pathlib import Path

import lightning
import torch
from lightning import LightningModule

from src.serve_apis.mnist_serve import MNISTServeAPI


def test_decoded_request_and_batched_predict(
    tmp_path: Path, mnist_module_factory: Callable[[], LightningModule]
) -> None:
    """A request must decode to [1, 28, 28] and predict must accept the batch LitServe builds.

    Regression tests: `decode_request` used to feed a `torch.Tensor` to
    `torchvision.transforms.ToTensor()` ("TypeError: pic should be PIL Image or ndarray"), and
    `predict` added a second batch dimension to the already batched input, producing a
    `[1, 1, 1, 28, 28]` tensor that the model rejects.
    """
    checkpoint_path = tmp_path / "model.ckpt"
    module = mnist_module_factory()
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        checkpoint_path,
    )

    api = MNISTServeAPI(model_class=type(module), checkpoint_path=str(checkpoint_path))
    api.setup("cpu")

    decoded = api.decode_request({"image": [[0.0] * 28 for _ in range(28)]})
    assert decoded.shape == (1, 28, 28)

    # this is what LitServe does with the decoded samples before calling predict
    batched = decoded.unsqueeze(0)
    output = api.predict(batched)

    assert len(output["prediction"]) == 1
    assert 0 <= output["prediction"][0] <= 9
    assert api.encode_response(output)["predicted_class"] == output["prediction"]
