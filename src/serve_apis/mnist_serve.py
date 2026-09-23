"""This is an example of a LitServe api for the Mnist LightningModule."""

from typing import cast

import lightning
import litserve as ls
import torch
from torchvision.transforms import transforms


# Define the LitServe API for serving the MNIST model
class MNISTServeAPI(ls.LitAPI):
    """LitServe API for serving the MNIST model."""

    def __init__(self, model_class: type[lightning.pytorch.LightningModule], checkpoint_path: str) -> None:
        """Initialize the MNISTServeAPI.

        Args:
            model_class: The LightningModule class to serve.
            checkpoint_path: The path to the model checkpoint.
        """
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.model_class = model_class

    def setup(self, device: str | torch.device) -> None:
        """Setup is called once at startup.

        Load the model, set the device, and prepare any other necessary components.
        """
        # Load the trained MNIST model (ensure model weights are loaded properly here)
        self.device = device
        self.model = self.model_class.load_from_checkpoint(self.checkpoint_path)
        self.model.to(device)  # Move the model to the appropriate device (CPU or GPU)
        self.model.eval()  # Set the model to evaluation mode

        # Define the transforms that match the training data processing pipeline.
        # The datamodule applies ToTensor() + Normalize(), so the input is expected as a
        # `[1, 28, 28]` float tensor scaled to [0, 1]: only the normalization is left to do here.
        self.transforms = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])

    def decode_request(self, request: dict) -> torch.Tensor:
        """Decode the incoming request and prepare the input for the model.

        The request payload is expected to be a `[28, 28]` nested list of floats in [0, 1]
        (e.g. a `PIL.Image` converted with `numpy.asarray(img) / 255`).
        """
        image_data = request["image"]
        # `[28, 28]` -> `[1, 28, 28]` (channels first, no batch dimension: LitServe batches the
        # decoded samples itself, so adding a batch dimension here would break the shapes)
        image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)
        return cast(torch.Tensor, self.transforms(image_tensor))

    def predict(self, x: torch.Tensor) -> dict[str, list[int]]:
        """Run inference using the MNIST model and return the prediction.

        `x` is already batched (`[batch, 1, 28, 28]`) by LitServe.
        """
        with torch.no_grad():
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1)  # Get the predicted classes
        return {"prediction": preds.tolist()}  # One class id per sample of the batch

    def encode_response(self, output: dict) -> dict[str, list[int]]:
        """Encode the model's output into a response payload."""
        return {"predicted_class": output["prediction"]}
