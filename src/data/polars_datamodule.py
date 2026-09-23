"""PyTorch Lightning DataModule for loading dataset using Polars."""

from typing import Any

import polars as pl
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


# Custom PyTorch Dataset wrapping a Polars DataFrame
class PolarsDataset(Dataset):
    """Custom PyTorch Dataset wrapping a Polars DataFrame."""

    def __init__(self, df: pl.DataFrame, output_column: str) -> None:
        """Initialize the PolarsDataset."""
        self.df = df
        self.output_column = output_column

    def __len__(self) -> int:
        """Return the number of rows in the dataset."""
        return self.df.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the features and label for the given index."""
        row = self.df.row(idx, named=True)
        features = torch.tensor(
            [value for column, value in row.items() if column != self.output_column], dtype=torch.float32
        )
        label = torch.tensor(row[self.output_column], dtype=torch.long)
        return features, label


# PyTorch Lightning DataModule for loading dataset using Polars
class PolarsDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for loading dataset using Polars."""

    def __init__(
        self,
        data_path: str,
        output_column: str,
        batch_size: int = 32,
        num_workers: int = 0,
        test_size: float = 0.2,
    ) -> None:
        """Initialize the PolarsDataModule.

        Args:
            data_path: Path to the dataset.
            output_column: Column name that contains the labels.
            batch_size: Batch size for the dataloaders.
            num_workers: Number of workers for the dataloaders.
            test_size: Fraction of the dataset to be used for validation.
        """
        super().__init__()
        self.data_path = data_path
        self.output_column = output_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load and split the dataset into train and validation sets."""
        # Load dataset using Polars
        df = pl.read_csv(self.data_path)

        # Shuffle deterministically, then split the dataframe in two contiguous parts
        df = df.sample(fraction=1.0, shuffle=True, seed=42)
        n_val = int(len(df) * self.test_size)

        self.train_dataset = PolarsDataset(df.head(len(df) - n_val), output_column=self.output_column)
        self.val_dataset = PolarsDataset(df.tail(n_val), output_column=self.output_column)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        assert self.train_dataset is not None, "Call setup() before requesting a dataloader."
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        assert self.val_dataset is not None, "Call setup() before requesting a dataloader."
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    # Path to dataset
    data_path = "path_to_your_dataset.csv"
    output_column = "label"  # Column name that contains the labels

    # Instantiate and use the DataModule
    datamodule = PolarsDataModule(data_path, output_column, batch_size=32)
    datamodule.setup()

    for batch in datamodule.train_dataloader():
        features, labels = batch
        print(f"Features: {features}, Labels: {labels}")
