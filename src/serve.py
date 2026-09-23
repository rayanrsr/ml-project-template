"""Main serve script."""

import hydra
import lightning
import litserve as ls
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    extras,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def serve(cfg: DictConfig) -> None:
    """Serve the specified model in the configuration as a FastAPI api.

    Args:
        cfg: A DictConfig configuration composed by Hydra.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)
    log.info(f"Getting model class <{cfg.model._target_}>")
    model_class = hydra.utils.get_class(cfg.model._target_)
    lit_server_api = hydra.utils.instantiate(cfg.serve.api, model_class=model_class)
    # Create the LitServe server with the MNISTServeAPI
    server = ls.LitServer(lit_server_api, accelerator=cfg.serve.accelerator, max_batch_size=cfg.serve.max_batch_size)
    log.info("Initialized LitServe server")
    # Run the server on port 8000
    log.info(f"Starting LitServe server on port {cfg.serve.port}")
    server.run(port=cfg.serve.port)


@hydra.main(version_base="1.3", config_path="../configs", config_name="serve.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for serving.

    Args:
        cfg: A DictConfig configuration composed by Hydra.

    Returns:
        Optional[float] with optimized metric value.
    """
    # apply extra utilities (print config tree, enforce tags, ...)
    extras(cfg)

    serve(cfg)


if __name__ == "__main__":
    main()
