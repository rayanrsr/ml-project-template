"""Utility functions for various tasks."""

import contextlib
import fcntl
import tempfile
import warnings
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

import requests
from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    Args:
        cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    Args:
        task_func: The task function to be wrapped.

    Returns:
        The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as e:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise e  # noqa: TRY201

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> None | float:
    """Safely retrieves value of the metric logged in LightningModule.

    Args:
        metric_dict: A dict containing metric values.
        metric_name: If provided, the name of the metric to retrieve.

    Returns:
        If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise ValueError(f"Metric value not found! <metric_name={metric_name}>\n")  # noqa: TRY003

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return float(metric_value)


# The following functions are useful to make your different operations multi-process safe


@contextlib.contextmanager
def file_lock(filename: Path, mode: str = "r") -> Any:
    """This context manager is used to acquire a file lock on a file.

    particularly useful for shared resources in multi-process environments (multi GPU/TPU training).

    The file is only used as a lock handle: it is created if it does not exist and is never
    truncated. Note that this relies on `fcntl`, so it is POSIX-only.

    Args:
        filename: Path to the file to lock
        mode: The mode to open the file with, either "r" (shared lock) or "w" (exclusive lock)

    Raises:
        ValueError: If the mode is invalid (neither "r" nor "w")
    """
    if mode not in ("r", "w"):
        raise ValueError("Expected mode 'r' or 'w'.")  # noqa: TRY003

    # open in append mode: it creates the file if needed without truncating its content
    with open(filename, "a+") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH if mode == "r" else fcntl.LOCK_EX)
            yield f
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def file_lock_operation(file_name: str, operation: Callable) -> Any:
    """This function is used to perform an operation on a file while acquiring a lock on it.

    The lock is acquired using the `file_lock` context manager, based on a file stored in the
    system temporary folder with a stable name, so that every process contends on the same
    file (a per-call temporary directory would give each process its own lock file, and hence
    no mutual exclusion at all).

    Args:
        file_name: Name of the lock file, shared by all processes
        operation: The operation to perform on the file

    Returns:
        The result of the operation
    """
    file_path = Path(tempfile.gettempdir()) / file_name
    with file_lock(file_path, mode="w"):
        result = operation(file_path)
    return result


def fetch_data(url: str) -> dict[str, Any] | None:
    """Fetches data from a URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return cast(dict, response.json())
    return None


def process_data(url: str) -> int:
    """Fetches data from a URL and processes it."""
    data = fetch_data(url)
    if data:
        return len(data)  # Just an example of processing, counting data length
    return 0
