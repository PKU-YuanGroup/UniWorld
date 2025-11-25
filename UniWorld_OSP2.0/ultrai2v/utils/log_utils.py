import logging
import torch


logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)

def get_logger():
    return logging.getLogger(__name__)


def log_on_main_process(logger, msg):
    """helper function to log only on global rank 0"""
    if torch.distributed.get_rank(torch.distributed.group.WORLD) == 0:
        logger.info(f" {msg}")


def verify_min_gpu_count(min_gpus: int = 1):
    """ verification that we have at least 1 gpu to run dist examples """
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    if not (has_gpu and gpu_count >= min_gpus):
        raise ValueError(
            f"Distributed examples require at least {min_gpus} GPUs. Detected {gpu_count} GPUs."
        )