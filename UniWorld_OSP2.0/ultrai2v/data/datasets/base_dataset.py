import random
import numpy as np
import torch

from torch.utils.data import Dataset
from torchdata.stateful_dataloader import Stateful

# base stateful dataset
class BaseDataset(Dataset, Stateful):
    def state_dict(self):
        return {
            "rng_random": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
        }

    def load_state_dict(self, state_dict):
        rng_random = state_dict.pop("rng_random", None)
        rng_numpy = state_dict.pop("rng_numpy", None)
        rng_torch = state_dict.pop("rng_torch", None)
        assert rng_random is not None and rng_numpy is not None and rng_torch is not None
        random.setstate(rng_random)
        np.random.set_state(rng_numpy)
        torch.set_rng_state(rng_torch)