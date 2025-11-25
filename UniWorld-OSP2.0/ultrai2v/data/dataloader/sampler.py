import torch
import math

from typing import Optional
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import Stateful
# from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
import logging

#adopted from https://github.com/meta-pytorch/data/blob/main/torchdata/stateful_dataloader/sampler.py#L18
class StatefulDistributedSampler(DistributedSampler, Stateful):

    _GENERATOR = "generator"
    _YIELDED = "yielded"

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        # initialize generator
        if self.shuffle:
            self.g = torch.Generator()
            self.g.manual_seed(self.seed)
            self.generator_state = self.g.get_state()

        # create first epoch’s indices
        self.indices = self._get_indices()

        # for restoring
        self.next_yielded = None
        self.yielded = 0  # only used for state save/load, NOT used for iteration


    def __iter__(self):
        logging.info(
            f"[Sampler.__iter__] rank={self.rank}, first 5 indices={self.indices[:5]}, "
            f"starting yielded (restored)={self.yielded}"
        )
        return _StatefulDistributedSamplerIter(self)

    def _get_indices(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=self.g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return indices


    def state_dict(self):
        return {
            self._YIELDED: self.yielded,
            self._GENERATOR: self.generator_state,
        }

    def load_state_dict(self, state_dict):
        self.next_yielded = state_dict[self._YIELDED]
        self.generator_state = state_dict[self._GENERATOR]
        self.g.set_state(self.generator_state)

        # regenerate indices using restored generator state
        self.indices = self._get_indices()

        if self.next_yielded is not None:
            self.yielded = self.next_yielded
            self.next_yielded = None


class _StatefulDistributedSamplerIter:
    """
    NEW iterator object created for each __iter__() call.
    """

    def __init__(self, sampler: StatefulDistributedSampler):
        self.sampler = sampler
        self.yielded = sampler.yielded  # possibly restored state

    def __iter__(self):
        return self

    def __next__(self):
        # finished one full epoch
        if self.yielded >= self.sampler.num_samples:
            # reshuffle for next epoch
            self.sampler.indices = self.sampler._get_indices()
            self.sampler.yielded = 0   # Reset sampler-level for next epoch
            self.yielded = 0

            logging.info(
                f"[SamplerIter] reshuffle, rank={self.sampler.rank}, "
                f"first 5 indices={self.sampler.indices[:5]}"
            )

            raise StopIteration()

        idx = self.sampler.indices[self.yielded]
        self.yielded += 1

        # keep sampler-level yielded synced (for state saving)
        self.sampler.yielded = self.yielded

        return idx


data_sampler = {
    'stateful_distributed': StatefulDistributedSampler
}