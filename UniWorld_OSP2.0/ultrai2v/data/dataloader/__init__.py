from .sampler import data_sampler
from .collator import data_collator

ultra_samplers = {}
ultra_samplers.update(data_sampler)

ultra_collators = {}
ultra_collators.update(data_collator)