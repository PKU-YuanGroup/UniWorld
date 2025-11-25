from .t2v_dataset import dataset as t2v
from .flashi2v_dataset import dataset as flashi2v

ultra_datasets = {}
ultra_datasets.update(t2v)
ultra_datasets.update(flashi2v)

