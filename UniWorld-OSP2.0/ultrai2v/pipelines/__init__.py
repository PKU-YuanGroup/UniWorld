from .t2v_pipeline import pipeline as t2v
from .flashi2v_pipeline import pipeline as flashi2v
from  .flashi2v_qwenvl_pipeline   import pipeline as flashi2v_qwenvl


pipelines = {}
pipelines.update(t2v)
pipelines.update(flashi2v)
pipelines.update(flashi2v_qwenvl)
__all__ = [
    'pipelines'
]