from .flow_matching import flow_scheduler as normal_flow
from .flashi2v_flow_matching import flow_scheduler as flashi2v_flow

schedulers = {}
schedulers.update(normal_flow)
schedulers.update(flashi2v_flow)