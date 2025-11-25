from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.placement_types import Placement
from torch.distributed.tensor.parallel import ParallelStyle

class Redistribution(ParallelStyle):
    def __init__(
        self,
        *,
        original_layouts: Union[Placement, tuple[Placement]],
        target_layouts: Union[Placement, tuple[Placement]],
        use_local_output: bool = True,
    ):
        self.original_layouts = [original_layouts]
        self.target_layouts = [target_layouts]
        self.use_local_output = use_local_output

        self.original_layouts = [
            (layout,) if isinstance(layout, Placement) else layout
            for layout in self.original_layouts
        ]

        self.target_layouts = [
            (layout,) if isinstance(layout, Placement) else layout
            for layout in self.target_layouts
        ]        

        for orig_layout, target_layout in zip(self.original_layouts, self.target_layouts):
            assert len(orig_layout) == len(target_layout), (
                "original_layout and target_layout should have same length!"
            )

    def _redistribute(self, outputs, device_mesh):
        target_outputs = []
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        if len(outputs) != len(self.target_layouts):
            raise ValueError(
                "module outputs and target_layouts should have same length!"
            )
        for out, orig_layout, tgt_layout in zip(
            outputs, self.original_layouts, self.target_layouts
        ):
            if orig_layout is not None:
                if isinstance(out, DTensor):
                    dt_out = out
                else:
                    dt_out = DTensor.from_local(
                        out, device_mesh, orig_layout, run_check=False
                    )

                if orig_layout != tgt_layout:
                    dt_out = dt_out.redistribute(placements=tgt_layout)
                target_outputs.append(
                    dt_out.to_local() if self.use_local_output else dt_out
                )
            else:
                target_outputs.append(out)
        if len(target_outputs) == 1:
            return target_outputs[0]
        else:
            return tuple(target_outputs)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        
        assert isinstance(module, nn.Identity), f"Redistribute should be bound to nn.Identifiy() to perform redistribution, but module is an instance of {module.__class__.__name__}!"

        module.register_forward_hook(
            lambda _, inputs, outputs: self._redistribute(outputs, device_mesh)
        )  # type: ignore[misc, call-arg]
        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"original_layout={self.original_layouts}, "
        tmpstr += f"target_layout={self.target_layouts}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr

if __name__ == "__main__":
    import os
    from ultrai2v.distributed.utils import setup_distributed_env, cleanup_distributed_env
    from torch.distributed.device_mesh import init_device_mesh
    
    setup_distributed_env()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    weight_dtype = torch.float32

    samples = torch.arange(0, 16, dtype=weight_dtype, device=device).reshape(2, 4, 2)
    if rank == 0:
        print(f"original samples: {samples}")
    device_mesh = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("dp", "cp", "ccp"))
    cp_mesh = device_mesh

    all_shard = Redistribution(original_layouts=(Replicate(), Replicate()), target_layouts=(Shard(1), Shard(2)))._apply(nn.Identity(), device_mesh=cp_mesh)
    all_gather = Redistribution(original_layouts=(Shard(1), Shard(2)), target_layouts=(Replicate()))._apply(nn.Identity(), device_mesh=cp_mesh)

    samples = all_shard(samples)
    print(f"{'=' * 10}After Sharding{'=' * 10}")
    print(f"rank={rank}, samples={samples}")
    print(f"{'=' * (10 + len('After Sharding') + 10)}")
    
    samples = all_gather(samples)
    print(f"{'=' * 10}After Gather{'=' * 10}")
    print(f"rank={rank}, samples={samples}")
    print(f"{'=' * (10 + len('After Gather') + 10)}")
    cleanup_distributed_env()