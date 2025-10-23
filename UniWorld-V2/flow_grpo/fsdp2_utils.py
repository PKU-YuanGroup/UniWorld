"""reference: https://github.com/pytorch/torchtune/blob/main/torchtune/training/_distributed.py"""

import os
from typing import Any, Callable, Dict, List, Optional, cast

import safetensors
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed._tensor import DTensor, distribute_tensor
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    _init_optim_state,
    get_optimizer_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torchao.dtypes.nf4tensor import NF4Tensor, to_nf4


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def prepare_fsdp_model(
    model: nn.Module,
    shard_conditions: List[Callable[[str, nn.Module], bool]],
    reshard_after_forward: bool = True,
    dp_mesh: Optional[DeviceMesh] = None,
    cpu_offload: bool = False,
    weight_dtype: torch.dtype = torch.bfloat16,
):
    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": dp_mesh,
    }  # dp_mesh is None means distributed to all nodes.

    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=weight_dtype,
        reduce_dtype=weight_dtype,
        output_dtype=weight_dtype,
    )

    num_layers_sharded = 0
    for n, m in reversed(list(model.named_modules())):
        if any(shard_condition(n, m) for shard_condition in shard_conditions):
            fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError(
            "No layer modules were sharded. Please check if shard conditions are working as expected."
        )

    fully_shard(model, **fsdp_kwargs)


def save_state(
    output_dir,
    global_step,
    model,
    is_fsdp,
    save_key_filter=None,
    optimizer=None,
    dataloader=None,
    sampler=None,
    scaler=None,
    lr_scheduler=None,
):
    """Save FSDP2 state dict to file."""
    full_state_dict = {}
    full_optimizer_state_dict = {}
    full_dataloader_state_dict = {}
    full_sampler_state_dict = {}
    full_scaler_state_dict = {}
    lr_scheduler_state_dict = {}

    training_state = {"global_step": global_step}

    for key, value in model.state_dict().items():
        if save_key_filter is None or save_key_filter in key:
            if not is_fsdp:
                full_state_dict[key] = value
            else:
                full_state_dict[key] = value.full_tensor()

    if not is_fsdp:
        full_optimizer_state_dict = optimizer.state_dict()
    else:
        options = StateDictOptions(
            full_state_dict=True, broadcast_from_rank0=True, cpu_offload=True
        )
        full_optimizer_state_dict = get_optimizer_state_dict(
            model=model, optimizers=optimizer, options=options
        )

    if dataloader is not None:
        full_dataloader_state_dict = dataloader.state_dict()

    if sampler is not None:
        full_sampler_state_dict = sampler.state_dict()

    if scaler is not None:
        full_scaler_state_dict = scaler.state_dict()

    if lr_scheduler is not None:
        lr_scheduler_state_dict = lr_scheduler.state_dict()

    if dist.get_rank() == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        safetensors.torch.save_file(
            full_state_dict, os.path.join(output_dir, "model.safetensors")
        )
        torch.save(training_state, os.path.join(output_dir, "training_state.pth"))
        if len(full_optimizer_state_dict) > 0:  # optimizer is not None
            torch.save(
                full_optimizer_state_dict, os.path.join(output_dir, "optimizer.pth")
            )
        if len(full_dataloader_state_dict) > 0:  # dataloader is not None
            torch.save(
                full_dataloader_state_dict, os.path.join(output_dir, "dataloader.pth")
            )
        if len(full_sampler_state_dict) > 0:  # sampler is not None
            torch.save(full_sampler_state_dict, os.path.join(output_dir, "sampler.pth"))

        if len(full_scaler_state_dict) > 0:  # scaler is not None
            torch.save(full_scaler_state_dict, os.path.join(output_dir, "scaler.pth"))

        if len(lr_scheduler_state_dict) > 0:  # lr_scheduler is not None
            torch.save(
                lr_scheduler_state_dict, os.path.join(output_dir, "lr_scheduler.pth")
            )


def load_model_state(model, path, device, is_fsdp=False, fsdp_cpu_offload=False):
    """Load FSDP2 state dict from file."""
    full_state_dict = safetensors.torch.load_file(
        os.path.join(path, "model.safetensors")
    )

    # # Replace all lora_*.default with lora_* in key
    # keys_to_replace = [
    #     key for key in full_state_dict.keys() if "lora" in key and "default" in key
    # ]
    # for key in keys_to_replace:
    #     new_key = key.replace("default.", "")
    #     full_state_dict[new_key] = full_state_dict[key]
    #     del full_state_dict[key]

    if is_fsdp:
        _load_from_full_model_state_dict(
            model, full_state_dict, device=device, cpu_offload=fsdp_cpu_offload
        )
    else:
        model.load_state_dict(full_state_dict, strict=False)


def load_optimizer_state(optimizer, path, is_fsdp=False):
    full_optimizer_state_dict = torch.load(os.path.join(path, "optimizer.pth"))
    if is_fsdp:
        _load_from_full_optimizer_state_dict(optimizer, full_optimizer_state_dict)
    else:
        optimizer.load_state_dict(full_optimizer_state_dict)


def load_state(path, dataloader=None, sampler=None, scaler=None, lr_scheduler=None):
    training_state = torch.load(os.path.join(path, "training_state.pth"))
    if dataloader is not None:
        dataloader.load_state_dict(torch.load(os.path.join(path, "dataloader.pth")))
    if sampler is not None:
        sampler.load_state_dict(torch.load(os.path.join(path, "sampler.pth")))
    if scaler is not None:
        scaler.load_state_dict(torch.load(os.path.join(path, "scaler.pth")))
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(torch.load(os.path.join(path, "lr_scheduler.pth")))
    return training_state["global_step"]


def _load_from_full_model_state_dict(
    model: "FSDPModule",  # noqa
    full_sd: Dict[str, Any],
    device: torch.device,
    strict: bool = False,
    cpu_offload: bool = False,
):
    # has_nf4 = any(
    #     hasattr(param, "_local_tensor") and isinstance(param._local_tensor, NF4Tensor)
    #     for param in model.parameters()
    # )
    meta_sharded_sd = model.state_dict()

    sharded_sd = {}
    for param_name, full_tensor in full_sd.items():
        sharded_meta_param = meta_sharded_sd.get(param_name)
        full_tensor = full_tensor.to(sharded_meta_param.dtype).to(device)
        if hasattr(sharded_meta_param, "_local_tensor") and isinstance(
            sharded_meta_param._local_tensor, NF4Tensor
        ):
            block_size = sharded_meta_param._local_tensor.block_size
            scaler_block_size = sharded_meta_param._local_tensor.scaler_block_size
            full_tensor = to_nf4(
                full_tensor,
                block_size=block_size,
                scaler_block_size=scaler_block_size,
            )
            # replicating logic from `_fsdp_param.py`` `_init_sharded_param`
            # otherwise `distribute_tensor(DTensor(local=NF4))`
            # requires dispatching `c10d.scatter_``
            # long-term solution is `swap_tensor`
            mesh = sharded_meta_param.device_mesh
            if mesh.ndim > 1:
                raise NotImplementedError(f"only support 1D FSDP but got {mesh.ndim=}")
            shard_mesh_dim = 0
            shard_world_size = mesh.size(shard_mesh_dim)
            shard_rank = cast(
                torch.distributed.ProcessGroup, mesh.get_group(shard_mesh_dim)
            ).rank()
            chunk = list(torch.chunk(full_tensor, shard_world_size, dim=0))[shard_rank]
            sharded_param = full_tensor.new_zeros(chunk.size())
            sharded_param[: chunk.size(0)].copy_(chunk)
            # TODO: change to from_local API (need to add view support for NF4)
            sharded_tensor = DTensor(
                local_tensor=sharded_param,
                spec=DTensorSpec(
                    mesh=sharded_meta_param.device_mesh,
                    placements=sharded_meta_param.placements,
                    tensor_meta=TensorMeta(
                        shape=sharded_meta_param.size(),
                        dtype=sharded_meta_param.dtype,
                        stride=sharded_meta_param.stride(),
                    ),
                ),
                requires_grad=sharded_meta_param.requires_grad,
            )
        elif not hasattr(sharded_meta_param, "device_mesh"):
            # In cases where parts of the model aren't sharded, some parameters will be plain tensors
            sharded_tensor = full_tensor
        else:
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
        if cpu_offload:
            sharded_tensor = sharded_tensor.cpu()
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)
    # choose `assign=True` since we cannot call `copy_` on meta tensor
    return model.load_state_dict(sharded_sd, strict=strict, assign=True)


def _load_from_full_optimizer_state_dict(
    opt: Optimizer,
    full_sd: Dict[str, Any],
) -> None:
    PARAMS = "params"  # noqa: N806
    _init_optim_state(opt)
    param_groups = opt.state_dict()["param_groups"]
    state = opt.state_dict()["state"]
    full_param_groups = full_sd["param_groups"]
    full_state = full_sd["state"]
    for param_group, full_param_group in zip(param_groups, full_param_groups):
        for key, value in full_param_group.items():
            if key == PARAMS:
                continue
            param_group[key] = value
        for pid, full_pid in zip(param_group[PARAMS], full_param_group[PARAMS]):
            if pid not in state:
                continue
            param_state = state[pid]
            full_param_state = full_state[full_pid]
            for attr, full_tensor in full_param_state.items():
                sharded_tensor = param_state[attr]
                if isinstance(sharded_tensor, DTensor):
                    # exp_avg is DTensor
                    param_state[attr] = distribute_tensor(
                        full_tensor,
                        sharded_tensor.device_mesh,
                        sharded_tensor.placements,
                    )
                else:
                    # step is plain tensor
                    param_state[attr] = full_tensor
    opt.load_state_dict(
        {
            "param_groups": param_groups,
            "state": state,
        }
    )