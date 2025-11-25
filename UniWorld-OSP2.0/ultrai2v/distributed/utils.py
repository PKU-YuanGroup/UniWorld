import os
import logging
import torch
import torch.distributed as dist
from datetime import timedelta
from torch.distributed.tensor import DTensor, Replicate, Shard
from typing import Iterable
from ultrai2v.utils.utils import str_to_precision, precision_to_str

def setup_distributed_env(backend: str = "nccl", timeout: int = 3600):
    """ Initialize distributed environment. """
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

def cleanup_distributed_env():
    """ Clean up distributed environment. """
    dist.destroy_process_group()
   
def set_modules_to_forward_prefetch(main_block_list, num_to_forward_prefetch=2):
    for i, block in enumerate(main_block_list):
        if i >= len(main_block_list) - num_to_forward_prefetch:
            break
        blocks_to_prefetch = [
            main_block_list[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        block.set_modules_to_forward_prefetch(blocks_to_prefetch)


def set_modules_to_backward_prefetch(main_block_list, num_to_backward_prefetch=2):
    for i, block in enumerate(main_block_list):
        if i < num_to_backward_prefetch:
            continue
        blocks_to_prefetch = [
            main_block_list[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        block.set_modules_to_backward_prefetch(blocks_to_prefetch)

def gather_data_from_all_ranks(data, dim=0, group=None):
    """ gather data from all ranks, return a tensor with data from all ranks """
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return data
    gather_list = [torch.empty_like(data) for _ in range(world_size)]
    dist.all_gather(gather_list, data, group=group)
    return torch.stack(gather_list, dim=dim)

def broadcast_tensor_list(tensors, group_src=0, group=None):
    """ Broadcast a list of tensors from source rank to all other ranks. """
    if group is None:
        group = dist.group.WORLD
    group_rank = dist.get_rank(group)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    # broadcast tensor list length
    if group_rank == group_src:
        nums = torch.tensor(len(tensors), device=tensors[0].device, dtype=torch.int)
    else:
        nums = torch.tensor(0, device=device, dtype=torch.int)
    dist.broadcast(nums, group=group, group_src=group_src)
    nums = int(nums.item())

    tensors = tensors if group_rank == group_src else [None] * nums

    for i in range(nums):
        # broadcast tensor ndim
        if group_rank == group_src:
            ndim = torch.tensor(len(tensors[i].shape), device=tensors[i].device, dtype=torch.int)
        else:
            ndim = torch.tensor(0, device=device, dtype=torch.int)
        dist.broadcast(ndim, group=group, group_src=group_src)
        ndim = int(ndim.item())

        # broadcast tensor shape
        if group_rank == group_src:
            shape = torch.tensor(tensors[i].shape, device=tensors[i].device, dtype=torch.int)
        else:
            shape = torch.empty((ndim, ), device=device, dtype=torch.int)
        dist.broadcast(shape, group=group, group_src=group_src)
        shape = tuple(shape.tolist())

        # broadcast tensor dtype
        if group_rank == group_src:
            dtype_str = [precision_to_str(tensors[i].dtype)]
        else:
            dtype_str = [None]
        dist.broadcast_object_list(dtype_str, group=group, group_src=group_src)
        dtype = str_to_precision(dtype_str[0])

        # broadcast tensor data
        if group_rank != group_src:
            tensors[i] = torch.empty(shape, device=device, dtype=dtype)
        dist.broadcast(tensors[i], group=group, group_src=group_src)

    return tensors

def gather_tensor_list_to_one(tensors, group_dst=0, group=None, active_ranks=None, to_cpu=True):
    """ gather a list of tensors from all other ranks to dst rank. """
    if group is None:
        group = dist.group.WORLD
    group_rank = dist.get_rank(group)
    if active_ranks is None:
        active_ranks = range(dist.get_world_size(group))

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    gathered_tensors = []
    if group_rank == group_dst:
        for r in active_ranks:
            if r != 0:
                # recv tensor list length
                nums = torch.tensor(0, device=device, dtype=torch.int)
                dist.recv(nums, group=group, group_src=r)
                nums = int(nums.item())

                for i in range(nums):
                    # recv tensor ndim
                    ndim = torch.tensor(0, device=device, dtype=torch.int)
                    dist.recv(ndim, group=group, group_src=r)
                    ndim = int(ndim.item())

                    # recv tensor shape
                    shape = torch.empty((ndim, ), device=device, dtype=torch.int)
                    dist.recv(shape, group=group, group_src=r)
                    shape = tuple(shape.tolist())

                    # recv tensor dtype
                    dtype_str = [None]
                    dist.recv_object_list(dtype_str, group=group, group_src=r)
                    dtype = str_to_precision(dtype_str[0])

                    # recv tensor data
                    tensor = torch.empty(shape, device=device, dtype=dtype)
                    dist.recv(tensor, group=group, group_src=r)
                    if to_cpu: tensor = tensor.cpu()
                    gathered_tensors.append(tensor)
            else:
                if to_cpu: tensors = [tensor.cpu() for tensor in tensors]
                gathered_tensors.extend(tensors)
    elif group_rank in active_ranks:
        # send tensor list length
        nums = torch.tensor(len(tensors), device=tensors[0].device, dtype=torch.int)
        dist.send(nums, group=group, group_dst=group_dst)

        for i in range(len(tensors)):
            # send tensor ndim
            ndim = torch.tensor(len(tensors[i].shape), device=tensors[i].device, dtype=torch.int)
            dist.send(ndim, group=group, group_dst=group_dst)

            # send tensor shape
            shape = torch.tensor(tensors[i].shape, device=tensors[i].device, dtype=torch.int)
            dist.send(shape, group=group, group_dst=group_dst)

            # send tensor dtype
            dtype_str = [precision_to_str(tensors[i].dtype)]
            dist.send_object_list(dtype_str, group=group, group_dst=group_dst)

            # send tensor data
            dist.send(tensors[i], group=group, group_dst=group_dst)
    return gathered_tensors if group_rank == group_dst else None
