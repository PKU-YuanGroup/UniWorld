import os
import gc

import random
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load
from accelerate.utils import load
from torch.distributed.checkpoint.state_dict import (
    _init_optim_state,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import distribute_tensor, DTensor
from torchdata.stateful_dataloader import StatefulDataLoader

from ultrai2v.utils.utils import is_npu_available

MODEL_CHECKPOINT = "model_state_dict.pt"
OPTIM_CHECKPOINT = "optim_state_dict.pt"
RNG_CHECKPOINT_DIR = "random_states"
DATALOADER_CHECKPOINT_DIR = "dataloader_states"
PARAMS = "params"
PREFIX = "iter_"


def get_latest_checkpoint_folder(path):
    max_num = None
    if not os.path.exists(path):
        return max_num
    for name in os.listdir(path):
        folder_path = os.path.join(path, name)
        if os.path.isdir(folder_path):
            name = name.replace(PREFIX, "")
            try:
                num = int(name)
                if max_num is None or num > max_num:
                    max_num = num
            except ValueError:
                pass  # Skip non-numeric folder names
    return max_num


class Checkpointer:
    def __init__(self, folder: str, dcp_api: bool):
        self.folder = folder
        self.dcp_api = dcp_api
        self.save_root_dir = folder
        self._last_training_iteration = get_latest_checkpoint_folder(
            self.save_root_dir
        )

    @property
    def last_training_iteration(self):
        return self._last_training_iteration
    
    @staticmethod
    def load_state_dict(model: FSDPModule, full_sd: dict, dcp_api: bool = False):
        if dcp_api:
            set_model_state_dict(
                model=model,
                model_state_dict=full_sd,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )
            del full_sd
            gc.collect()
            return
        meta_sharded_sd = model.state_dict()
        sharded_sd = {}
        for param_name, full_tensor in full_sd.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)
        # choose `assign=True` since we cannot call `copy_` on meta tensor
        missing_keys, unexpected_keys = model.load_state_dict(sharded_sd, strict=False, assign=True)
        if torch.distributed.get_rank() == 0:
            print("missing_keys", missing_keys)
            print("unexpected_keys", unexpected_keys)
        del full_sd, sharded_sd
        gc.collect()
    
    @staticmethod
    def load_model_from_path(model: FSDPModule, model_path: str, dcp_api: bool = False):
        print(f'load model from {model_path}')
        if not model_path.endswith(".safetensors"):
            full_sd = torch.load(
                model_path, mmap=True, weights_only=True, map_location="cpu"
            )
        else:
            full_sd = safe_load(model_path, device="cpu")
        Checkpointer.load_state_dict(model, full_sd, dcp_api=dcp_api)
        del full_sd

    def load_model(self, model: FSDPModule, ema: bool = False):
        model_checkpoint = f"ema_{MODEL_CHECKPOINT}" if ema else MODEL_CHECKPOINT
        last_model_checkpoint = (
            f"{self.save_root_dir}/{PREFIX}{self.last_training_iteration:09d}/{model_checkpoint}"
        )
        if os.path.exists(last_model_checkpoint):
            print(f'resume last_model_checkpoint from {last_model_checkpoint}')
            full_sd = torch.load(
                last_model_checkpoint, mmap=True, weights_only=True, map_location="cpu"
            )
            self.load_state_dict(model, full_sd, dcp_api=self.dcp_api)
            del full_sd
        else:
            print(f'warning! nothing find in {last_model_checkpoint}')

    def load_optim(self, model: FSDPModule, opt: torch.optim.Optimizer):
        last_optim_checkpoint = (
            f"{self.save_root_dir}/{PREFIX}{self.last_training_iteration:09d}/{OPTIM_CHECKPOINT}"
        )
        if os.path.exists(last_optim_checkpoint):
            print(f'resume last_optim_checkpoint from {last_optim_checkpoint}')
            full_sd = torch.load(
                last_optim_checkpoint, mmap=True, weights_only=True, map_location="cpu"
            )
            if self.dcp_api:
                set_optimizer_state_dict(
                    model=model,
                    optimizers=opt,
                    optim_state_dict=full_sd,
                    options=StateDictOptions(
                        full_state_dict=True,
                        broadcast_from_rank0=True,
                    ),
                )
                del full_sd
                gc.collect()
                return
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
                    full_param_state = full_state.get(full_pid, None)
                    if full_param_state is None:
                        if torch.distributed.get_rank() == 0:
                            print(f"WARN: param [{full_pid}] does NOT have param_state")
                        continue
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
            del full_sd
            gc.collect()
        else:
            print(f'warning! nothing find in {last_optim_checkpoint}')

    def _get_full_model_state_dict(self, model: FSDPModule):
        if self.dcp_api:
            return get_model_state_dict(
                model=model,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )

        sharded_sd = model.state_dict()
        cpu_state_dict = {}
        for param_name, sharded_param in sharded_sd.items():
            full_param = sharded_param.full_tensor()
            if torch.distributed.get_rank() == 0:
                cpu_state_dict[param_name] = full_param.cpu()
            else:
                del full_param
        return cpu_state_dict

    def _get_full_optimizer_state_dict(
        self,
        model: FSDPModule,
        opt: torch.optim.Optimizer,
    ):
        if self.dcp_api:
            return get_optimizer_state_dict(
                model=model,
                optimizers=opt,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
        is_rank_zero = torch.distributed.get_rank() == 0
        sharded_sd = opt.state_dict()
        sharded_state = sharded_sd["state"]
        full_state = {}
        for group_id, sharded_group in sharded_state.items():
            group_state = {}
            for attr, sharded_tensor in sharded_group.items():
                if isinstance(sharded_tensor, DTensor):
                    # "exp_avg" in AdamW is `DTensor`
                    full_tensor = sharded_tensor.full_tensor()
                else:
                    # "step" in AdamW is plain tensor
                    full_tensor = sharded_tensor
                if is_rank_zero:
                    group_state[attr] = full_tensor.cpu()
                else:
                    del full_tensor
            if is_rank_zero:
                full_state[group_id] = group_state
            else:
                del group_state
        if is_rank_zero:
            return {
                "param_groups": sharded_sd["param_groups"],
                "state": full_state,
            }
        else:
            return {}

    def load_dataloader_state_dict(self, dataloader):
        assert isinstance(dataloader, StatefulDataLoader), "only StatefulDataLoader has state."
        last_dataloader_checkpoint_dir = (
            f"{self.save_root_dir}/{PREFIX}{self.last_training_iteration:09d}/{DATALOADER_CHECKPOINT_DIR}"
        )
        if os.path.exists(last_dataloader_checkpoint_dir):
            print(f'resume dataloader_state_dict from {last_dataloader_checkpoint_dir}')
            if is_npu_available(): # npu will raise serialization error if we use 'load' func from accelerate
                dataloader_state_dict = torch.load(f"{last_dataloader_checkpoint_dir}/rank_{torch.distributed.get_rank():06d}.pkl", weights_only=False, map_location="cpu")
            else:
                dataloader_state_dict = load(f"{last_dataloader_checkpoint_dir}/rank_{torch.distributed.get_rank():06d}.pkl", map_location="cpu")
            dataloader.load_state_dict(dataloader_state_dict)
            del dataloader_state_dict
            gc.collect()
        else:
            print(f'warning! nothing find in {last_dataloader_checkpoint_dir}')
    
    def _get_full_dataloader_state_dict(self, dataloader):
        assert isinstance(dataloader, StatefulDataLoader), "only StatefulDataLoader has state."
        return dataloader.state_dict()
        
    def load_rng_state_dict(self):
        last_rng_checkpoint_dir = (
            f"{self.save_root_dir}/{PREFIX}{self.last_training_iteration:09d}/{RNG_CHECKPOINT_DIR}"
        )
        if os.path.exists(last_rng_checkpoint_dir):
            print(f'resume rng_state_dict from {last_rng_checkpoint_dir}')
            if is_npu_available(): # npu will raise serialization error if we use 'load' func from accelerate
                rng_state_dict = torch.load(f"{last_rng_checkpoint_dir}/rank_{torch.distributed.get_rank():06d}.pkl", weights_only=False, map_location="cpu")
            else:
                rng_state_dict = load(f"{last_rng_checkpoint_dir}/rank_{torch.distributed.get_rank():06d}.pkl", map_location="cpu")
            random.setstate(rng_state_dict["random_state"])
            np.random.set_state(rng_state_dict["numpy_random_seed"])
            torch.set_rng_state(rng_state_dict["torch_manual_seed"])
            torch.cuda.set_rng_state_all(rng_state_dict["torch_cuda_manual_seed"])
            if is_npu_available():
                import torch_npu
                torch_npu.npu.set_rng_state_all(rng_state_dict["torch_npu_manual_seed_all"])
                torch_npu.npu.set_rng_state(rng_state_dict["torch_npu_manual_seed"])
            del rng_state_dict
            gc.collect()
        else:
            print(f'warning! nothing find in {last_rng_checkpoint_dir}')

    def _get_full_rng_state_dict(self):
        states = {}
        states["random_state"] = random.getstate()
        states["numpy_random_seed"] = np.random.get_state()
        states["torch_manual_seed"] = torch.get_rng_state()
        states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()
        if is_npu_available():
            import torch_npu
            states["torch_npu_manual_seed_all"] = torch_npu.npu.get_rng_state_all()
            states["torch_npu_manual_seed"] = torch_npu.npu.get_rng_state()
        else:
            states["torch_npu_manual_seed_all"] = None
            states["torch_npu_manual_seed"] = None
        return states

    def save(self, model: FSDPModule, optim: torch.optim.Optimizer, dataloader: StatefulDataLoader, iteration: int):
        model_state_dict = self._get_full_model_state_dict(model)
        optim_state_dict = self._get_full_optimizer_state_dict(model, optim)
        rng_state_dict = self._get_full_rng_state_dict()
        dataloader_state_dict = self._get_full_dataloader_state_dict(dataloader)
        new_training_iteration = f"{PREFIX}{iteration:09d}"
        new_checkpoint_folder = f"{self.save_root_dir}/{new_training_iteration}"
        rng_checkpoint_dir = f"{new_checkpoint_folder}/{RNG_CHECKPOINT_DIR}"
        dataloader_checkpoint_dir = f"{new_checkpoint_folder}/{DATALOADER_CHECKPOINT_DIR}"
        if torch.distributed.get_rank() == 0:
            os.makedirs(rng_checkpoint_dir, exist_ok=True)
            os.makedirs(dataloader_checkpoint_dir, exist_ok=True)
            new_model_checkpoint = f"{new_checkpoint_folder}/{MODEL_CHECKPOINT}"
            new_optim_checkpoint = f"{new_checkpoint_folder}/{OPTIM_CHECKPOINT}"
            os.makedirs(new_checkpoint_folder, exist_ok=True)
            torch.save(model_state_dict, new_model_checkpoint)
            torch.save(optim_state_dict, new_optim_checkpoint)
        torch.distributed.barrier()
        torch.save(rng_state_dict, f"{rng_checkpoint_dir}/rank_{torch.distributed.get_rank():06d}.pkl")
        torch.save(dataloader_state_dict, f"{dataloader_checkpoint_dir}/rank_{torch.distributed.get_rank():06d}.pkl")
        del model_state_dict
        del optim_state_dict
        del rng_state_dict
        del dataloader_state_dict

    def save_ema_model(self, ema_model: FSDPModule, iteration: int):
        model_state_dict = self._get_full_model_state_dict(ema_model)
        if torch.distributed.get_rank() == 0:
            new_training_iteration = f"{PREFIX}{iteration:09d}"
            new_checkpoint_folder = f"{self.save_root_dir}/{new_training_iteration}"
            new_model_checkpoint = f"{new_checkpoint_folder}/ema_{MODEL_CHECKPOINT}"
            os.makedirs(new_checkpoint_folder, exist_ok=True)
            torch.save(model_state_dict, new_model_checkpoint)
        del model_state_dict