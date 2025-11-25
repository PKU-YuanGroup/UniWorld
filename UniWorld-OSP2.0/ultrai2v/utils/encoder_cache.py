import torch
import torch.distributed as dist
from ultrai2v.distributed.utils import broadcast_tensor_list

class EncoderCacheManager:
    def __init__(self, tp_cp_group: dist.ProcessGroup = None):
        self.tp_cp_group = tp_cp_group
        self.tp_cp_size = dist.get_world_size(group=tp_cp_group) if tp_cp_group is not None else 1

        self.vae_cache = None
        self.text_cache = None

    def save_cache(self, vae_latents_list, text_embeds_list):
        self.vae_cache = vae_latents_list
        self.text_cache = text_embeds_list

    def get_cache_from_rank(self, src_rank=0):
        if self.vae_cache is None or self.text_cache is None:
            return ValueError("Cache is empty!")
        rank = dist.get_rank(group=self.tp_cp_group) if self.tp_cp_group is not None else 0
        if rank == src_rank:
            vae_latents_list = self.vae_cache
            text_embeds_list = self.text_cache
        else:
            vae_latents_list = None
            text_embeds_list = None
        vae_latents_list = broadcast_tensor_list(vae_latents_list, group_src=src_rank, group=self.tp_cp_group)
        text_embeds_list = broadcast_tensor_list(text_embeds_list, group_src=src_rank, group=self.tp_cp_group)
        return vae_latents_list, text_embeds_list

    def __call__(self, vae_latents_list=None, text_embeds_list=None, step=0):
        if self.tp_cp_size <= 1:
            return vae_latents_list, text_embeds_list
        if step % self.tp_cp_size == 0:
            self.save_cache(vae_latents_list, text_embeds_list)
        return self.get_cache_from_rank(src_rank=step % self.tp_cp_size)


if __name__ == "__main__":
    from ultrai2v.distributed.utils import setup_distributed_env, cleanup_distributed_env
    from torch.distributed.device_mesh import init_device_mesh
    setup_distributed_env()
    mesh = init_device_mesh("cuda", [2, 4], mesh_dim_names=["dp", "cp"])
    manager = EncoderCacheManager(tp_cp_group=mesh["cp"].get_group())
    rank = dist.get_rank()
    vae_data, text_data = [torch.tensor([range(rank, rank + 5)], device="cuda")], [torch.tensor([range(rank + 5, rank + 10)], device="cuda")]
    print(f"Rank {rank} original data: vae {vae_data}, text {text_data}")
    for step in range(1):
        vae_data, text_data = manager(vae_data, text_data, step)
        print(f"Rank {rank} step {step} received data: vae {vae_data}, text {text_data}")
        print(f"data device: vae {vae_data[0].device}, text {text_data[0].device}")
    cleanup_distributed_env()