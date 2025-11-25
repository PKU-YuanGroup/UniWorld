import torch
from ultrai2v.utils.utils import is_npu_available, check_and_import_npu
check_and_import_npu()
import logging
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)

def FSDP2_mix_wrapper(
    model,
    dp_mesh=None,
    weight_dtype=torch.bfloat16,
    main_block_to_half=None,
    blocks_to_float=None,
    blocks_to_output_float=None,
    reshard_after_forward=None,
    cpu_offload=False,
):
    is_rank_zero = torch.distributed.get_rank() == 0
    if is_rank_zero:
        logging.info("Parallelize Module with FSDP2...")
    low_precision_policy = MixedPrecisionPolicy(
        param_dtype=weight_dtype,
        reduce_dtype=torch.float32,
        output_dtype=weight_dtype,
    )
    high_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=weight_dtype,
    )
    fp32_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": dp_mesh,
    }  # dp_mesh is None means distributed to all nodes.

    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    if blocks_to_output_float is not None and len(blocks_to_output_float) > 0:
        for module in model.modules():
            for block in blocks_to_output_float:
                if isinstance(module, block):
                    if is_rank_zero:
                        logging.info(f"FSDP {block} Module with All Float Precision to Output Float Results.")
                    fully_shard(module, mp_policy=fp32_precision_policy, **fsdp_kwargs)

    if blocks_to_float is not None and len(blocks_to_float) > 0:
        for module in model.modules():
            for block in blocks_to_float:
                if isinstance(module, block):
                    if is_rank_zero:
                        logging.info(f"FSDP {block} Module with High Precision.")
                    fully_shard(module, mp_policy=high_precision_policy, **fsdp_kwargs)

    if main_block_to_half is not None:
        for module in model.modules():
            if isinstance(module, main_block_to_half):
                if is_rank_zero:
                    logging.info(f"FSDP {main_block_to_half} Module with Low Precision.")
                fully_shard(module, mp_policy=low_precision_policy, **fsdp_kwargs)

    if is_rank_zero:
        logging.info(f"FSDP Other Modules.")
    fully_shard(model, mp_policy=low_precision_policy, **fsdp_kwargs)

    if is_rank_zero:
        logging.info("FSDP Down!")
        logging.info(f"Model Overview: \n{model}")


def FSDP2_fp32_wrapper(
    model,
    dp_mesh=None,
    main_block=None,
    reshard_after_forward=None,
    cpu_offload=False,
):
    fp32_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": dp_mesh,
    }  # dp_mesh is None means distributed to all nodes.

    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    if main_block is not None:
        for module in model.modules():
            if isinstance(module, main_block):
                fully_shard(module, mp_policy=fp32_precision_policy, **fsdp_kwargs)
    fully_shard(model, mp_policy=fp32_precision_policy, **fsdp_kwargs)

if __name__ == "__main__":
    from torch.distributed.device_mesh import init_device_mesh
    from ultrai2v.modules import models, models_blocks_to_float, models_main_block
    from ultrai2v.distributed.utils import setup_distributed_env, cleanup_distributed_env, gather_data_from_all_ranks
    from ultrai2v.utils.random_utils import set_seed
    
    setup_distributed_env()

    ddp_fsdp_mesh = init_device_mesh(
        "cuda",
        (4, 1),
        mesh_dim_names=("ddp", "fsdp"),
    )
    print("ddp_fsdp_mesh:", ddp_fsdp_mesh)
    if not is_npu_available():
        pretrained_model_dir = "/mnt/data2/Wan2.1-T2V-1.3B/"
    else:
        pretrained_model_dir = "/work/share1/checkpoint/Wan-AI/Wan2.1-T2V-1.3B/"

    model_name = "flashi2v"
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    dtype = torch.float32

    set_seed(1024, device_specific=False)
    latents = torch.randn(1, 16, 2, 16, 16).to(device=device, dtype=dtype)
    start_frame_latents = torch.randn(1, 16, 2, 16, 16).to(device=device, dtype=dtype)
    fourier_features = torch.randn(1, 16, 2, 16, 16).to(device=device, dtype=dtype)
    text_embeddings = torch.randn(1, 512, 4096).to(device=device, dtype=dtype)
    timesteps = torch.randint(0, 1000, (1,)).to(device=device)

    # ddp_model = wan_model[model_name].from_pretrained(pretrained_model_dir)
    set_seed(1024, device_specific=True)
    ddp_model = models[model_name]()
    with torch.device("cpu"):
        fsdp_model = models[model_name]()
        ddp_fsdp_model = models[model_name]()
    ddp_model = torch.nn.parallel.DistributedDataParallel(ddp_model.to(device=device, dtype=dtype))

    # fsdp_model = wan_model[model_name].from_pretrained(pretrained_model_dir)
    # ddp_fsdp_model = wan_model[model_name].from_pretrained(pretrained_model_dir)

    # set_seed(1024, device_specific=False)
    # fsdp_model.reset_parameters()
    # set_seed(1024, device_specific=False)
    # ddp_fsdp_model.reset_parameters()


    FSDP2_mix_wrapper(
        ddp_fsdp_model,
        dp_mesh=ddp_fsdp_mesh,
        weight_dtype=dtype,
        main_block_to_half=models_main_block[model_name],
        blocks_to_float=models_blocks_to_float[model_name],
        reshard_after_forward=True,
        cpu_offload=False,
    )

    FSDP2_mix_wrapper(
        fsdp_model,
        dp_mesh=None,
        weight_dtype=dtype,
        main_block_to_half=models_main_block[model_name],
        blocks_to_float=models_blocks_to_float[model_name],
        reshard_after_forward=True,
        cpu_offload=False,
    )

    # set_seed(1024, device_specific=True)
    ddp_model.module.reset_parameters()
    # set_seed(1024, device_specific=True)
    ddp_fsdp_model.to_empty(device=device)
    ddp_fsdp_model.reset_parameters()
    # set_seed(1024, device_specific=True)
    fsdp_model.to_empty(device=device)
    fsdp_model.reset_parameters()

    all_weights = gather_data_from_all_ranks(fsdp_model.patch_embedding.weight.data.to_local(), group=ddp_fsdp_mesh.get_group(0))
    if torch.equal(all_weights[0], all_weights[1]):
        print("fsdp_model True")
    else:
        print("fsdp_model False")

    all_weights = gather_data_from_all_ranks(ddp_fsdp_model.patch_embedding.weight.data.to_local(), group=ddp_fsdp_mesh.get_group(0))
    if torch.equal(all_weights[0], all_weights[1]):
        print("ddp_fsdp_model True")
    else:
        print("ddp_fsdp_model False")

    with torch.no_grad():
        fsdp_output = fsdp_model(latents, timesteps, text_embeddings, start_frame_latents=start_frame_latents, fourier_features=fourier_features)
        ddp_output = ddp_model(latents, timesteps, text_embeddings, start_frame_latents=start_frame_latents, fourier_features=fourier_features)
        ddp_fsdp_output = ddp_fsdp_model(latents, timesteps, text_embeddings, start_frame_latents=start_frame_latents, fourier_features=fourier_features)
    if torch.distributed.get_rank() < 2:
        print(f"rank = {torch.distributed.get_rank()}"
              f"ddp_output[0, 0, 0, 0, 0:10]: {ddp_output[0, 0, 0, 0, 0:10]}"
              f"fsdp_output[0, 0, 0, 0, 0:10]: {fsdp_output[0, 0, 0, 0, 0:10]}"
              f"ddp_fsdp_output[0, 0, 0, 0, 0:10]: {ddp_fsdp_output[0, 0, 0, 0, 0:10]}")
        print("fsdp_output - ddp_output MSE:", torch.mean((fsdp_output.float() - ddp_output.float()) ** 2))
        print("fsdp_output - ddp_fsdp_output MSE:", torch.mean((fsdp_output.float() - ddp_fsdp_output.float()) ** 2))
        print("ddp_output - ddp_fsdp_output MSE:", torch.mean((ddp_output.float() - ddp_fsdp_output.float()) ** 2))

    cleanup_distributed_env()