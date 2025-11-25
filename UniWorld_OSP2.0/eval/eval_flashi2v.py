import os
import yaml
import math
from argparse import ArgumentParser

import torch
import torch.nn as nn
from ultrai2v.utils.utils import check_and_import_npu
check_and_import_npu()

from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer
from torchdata.stateful_dataloader import StatefulDataLoader

from ultrai2v.utils.constant import PROMPT, START_FRAME, NAME_INDEX
from ultrai2v.distributed.utils import setup_distributed_env, cleanup_distributed_env, gather_tensor_list_to_one
from ultrai2v.distributed.fsdp2_wrapper import FSDP2_mix_wrapper
from ultrai2v.distributed.cp_wrapper import CP_wrapper
from ultrai2v.modules import (
    WanVAE, 
    T5EncoderModel, 
    models, 
    models_main_block, 
    models_blocks_to_float,
    models_blocks_to_output_float,
    models_cp_plans,
)
from ultrai2v.schedulers import schedulers
from ultrai2v.distributed.checkpoint import Checkpointer
from ultrai2v.data import ultra_datasets, ultra_samplers, ultra_collators
from ultrai2v.utils.utils import str_to_precision, get_memory_allocated
from ultrai2v.utils.log_utils import get_logger, log_on_main_process
from ultrai2v.pipelines import pipelines
from ultrai2v.utils.infer_utils import save_videos, save_video_with_name
from ultrai2v.utils.filter import HighFrequencyExtractor
from ultrai2v.utils.random_utils import set_seed

class FlashI2VWrapper(nn.Module):
    def __init__(
        self, 
        model, 
        scheduler,
        low_freq_energy_ratio=[0.05, 0.95],
        fft_return_abs=True,
    ):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.high_freq_extractor = HighFrequencyExtractor(
            low_freq_energy_ratio=low_freq_energy_ratio,
            return_abs=fft_return_abs,
        )

    def forward(
        self,
        latents,
        timesteps,
        text_embeddings,
        start_frame_latents,
        fourier_features=None,
        start_frame_latents_proj=None,
        **kwargs,
    ):
        weight_dtype = text_embeddings.dtype
        
        if fourier_features is None or start_frame_latents_proj is None:
            fourier_features = self.high_freq_extractor(start_frame_latents.squeeze(2)).unsqueeze(2)
            fourier_features = fourier_features.repeat(1, 1, latents.shape[2], 1, 1)
            fourier_features = fourier_features.to(weight_dtype)

            start_frame_latents_proj = self.model.learnable_proj(start_frame_latents)
            assert start_frame_latents_proj.dtype == torch.float32

        latents = latents - start_frame_latents_proj

        with torch.autocast("cuda", dtype=weight_dtype):
            model_output = self.model(
                latents.to(weight_dtype),
                timesteps,
                text_embeddings,
                fourier_features=fourier_features,
            )

        return dict(model_output=model_output, fourier_features=fourier_features, start_frame_latents_proj=start_frame_latents_proj)


def main(config):
    logger = get_logger()

    # config analysis
    seed = config.get("seed", 42)

    # model config
    model_name = config.get("model_name", "wan_t2v")
    model_config = config.get("model_config", {})
    vae_config = config.get("vae_config", {})
    text_encoder_config = config.get("text_encoder_config", {})
    scheduler_config = config.get("scheduler_config", {})

    # data config
    data_config = config.get("data_config", {})

    # inference config
    pipeline_name = config.get("pipeline_name", "t2v")
    weight_dtype = config.get("weight_dtype", "bfloat16")
    batch_size = config.get("batch_size", 1)
    num_frames = config.get("num_frames", 49)
    height = config.get("height", 480)
    width = config.get("width", 832)
    save_fps = config.get("save_fps", 16)
    use_context_parallel = config.get("use_context_parallel", False)
    reshard_after_forward = config.get("reshard_after_forward", None)
    model_cpu_offload = config.get("model_cpu_offload", False)

    # save config
    output_dir = config.get("output_dir", "./output")
    save_with_dcp_api = config.get("save_with_dcp_api", False)

    # distributed setup
    setup_distributed_env()
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    weight_dtype = str_to_precision(weight_dtype)

    # init fsdp config
    fsdp_size = config.get("fsdp_size", 8)
    if fsdp_size > world_size: 
        fsdp_size = world_size
        log_on_main_process(logger, f"Warning, GPU nums are not enough! FSDP size reset to {fsdp_size}!")
    ddp_size = config.get("ddp_size", world_size // fsdp_size)
    ddp_fsdp_mesh = init_device_mesh("cuda", (ddp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp"))
    logger.info(f"rank {rank} use ddp mesh {ddp_fsdp_mesh['ddp']} and fsdp mesh {ddp_fsdp_mesh['fsdp']}")

    # init cp mesh if use context parallel
    dp_group = torch.distributed.group.WORLD
    cp_size = 1
    cp_mesh = None
    use_context_parallel = use_context_parallel and config.get("cp_size", 1) > 1
    if use_context_parallel:
        # cp size == model parallel (FSDP) size
        cp_size = config.get("cp_size", fsdp_size)
        if cp_size == fsdp_size:
            cp_mesh = ddp_fsdp_mesh["fsdp"]
            dp_group = ddp_fsdp_mesh["ddp"].get_group()
        # cp size != model parallel (FSDP) size
        else:
            dp_cp_mesh = init_device_mesh("cuda", (world_size // cp_size, cp_size), mesh_dim_names=("dp", "cp"))
            cp_mesh = dp_cp_mesh["cp"]
            dp_group = dp_cp_mesh["dp"].get_group()
        log_on_main_process(logger, f"We use context parallel, cp_size: {cp_size}")
        logger.info(f"rank {rank} use cp mesh {cp_mesh}")

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    log_on_main_process(logger, "Initializing VAE model...")
    vae = WanVAE(
        vae_pth=vae_config.get("vae_path", None),
        dtype=str_to_precision(vae_config.get("dtype", "fp32")),
        device=device # for vae, we do not use fsdp
    )
    log_on_main_process(logger, f"VAE model initialized, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing text encoder model...")
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_config.get("text_tokenizer_path", None))
    text_encoder = T5EncoderModel(
        text_len=text_encoder_config.get("text_len", 512),
        dtype=text_encoder_config.get("dtype", weight_dtype),
        device=device, # when no fsdp, we init the text_encoder on device
        checkpoint_path=text_encoder_config.get("checkpoint_path", None),
        use_fsdp=text_encoder_config.get("use_fsdp", False), # when using fsdp, we shard the text encoder by ddp_fsdp mesh
        device_mesh=ddp_fsdp_mesh if text_encoder_config.get("use_fsdp", False) else None,
    )
    log_on_main_process(logger, f"Text encoder model initialized, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing diffusion model and scheduler...")

    scheduler = schedulers[scheduler_config.pop("scheduler_name", "flow_matching")](**scheduler_config)

    has_loaded_pretrained_model = False
    pretrained_model_dir_or_checkpoint = model_config.get("pretrained_model_dir_or_checkpoint", None)
    if pretrained_model_dir_or_checkpoint is not None and os.path.isdir(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load model from pretrained_model_dir {pretrained_model_dir_or_checkpoint}")
        model = models[model_name].from_pretrained(pretrained_model_dir_or_checkpoint)
        has_loaded_pretrained_model = True
    elif pretrained_model_dir_or_checkpoint is not None and os.path.isfile(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Init model from scratch")
        with torch.device("meta"):
            model = models[model_name](**model_config)
    else:
        raise ValueError(f"In inference mode, pretrained_model_dir_or_checkpoint must be specified!")

    if use_context_parallel and model.num_heads % cp_size != 0:
        raise ValueError(f"When using context parallel, num_heads {model.num_heads} mush be mutiple of cp_size {cp_size}!")
    
    model.eval()
    flashi2v_wrapper = FlashI2VWrapper(
        model=model,
        scheduler=scheduler,
        low_freq_energy_ratio=model_config.get("low_freq_energy_ratio", 0.1),
        fft_return_abs=model_config.get("fft_return_abs", True)
    )

    # wrap model with cp wrapper if use context parallel
    if use_context_parallel:
        CP_wrapper(flashi2v_wrapper, models_cp_plans[model_name], cp_mesh=cp_mesh)

    # wrap model with fsdp2 mix-precision wrapper
    FSDP2_mix_wrapper(
        flashi2v_wrapper,
        dp_mesh=ddp_fsdp_mesh,
        weight_dtype=weight_dtype,
        main_block_to_half=models_main_block[model_name],
        blocks_to_float=models_blocks_to_float[model_name],
        blocks_to_output_float=models_blocks_to_output_float[model_name],
        reshard_after_forward=reshard_after_forward,
        cpu_offload=model_cpu_offload,
    )

    log_on_main_process(logger, f"Diffusion model initialized, memory allocated: {get_memory_allocated()} GiB")

    if not has_loaded_pretrained_model:
        model.to_empty(device=device)
        set_seed(seed, device_specific=False) # for init
        model.reset_parameters() # we should call reset_parameters because we init model at meta device 

    if pretrained_model_dir_or_checkpoint is not None and os.path.isfile(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load model from pretrained_model_checkpoint {pretrained_model_dir_or_checkpoint}")
        Checkpointer.load_model_from_path(flashi2v_wrapper.model, pretrained_model_dir_or_checkpoint, dcp_api=save_with_dcp_api)

    # dataset
    dataset = ultra_datasets[data_config.get("dataset_name", "t2v_random")](**data_config.get("dataset_config", {}))
    if use_context_parallel:
        video_shape = (
            dataset.sample_num_frames // (4 * model.patch_size[0]) + 1,
            dataset.sample_height // (8 * model.patch_size[1]),
            dataset.sample_width // (8 * model.patch_size[2]),
        )
        text_len = 512
        if math.prod(video_shape) % cp_size != 0 or text_len % cp_size != 0:
            raise ValueError(f"When using context parallel, sequence length {math.prod(video_shape)} must be multiple of cp_size {cp_size}!")
    
    # sampler
    dp_size = dp_group.size() 
    dp_rank = torch.distributed.get_rank(dp_group)
    sampler = ultra_samplers[data_config.get("sampler_name", "stateful_distributed")](
        dataset, 
        num_replicas=dp_size, # eval mode, we do not use encoder cache to be compatible with pipeline
        rank=dp_rank, # eval mode, we do not use encoder cache to be compatible with pipeline
        shuffle=True,
        # consumed_samples=consumed_samples,
        drop_last=False,
    )
    # dataloader
    num_workers = data_config.get("num_workers", 16)
    collator = ultra_collators[data_config.get("collator_name", "wan_t2v")](**data_config.get("collator_config", {}))
    assert batch_size == 1, f"in eval mode, batch_size should be set to 1, but current batch_size is {batch_size}"
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=data_config.get("pin_memory", False),
        generator=torch.Generator().manual_seed(seed + dp_rank)
    )

    pipeline = pipelines[pipeline_name](
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        predictor=flashi2v_wrapper,
        scheduler=scheduler
    )

    set_seed(seed, device_specific=True, process_group=dp_group)

    if cp_mesh is not None:
        cp_rank = torch.distributed.get_rank(cp_mesh.get_group()) 
        cp_size = torch.distributed.get_world_size(cp_mesh.get_group())
    else:
        cp_rank = 0
        cp_size = 1

    iteration_nums = len(dataloader)
    log_on_main_process(logger, f"we need to sample {iteration_nums} counts...")
    dataloader_iter = iter(dataloader)
    for _ in range(iteration_nums):
        batch = next(dataloader_iter)
        prompt = batch[PROMPT]
        start_frame = batch[START_FRAME]
        name_index = batch[NAME_INDEX]
        print(f'processing {name_index}...')
        videos = pipeline(
            prompt=prompt,
            conditional_image=start_frame,
            num_frames=num_frames,
            height=height,
            width=width,
            max_sequence_length=512,
            device=device
        )
        if cp_rank == 0:
            for video, name in zip(videos, name_index):
                save_video_with_name(video, name, output_dir, save_fps)

    cleanup_distributed_env()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/t2v.yaml")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise ValueError
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
