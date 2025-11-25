import os
import math
import yaml
from tqdm import tqdm
import wandb

from ultrai2v.utils.utils import check_and_import_npu
import torch
import torch.nn as nn
check_and_import_npu()

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from ultrai2v.data import ultra_datasets, ultra_samplers, ultra_collators
from ultrai2v.data.utils.utils import cyclic_iter
from ultrai2v.utils.log_utils import get_logger, log_on_main_process, verify_min_gpu_count
from ultrai2v.utils.random_utils import set_seed
from ultrai2v.utils.filter import HighFrequencyExtractor
from ultrai2v.distributed.utils import (
    setup_distributed_env, 
    cleanup_distributed_env, 
    set_modules_to_forward_prefetch, 
    set_modules_to_backward_prefetch, 
    gather_data_from_all_ranks
)
from ultrai2v.distributed.fsdp2_wrapper import FSDP2_mix_wrapper
from ultrai2v.distributed.fsdp_ema import FSDPEMAModel as EMAModel
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

from ultrai2v.distributed.checkpoint import Checkpointer, PREFIX as checkpoint_prefix
from ultrai2v.utils.constant import VIDEO,  START_FRAME, VLM_INPUTS # PROMPT_IDS, PROMPT_MASK,
from ultrai2v.utils.utils import str_to_precision, params_nums_to_str, get_memory_allocated
from ultrai2v.utils.clip_grads import AdaptiveGradClipper
from ultrai2v.utils.encoder_cache import EncoderCacheManager

# Qwenvl
from transformers import Qwen2_5_VLTextModel, AutoTokenizer, AutoProcessor
from ultrai2v.modules.flashi2v import EncoderAdapter
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention, Qwen2_5_VLDecoderLayer
import logging


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
        start_frame_latents,
        # text_embeddings,
        vlm_embeddings,
        weight_dtype,
    ):
        fourier_features = self.high_freq_extractor(start_frame_latents.squeeze(2)).unsqueeze(2)
        fourier_features = fourier_features.repeat(1, 1, latents.shape[2], 1, 1)
        fourier_features = fourier_features.to(weight_dtype)

        start_frame_latents = self.model.learnable_proj(start_frame_latents)
        assert start_frame_latents.dtype == torch.float32

        q_sample_results = self.scheduler.q_sample(latents, start_frame_latents=start_frame_latents)
        interpolated_latents = q_sample_results["x_t"].to(weight_dtype)
        prior_dist = q_sample_results["prior_dist"]
        sigmas = q_sample_results["sigmas"]
        timesteps = q_sample_results["timesteps"]
        with torch.autocast("cuda", dtype=weight_dtype):
            model_output = self.model(
                interpolated_latents,
                timesteps,
                # text_embeddings,
                vlm_embeddings,
                fourier_features=fourier_features,
            )

        return dict(model_output=model_output, prior_dist=prior_dist, latents=latents, sigmas=sigmas)

def main(config):
    logger = get_logger()

    # config analysis
    seed = config.get("seed", 42)

    # model config
    model_name = config.get("model_name", "wan_t2v")
    model_config = config.get("model_config", {})
    vae_config = config.get("vae_config", {})
    
    # text_encoder_config = config.get("text_encoder_config", {})
    vlm_encoder_config = config.get("vlm_encoder_config", {})
    scheduler_config = config.get("scheduler_config", {})

    # data config
    data_config = config.get("data_config", {})

    # optimizer config
    optimizer_config = config.get("optimizer_config", {})

    # training config
    training_iteration = config.get("training_iteration", 1000000)
    gradient_checkpointing = config.get("gradient_checkpointing", False)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    init_max_grad_norm = config.get("init_max_grad_norm", 1.0)
    log_interval = config.get("log_interval", 1)
    save_interval = config.get("save_interval", 1000)
    weight_dtype = config.get("weight_dtype", "bfloat16")
    reshard_after_forward = config.get("reshard_after_forward", None)
    model_cpu_offload = config.get("model_cpu_offload", False)
    ema_decay = config.get("ema_decay", 0.9999)
    ema_update_interval = config.get("ema_update_interval", 1)
    explicit_prefetching_num_blocks = config.get("explicit_prefetching_num_blocks", 0)
    use_context_parallel = config.get("use_context_parallel", False)
    deterministic_training = config.get("deterministic_training", False)

    # save config
    output_dir = config.get("output_dir", "./output")
    save_with_dcp_api = config.get("save_with_dcp_api", False)

    # distributed setup
    setup_distributed_env()
    verify_min_gpu_count()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    weight_dtype = str_to_precision(weight_dtype)

    # wandb config
    wandb_config = config.get("wandb_config", {})
    if wandb_config.get("project_name", None) is not None and rank == 0:
        project_name = wandb_config.get("project_name")
        wandb.init(
            project=project_name,
            name=wandb_config.get("exp_name", project_name),
            config=config,
            dir=output_dir
        )

    # init fsdp config
    fsdp_size = config.get("fsdp_size", 8)
    if fsdp_size > world_size: 
        fsdp_size = world_size
        log_on_main_process(logger, f"Warning, GPU nums are not enough! FSDP size reset to {fsdp_size}!")
    ddp_size = config.get("ddp_size", world_size // fsdp_size)
    ddp_fsdp_mesh = init_device_mesh("cuda", (ddp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp"))
    logger.info(f"rank {rank} use ddp mesh {ddp_fsdp_mesh['ddp']} and fsdp mesh {ddp_fsdp_mesh['fsdp']}")

    dp_group = dist.group.WORLD # use default world group
    # init cp mesh if use context parallel
    cp_size = 1
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

    if (save_interval * gradient_accumulation_steps) % cp_size != 0:
        raise ValueError(
            f"""because we use context parallel and encoder cache,
            save_interval * gradient_accumulation_steps ({save_interval} * {gradient_accumulation_steps} = {save_interval * gradient_accumulation_steps}) must be multiple of cp_size {cp_size}!
            """
        )

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    set_seed(seed, device_specific=False) # for init
    # init model
    log_on_main_process(logger, "Initializing VAE model...")
    vae = WanVAE(
        vae_pth=vae_config.get("vae_path", None),
        dtype=str_to_precision(vae_config.get("dtype", "fp32")),
        device=device # for vae, we do not use fsdp
    )
    log_on_main_process(logger, f"VAE model initialized, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing text encoder model...")
    # text_encoder = T5EncoderModel(
    #     text_len=text_encoder_config.get("text_len", 512),
    #     dtype=text_encoder_config.get("dtype", weight_dtype),
    #     device=device, # when no fsdp, we init the text_encoder on device
    #     checkpoint_path=text_encoder_config.get("checkpoint_path", None),
    #     use_fsdp=text_encoder_config.get("use_fsdp", False), # when using fsdp, we shard the text encoder by ddp_fsdp mesh
    #     device_mesh=ddp_fsdp_mesh if text_encoder_config.get("use_fsdp", False) else None,
    # )

    # QwenVL and adapter FSDP
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

    vlm_encoder = Qwen2_5_VLTextModel.from_pretrained(
        vlm_encoder_config.get("checkpoint_path", None), torch_dtype=torch.bfloat16, 
    ) # device_map="auto"
    logging.info('use fsdp to shard vlm_encoder')
    for module in vlm_encoder.modules():
        if isinstance(module, Qwen2_5_VLDecoderLayer):
            fully_shard(module, mesh=ddp_fsdp_mesh)
    fully_shard(vlm_encoder, mesh=ddp_fsdp_mesh)
    vlm_encoder.requires_grad_(False)

    log_on_main_process(logger, f"Text encoder model initialized, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, f"Initializing text encoder adapter...")

    low_precision_policy = MixedPrecisionPolicy(
        param_dtype=weight_dtype,
        reduce_dtype=torch.float32,
        output_dtype=weight_dtype,
    )

    with torch.device('meta'):
        vlm_encoder_adapter = EncoderAdapter(
            in_dim=vlm_encoder.config.hidden_size, 
            hidden_dim=4096,
            out_dim=4096, # T5
        )
    vlm_encoder_adapter = fully_shard(vlm_encoder_adapter, mp_policy=low_precision_policy, mesh=ddp_fsdp_mesh, reshard_after_forward=False) 
    vlm_encoder_adapter.to_empty(device=device)
    set_seed(seed, device_specific=False) # for init
    vlm_encoder_adapter.reset_parameters()
    log_on_main_process(logger, f"Text encoder adapter initialized, memory allocated: {get_memory_allocated()} GiB")

    # vae.to(device)
    # if not text_encoder_config.get("use_fsdp", False):
    #     text_encoder.to(device)
    # vae.requires_grad_(False) # in vae
    # text_encoder.requires_grad_(False) # in text_encoder

    log_on_main_process(logger, "Initializing diffusion model and scheduler...")

    scheduler = schedulers[scheduler_config.get("scheduler_name", "flow_matching")](**scheduler_config)

    pretrained_model_dir_or_checkpoint = model_config.get("pretrained_model_dir_or_checkpoint", None)
    has_loaded_pretrained_model = False
    if pretrained_model_dir_or_checkpoint is not None and os.path.isdir(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load model from pretrained_model_dir {pretrained_model_dir_or_checkpoint}")
        model = models[model_name].from_pretrained(pretrained_model_dir_or_checkpoint)
        has_loaded_pretrained_model = True
    else:
        log_on_main_process(logger, f"Init model from scratch")
        with torch.device("meta"):
            model = models[model_name](**model_config)

    if not has_loaded_pretrained_model:
        raise NotImplementedError(f"Training FlashI2V model must init with a pretrained t2v model, but pretrained_model_dir_or_checkpoint {pretrained_model_dir_or_checkpoint} does not exist!")

    if use_context_parallel and model.num_heads % cp_size != 0:
        raise ValueError(f"When using context parallel, num_heads {model.num_heads} mush be mutiple of cp_size {cp_size}!")


    flashi2v_wrapper = FlashI2VWrapper(
        model=model, 
        scheduler=scheduler, 
        low_freq_energy_ratio=model_config.get("low_freq_energy_ratio", [0.05, 0.95]),
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

    model = flashi2v_wrapper.model

    if not has_loaded_pretrained_model:
        model.to_empty(device=device)
        set_seed(seed, device_specific=False) # for init
        model.reset_parameters() # we should call reset_parameters because we init model at meta device 

    model.requires_grad_(False)

    log_on_main_process(logger, f"Diffusion model initialized, memory allocated: {get_memory_allocated()} GiB")
    if gradient_checkpointing:
        log_on_main_process(logger, "Use gradient checkpointing to save memory")
        model.set_gradient_checkpointing(True)

    # FSDP EMA model
    log_on_main_process(logger, "Initializing ema model...")
    ema_vlm_encoder_adapter = EMAModel(vlm_encoder_adapter, decay=ema_decay, update_interval=ema_update_interval)
    log_on_main_process(logger, f"EMA model initialized, memory allocated: {get_memory_allocated()} GiB")

    if explicit_prefetching_num_blocks > 0:
        set_modules_to_forward_prefetch(model.blocks, num_to_forward_prefetch=explicit_prefetching_num_blocks)
        set_modules_to_backward_prefetch(model.blocks, num_to_backward_prefetch=explicit_prefetching_num_blocks)

    checkpointer = Checkpointer(folder=output_dir, dcp_api=save_with_dcp_api)
    if checkpointer.last_training_iteration is not None:
        log_on_main_process(logger, "Loading model checkpoint...")
        checkpointer.load_model(vlm_encoder_adapter)
        log_on_main_process(logger, "Loading EMA model checkpoint...")
        ema_vlm_encoder_adapter.store(vlm_encoder_adapter)
        checkpointer.load_model(vlm_encoder_adapter, ema=True)
        ema_vlm_encoder_adapter.model_copy_to_ema(vlm_encoder_adapter)
        ema_vlm_encoder_adapter.restore(vlm_encoder_adapter)


    log_on_main_process(logger, "Initializing and loading optimizer checkpoint...")
    learning_rate = optimizer_config.get("lr", 1e-4)
    weight_decay = optimizer_config.get("weight_decay", 1e-2)
    optimizer = torch.optim.AdamW(
        vlm_encoder_adapter.parameters(),
        lr=learning_rate,
        betas=optimizer_config.get("betas", (0.9, 0.999)),
        weight_decay=weight_decay,
        eps=optimizer_config.get("eps", 1e-15),
    )
    log_on_main_process(logger, "Initializing adaptive gradient clipping...")
    adaptive_grad_clipper = AdaptiveGradClipper(init_max_grad_norm=init_max_grad_norm, model_parallel_group=ddp_fsdp_mesh["fsdp"].get_group())

    if checkpointer.last_training_iteration is not None:
        checkpointer.load_optim(vlm_encoder_adapter, optimizer)
        adaptive_grad_clipper.load(output_dir=f"{output_dir}/{checkpoint_prefix}{checkpointer.last_training_iteration:09d}")

    current_iteration = 0 if checkpointer.last_training_iteration is None else checkpointer.last_training_iteration
    current_batch_nums = current_iteration * gradient_accumulation_steps

    set_seed(seed, device_specific=True, process_group=dp_group, deterministic=deterministic_training) # for training
    
    log_on_main_process(logger, "Initializing dataset, sampler and dataloader...")
    # dataset
    dataset = ultra_datasets[data_config.get("dataset_name", "t2v_random")](**data_config.get("dataset_config", {}))
    if use_context_parallel:
        video_shape = (
            dataset.sample_num_frames // (4 * model.patch_size[0]) + 1,
            dataset.sample_height // (8 * model.patch_size[1]),
            dataset.sample_width // (8 * model.patch_size[2]),
        )
        text_len = dataset.text_max_length
        if math.prod(video_shape) % cp_size != 0 or text_len % cp_size != 0:
            raise ValueError(f"When using context parallel, sequence length {math.prod(video_shape)} must be multiple of cp_size {cp_size}!")
    
    # sampler
    batch_size = data_config.get("batch_size", 1)
    dp_size = dp_group.size() 
    sampler = ultra_samplers[data_config.get("sampler_name", "stateful_distributed")](
        dataset, 
        num_replicas=dist.get_world_size(), # we use encoder cache, so num_replicas = world_size
        rank=dist.get_rank(), # we use encoder cache, so rank in dp_group is same as global rank
        shuffle=data_config.get("shuffle", True),
        # consumed_samples=consumed_samples,
        drop_last=data_config.get("drop_last", True),
    )
    # dataloader
    num_workers = data_config.get("num_workers", 16)
    # num_workers = 0 # !!!!!!!!!!!!!!!!!!!!!!!!!!!
    collator = ultra_collators[data_config.get("collator_name", "wan_t2v")](**data_config.get("collator_config", {}))
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=data_config.get("pin_memory", False),
        generator=torch.Generator().manual_seed(seed + rank) # make sure all workers have different random patterns because we use encoder cache
    )

    if checkpointer.last_training_iteration is not None:
        log_on_main_process(logger, "Loading dataloader state...")
        checkpointer.load_dataloader_state_dict(dataloader)

    encoder_cache_manager = EncoderCacheManager(tp_cp_group=cp_mesh.get_group() if use_context_parallel else None)

    trainable_params_before_sharding = trainable_params_after_sharding = 0
    locked_params_before_sharding = locked_params_after_sharding = 0
    # for p in model.parameters():
    #     if p.requires_grad:
    #         trainable_params_before_sharding += p.numel()
    #         if isinstance(p, DTensor):
    #             trainable_params_after_sharding += p._local_tensor.numel()
    #         else:
    #             trainable_params_after_sharding += p.numel()
    #     else:
    #         locked_params_before_sharding += p.numel()
    #         if isinstance(p, DTensor):
    #             locked_params_after_sharding += p._local_tensor.numel()
    #         else:
    #             locked_params_after_sharding += p.numel()

    for p in vlm_encoder_adapter.parameters():
        if p.requires_grad:
            trainable_params_before_sharding += p.numel()
            if isinstance(p, DTensor):
                trainable_params_after_sharding += p._local_tensor.numel()
            else:
                trainable_params_after_sharding += p.numel()
        else:
            locked_params_before_sharding += p.numel()
            if isinstance(p, DTensor):
                locked_params_after_sharding += p._local_tensor.numel()
            else:
                locked_params_after_sharding += p.numel()


    start_training_logs = f"""
        {'=' * 20}Start Training{'=' * 20}
        Model: {model_name}
        Before FSDP sharding,
        Model has {params_nums_to_str(trainable_params_before_sharding)} trainable parameters and {params_nums_to_str(locked_params_before_sharding)} locked parameters
        After FSDP sharding,
        Model has {params_nums_to_str(trainable_params_after_sharding)} trainable parameters and {params_nums_to_str(locked_params_after_sharding)} locked parameters
        Scheduler: {scheduler_config.get("scheduler_name", "flow_matching")}
        Dataset: {data_config.get("dataset_name", "t2v_random")}
        Sampler: {data_config.get("sampler_name", "stateful_distributed")}
        Collator: {data_config.get("collator_name", "wan_t2v")}
        Use Context Parallel: {use_context_parallel}
        world_size: {world_size} GPUs
        dp_size: {dp_size} GPUs
        cp_size: {cp_size} GPUs
        Gradient checkpointing: {gradient_checkpointing}
        Weight dtype: {weight_dtype}
        Reshard after forward: {reshard_after_forward}
        Model CPU offload: {model_cpu_offload}
        EMA decay: {ema_decay} (update every {ema_update_interval} step)
        Random seed: {seed}
        Training iterations: {training_iteration}
        Current iteration: {current_iteration}
        Initial learning rate: {learning_rate}
        Weight decay: {weight_decay}
        Batch size per GPU: {batch_size}
        Gradient accumulation steps: {gradient_accumulation_steps}
        Effective batch size (dp_size x batch_size x gradient_accumulation_steps): {dp_size * batch_size * gradient_accumulation_steps}
        Consumed samples (current_iteration * batch_size * gradient_accumulation_steps * dp_size): {current_iteration * batch_size * gradient_accumulation_steps * dp_size}
        Save model to {output_dir} every {save_interval} iterations
        Training ...
        {'=' * 20}{'=' * len('Start Training')}{'=' * 20}
    """
    log_on_main_process(logger, start_training_logs)
    tqdm_bar = tqdm(total=training_iteration, disable=(rank != 0), initial=current_iteration, desc="Training")
    gathered_avg_loss = 0.0
    dataloader_iter = iter(cyclic_iter(dataloader)) # when one epoch is finished, this func will call __iter__ in sampler to produce next iter with different seed

    if checkpointer.last_training_iteration is not None:
        log_on_main_process(logger, "Loading rng state...")
        checkpointer.load_rng_state_dict()

    while current_iteration < training_iteration:
        if current_batch_nums % cp_size == 0:
            batch = next(dataloader_iter)
            video = batch.pop(VIDEO, None).to(dtype=torch.float32, device=device)
            # prompt_ids = batch.pop(PROMPT_IDS, None).to(device=device)
            # prompt_mask = batch.pop(PROMPT_MASK, None).to(device=device)
            vlm_inputs = batch.pop(VLM_INPUTS, None)
            vlm_inputs = { key:value.to(device=device)  for key, value in vlm_inputs.items()}

            start_frame = batch.pop(START_FRAME, None).to(dtype=torch.float32, device=device)
            with torch.no_grad():
                latents = vae.encode(video)
                start_frame_latents = vae.encode(start_frame)
                # text_embeddings = text_encoder(prompt_ids, prompt_mask)
                # 将T5的embeddings换成QwenVL理解特征
                vlm_embeddings = vlm_encoder(**vlm_inputs, return_dict=False, use_cache=False,)[0]
            vae_latents_list, text_embeds_list = encoder_cache_manager(
                vae_latents_list=[latents, start_frame_latents],
                # text_embeds_list=[text_embeddings],
                text_embeds_list=[vlm_embeddings],
                step=current_batch_nums
            ) # , text_embeds_list
        else:
            # pass
            vae_latents_list, text_embeds_list = encoder_cache_manager(step=current_batch_nums)
        latents = vae_latents_list[0]
        start_frame_latents = vae_latents_list[1]
        vlm_embeddings = text_embeds_list[0]
        
        vlm_embeddings_new = vlm_encoder_adapter(vlm_embeddings)
        # print(vlm_embeddings_new.shape,'!!!!!!!!!!')
        current_batch_nums += 1
        wrapper_output = flashi2v_wrapper(
            latents=latents,
            start_frame_latents=start_frame_latents,
            # text_embeddings=text_embeddings,
            vlm_embeddings = vlm_embeddings_new,
            weight_dtype=weight_dtype,
        )

        model_output = wrapper_output["model_output"]
        prior_dist = wrapper_output["prior_dist"]
        latents = wrapper_output["latents"]
        sigmas = wrapper_output["sigmas"]

        loss = scheduler.training_losses(model_output, latents, prior_dist)[0]
        loss = loss / gradient_accumulation_steps # default value of gradient_accumulation_steps is 1
        loss.backward()
        loss_for_log = loss.clone().detach().unsqueeze(0)
        gathered_loss = gather_data_from_all_ranks(loss_for_log, dim=0)
        gathered_avg_loss += gathered_loss.mean().item()
        if current_batch_nums % gradient_accumulation_steps == 0:
            current_iteration += 1
            grad_norm_after_clip = adaptive_grad_clipper.adaptive_clip(vlm_encoder_adapter.parameters())
            # grad_norm_after_clip = adaptive_grad_clipper.adaptive_clip(model.parameters())
            # grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), init_max_grad_norm, foreach=False)
            optimizer.step()
            optimizer.zero_grad()
            ema_vlm_encoder_adapter.update(vlm_encoder_adapter, current_iteration)

            if current_iteration % log_interval == 0:
                tqdm_bar.update(log_interval)
                tqdm_bar.set_postfix({"loss": gathered_avg_loss, "lr": optimizer.param_groups[0]['lr'], "grad_norm": grad_norm_after_clip.item()})
                if rank == 0 and wandb.run is not None:
                    wandb_logs = {
                        "train/loss": gathered_avg_loss,
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "train/grad_norm": grad_norm_after_clip.item(),
                    }
                    wandb_logs.update(adaptive_grad_clipper.state_dict())
                    wandb.log(wandb_logs, step=current_iteration)

            if current_iteration % save_interval == 0 or current_iteration == training_iteration:
                log_on_main_process(logger, f"Saving model checkpoint at iteration {current_iteration}...")
                checkpointer.save(vlm_encoder_adapter, optimizer, dataloader, current_iteration)
                ema_vlm_encoder_adapter.store(vlm_encoder_adapter)
                ema_vlm_encoder_adapter.ema_copy_to_model(vlm_encoder_adapter)
                checkpointer.save_ema_model(vlm_encoder_adapter, current_iteration)
                ema_vlm_encoder_adapter.restore(vlm_encoder_adapter)
                adaptive_grad_clipper.save(output_dir=f"{output_dir}/{checkpoint_prefix}{current_iteration:09d}")
            
            gathered_avg_loss = 0.0    

    end_training_logs = f"""
    {'=' * 20}End Training{'=' * 20}
    training iteration: {current_iteration}
    consumed samples: {current_batch_nums * batch_size}
    Model saved to {output_dir}
    Training finished.
    {'=' * 20}{'=' * len('End Training')}{'=' * 20}
    """
    log_on_main_process(logger, end_training_logs)
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
