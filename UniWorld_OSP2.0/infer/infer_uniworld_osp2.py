import os
import yaml
from argparse import ArgumentParser

import torch
import torch.nn as nn
from ultrai2v.utils.utils import check_and_import_npu
check_and_import_npu()

from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer

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
from ultrai2v.utils.utils import str_to_precision, get_memory_allocated
from ultrai2v.utils.log_utils import get_logger, log_on_main_process
from ultrai2v.pipelines import pipelines
from ultrai2v.utils.infer_utils import load_prompts, load_images, save_videos, save_video_grid
from ultrai2v.utils.filter import HighFrequencyExtractor
from ultrai2v.utils.random_utils import set_seed


from transformers import Qwen2_5_VLTextModel,  AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention, Qwen2_5_VLDecoderLayer
import logging
from ultrai2v.modules.flashi2v import EncoderAdapter

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
    vlm_encoder_adapter_config = config.get("vlm_encoder_adapter_config", {})
    scheduler_config = config.get("scheduler_config", {})

    # inference config
    pipeline_name = config.get("pipeline_name", "t2v")
    weight_dtype = config.get("weight_dtype", "bfloat16")
    prompt_txt = config.get("prompt_txt", None)
    image_txt = config.get("image_txt", None)
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
        print(f'{output_dir} is created!!!')

    log_on_main_process(logger, "Initializing VAE model...")
    vae = WanVAE(
        vae_pth=vae_config.get("vae_path", None),
        dtype=str_to_precision(vae_config.get("dtype", "fp32")),
        device=device # for vae, we do not use fsdp
    )
    log_on_main_process(logger, f"VAE model initialized, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing text encoder model...")
    # tokenizer = AutoTokenizer.from_pretrained(text_encoder_config.get("text_tokenizer_path", None))
    # text_encoder = T5EncoderModel(
    #     text_len=text_encoder_config.get("text_len", 512),
    #     dtype=text_encoder_config.get("dtype", weight_dtype),
    #     device=device, # when no fsdp, we init the text_encoder on device
    #     checkpoint_path=text_encoder_config.get("checkpoint_path", None),
    #     use_fsdp=text_encoder_config.get("use_fsdp", False), # when using fsdp, we shard the text encoder by ddp_fsdp mesh
    #     device_mesh=ddp_fsdp_mesh if text_encoder_config.get("use_fsdp", False) else None,
    # )
    # log_on_main_process(logger, f"Text encoder model initialized, memory allocated: {get_memory_allocated()} GiB")
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    low_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )

    # QwenVL and adapter FSDP
    vlm_encoder = Qwen2_5_VLTextModel.from_pretrained(
        "/work/share/projects/gyy/zhubin_FlashI2V/checkpoints/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, 
    ) # device_map="auto"
    logging.info('use fsdp to shard vlm_encoder')
    for module in vlm_encoder.modules():
        if isinstance(module, Qwen2_5_VLDecoderLayer):
            fully_shard(module, mesh=ddp_fsdp_mesh)
    fully_shard(vlm_encoder, mesh=ddp_fsdp_mesh)
    vlm_encoder.requires_grad_(False)
    log_on_main_process(logger, f"VLM encoder model initialized, memory allocated: {get_memory_allocated()} GiB")

    # with torch.device('meta'):
    #     vlm_encoder_adapter = EncoderAdapter(
    #         in_dim=vlm_encoder.config.hidden_size, 
    #         hidden_dim=4096,
    #         out_dim=4096, # T5
    #     )
    # vlm_encoder_adapter = fully_shard(vlm_encoder_adapter, mp_policy=low_precision_policy, mesh=ddp_fsdp_mesh, reshard_after_forward=False) 
    # vlm_encoder_adapter.to_empty(device=device)
    # set_seed(seed, device_specific=False) # for init
    # vlm_encoder_adapter.reset_parameters()

    vlm_encoder_adapter = EncoderAdapter(
            in_dim=vlm_encoder.config.hidden_size, 
            hidden_dim=4096,
            out_dim=4096, # T5
        )
    log_on_main_process(logger, f"vlm_encoder_adapter initialized, memory allocated: {get_memory_allocated()} GiB")
    vlm_encoder_adapter_path = vlm_encoder_adapter_config.get('checkpoint_path',None)

    # import ipdb; ipdb.set_trace()
    print(f'{vlm_encoder_adapter_path=}!!!')
    state_dict = torch.load(vlm_encoder_adapter_path, map_location="cpu")
    m, u = vlm_encoder_adapter.load_state_dict(state_dict, strict=True)
    print(f"{vlm_encoder_adapter_path=}, missing keys: {len(m)}, unexpected keys: {len(u)}")
    vlm_encoder_adapter = fully_shard(vlm_encoder_adapter, mp_policy=low_precision_policy, mesh=ddp_fsdp_mesh, reshard_after_forward=False) 
    # vlm_encoder_adapter = vlm_encoder_adapter.to(device)
    # 如果 checkpointer.save_ema_model 保存的是完整的 checkpointer 字典，
    # 您可能需要解包 model state_dict。
    # 例如: state_dict = torch.load(CHECKPOINT_PATH)['model'] 
    # 请根据您 checkpointer.save_ema_model 的实际保存格式进行调整。

    # 载入模型的权重
    # state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
    # vlm_encoder_adapter.load_state_dict(state_dict)

    # 设置模型为评估模式 (通常在加载后进行推理或微调时需要)
    # vlm_encoder_adapter.eval()
    # 加载 vlm adapter权重


    
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

    pipeline = pipelines['flashi2v_qwenvl'](
        vae=vae,
        tokenizer=None,
        text_encoder=None,
        vlm_encoder=vlm_encoder,
        vlm_encoder_adapter=vlm_encoder_adapter,
        predictor=flashi2v_wrapper,
        scheduler=scheduler
    )

    prompts = load_prompts(prompt_txt)
    img_paths = load_prompts(image_txt)
    img_paths = [ os.path.join('/work/share/projects/gyy/zhubin_FlashI2V/', rela_path) for rela_path in img_paths]
    conditional_images = load_images(image_txt)

    assert len(prompts) == len(conditional_images), "In I2V mode, prompts num must be equal to conditional images num."

    set_seed(seed, device_specific=True, process_group=dp_group)

    dp_rank = torch.distributed.get_rank(dp_group)
    dp_size = torch.distributed.get_world_size(dp_group)
    if cp_mesh is not None:
        cp_rank = torch.distributed.get_rank(cp_mesh.get_group()) 
        cp_size = torch.distributed.get_world_size(cp_mesh.get_group())
    else:
        cp_rank = 0
        cp_size = 1

    if len(prompts) % dp_size > 0:
        log_on_main_process(logger, f"Warning! Caused by using FSDP, we will pad some dummy data to make sure len(prompts) {len(prompts)} == dp_size {dp_size}"
                                    f" and len(conditional_images) {len(conditional_images)} == dp_size {dp_size}.")
        while len(prompts) % dp_size > 0:
            prompts.append(prompts[0])
            conditional_images.append(conditional_images[0])
            img_paths.append(img_paths[0])

    video_grid = []
    for index in range(dp_rank * batch_size, len(prompts), batch_size * dp_size):
        batch_prompts = prompts[index: index + batch_size]
        batch_images = conditional_images[index: index + batch_size]
        batch_img_paths = img_paths[index: index + batch_size]
        print(f'{batch_prompts=},   {batch_img_paths=}!!')
        videos = pipeline(
            prompt=batch_prompts,
            conditional_image=batch_images,
            img_path=batch_img_paths,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            max_sequence_length=512,
            device=device
        )
        if cp_rank == 0:
            save_videos(videos, index, output_dir, save_fps)
            video_grid.append(videos)

    if len(video_grid) > 0:
        video_grid = torch.cat(video_grid, dim=0).to(device)

    if len(prompts) < batch_size * dp_size:
        active_ranks = range(len(prompts) // batch_size)
    else:
        active_ranks = range(dp_size)

    active_ranks = [x * cp_size for x in active_ranks]
    # torch.distributed.barrier()
    gathered_videos = gather_tensor_list_to_one([video_grid], group_dst=0, active_ranks=active_ranks)
    # torch.distributed.barrier()

    if rank == 0:
        video_grid = torch.cat(gathered_videos, dim=0)
        save_video_grid(video_grid, output_dir, fps=save_fps)
        print("Inference finished.")
        print(f"Saved {video_grid.shape[0]} samples to {output_dir}")

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
