
import sys
import os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(root)
import json
import torch
import random
import subprocess
import numpy as np
import torch.distributed as dist
import pandas as pd
import argparse
import torch
import os
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
from qwen_vl_utils import process_vision_info
from torchvision import transforms
from transformers import AutoProcessor
from transformers import SiglipImageProcessor, SiglipVisionModel
from univa.utils.flux_pipeline import FluxPipeline
from univa.eval.configuration_eval import EvalConfig
from univa.utils.get_ocr import get_ocr_result
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration

# adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/random.py#L31
def set_seed(seed, rank, device_specific=True):
    if device_specific:
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_models(args, device):

    # Load main model and task head
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        args.pretrained_lvlm_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        args.pretrained_lvlm_name_or_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    # Load FLUX pipeline
    pipe = FluxPipeline.from_pretrained(
        args.pretrained_denoiser_name_or_path,
        transformer=model.denoise_tower.denoiser,
        torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]

    siglip_processor = SiglipImageProcessor.from_pretrained(args.pretrained_siglip_name_or_path)
    siglip_model = SiglipVisionModel.from_pretrained(
        args.pretrained_siglip_name_or_path,
        torch_dtype=torch.bfloat16,
    ).to(device)

    return {
        'model': model,
        'processor': processor,
        'pipe': pipe,
        'tokenizers': tokenizers,
        'text_encoders': text_encoders,
        'device': device,
        'siglip_model': siglip_model,
        'siglip_processor': siglip_processor,
    }


def init_gpu_env(args):
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    args.local_rank = local_rank
    args.world_size = world_size
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl', init_method='env://', 
        world_size=world_size, rank=local_rank
        )
    return args


def run_model_and_return_samples(args, state, text, image1=None, image2=None):
    
    # Build content
    convo = []
    image_paths = []
    content = []
    for img in (image1, image2):
        if img:
            content.append({'type':'image','image':img,'min_pixels':args.min_pixels,'max_pixels':args.max_pixels})
            image_paths.append(img)
    if text:
        ocr_text = ''
        if args.ocr_enhancer and content:
            ocr_texts = []
            for img in (image1, image2):
                if img:
                    ocr_texts.append(get_ocr_result(img, cur_ocr_i))
                    cur_ocr_i += 1
            ocr_text = '\n'.join(ocr_texts)
        content.append({'type':'text','text': text + ocr_text})

    if not args.only_use_t5:
        convo.append({'role':'user','content':content})

        # Prepare inputs
        chat_text = state['processor'].apply_chat_template(
            convo,
            tokenize=False, 
            add_generation_prompt=True
            )
        chat_text = '<|im_end|>\n'.join(chat_text.split('<|im_end|>\n')[1:])
        image_inputs, video_inputs = process_vision_info(convo)
        inputs = state['processor'](
            text=[chat_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors='pt'
        ).to(state['device'])

        # Generate
        # image generation pipeline
        siglip_hs = None
        if state['siglip_processor'] and image_paths:
            vals = [state['siglip_processor'].preprocess(
                        images=Image.open(p).convert('RGB'), do_resize=True,
                        return_tensors='pt', do_convert_rgb=True
                    ).pixel_values.to(state['device'])
                    for p in image_paths]
            siglip_hs = state['siglip_model'](torch.concat(vals)).last_hidden_state

        with torch.no_grad():
            lvlm = state['model'](
                inputs.input_ids, pixel_values=getattr(inputs,'pixel_values',None),
                attention_mask=inputs.attention_mask,
                image_grid_thw=getattr(inputs,'image_grid_thw',None),
                siglip_hidden_states=siglip_hs,
                output_type='denoise_embeds'
            )
            prm_embeds, pooled = encode_prompt(
                state['text_encoders'], state['tokenizers'],
                text if args.joint_with_t5 else '', 512, state['device'], 1
            )
        emb = torch.concat([lvlm, prm_embeds], dim=1) if args.joint_with_t5 else lvlm
    else:
        prm_embeds, pooled = encode_prompt(
            state['text_encoders'], state['tokenizers'],
            text, 512, state['device'], 1
        )
        emb = prm_embeds

    with torch.no_grad():
        img = state['pipe'](
            prompt_embeds=emb, 
            pooled_prompt_embeds=pooled,
            height=args.height, 
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt, 
        ).images
    return img
    

def main(args):

    args = init_gpu_env(args)

    torch.backends.cuda.matmul.allow_tf32 = False 
    torch.backends.cudnn.allow_tf32 = False
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed, rank=args.local_rank, device_specific=True)
    device = torch.cuda.current_device()
    state = initialize_models(args, device)

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the evaluation prompts
    with open(args.geneval_prompt_path, "r") as f:
        metadatas = [json.loads(line) for line in f]

    inference_list = []
    
    for index, metadata in enumerate(metadatas):
        outpath = os.path.join(args.output_dir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata["prompt"]
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)
        all_samples = list()
        
        for idx, n in enumerate(range(args.n_samples)):
            inference_list.append([prompt, sample_path, idx])
            
    inference_list = inference_list[args.local_rank::args.world_size]
    for prompt, sample_path, sample_count in tqdm(inference_list):
        if os.path.exists(os.path.join(sample_path, f"{sample_count:05}.png")):
            continue
        image = run_model_and_return_samples(args, state, prompt, image1=None, image2=None)
        image = image[0]
        image = image.resize((args.resized_width, args.resized_height))
        # Save image
        image.save(
            os.path.join(sample_path, f"{sample_count:05}.png")
        )


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--pretrained_lvlm_name_or_path", type=str, default=None, required=False)
    parser.add_argument("--output_dir", type=str, default=None, required=False)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    schema = OmegaConf.structured(EvalConfig)
    conf = OmegaConf.merge(schema, config)
    if args.pretrained_lvlm_name_or_path is not None:
        assert args.output_dir is not None
        conf.pretrained_lvlm_name_or_path = args.pretrained_lvlm_name_or_path
        conf.output_dir = args.output_dir
    main(conf)