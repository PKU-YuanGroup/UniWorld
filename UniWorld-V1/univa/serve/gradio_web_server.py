import gradio as gr
import sys
sys.path.append("..")
from transformers import AutoProcessor, SiglipImageProcessor, SiglipVisionModel, T5EncoderModel, BitsAndBytesConfig
from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration
from univa.utils.flux_pipeline import FluxPipeline
from univa.utils.get_ocr import get_ocr_result
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
from qwen_vl_utils import process_vision_info
from univa.utils.anyres_util import dynamic_resize, concat_images_adaptive
import torch
from torch import nn
import os
import uuid
import base64
from typing import Dict
from PIL import Image, ImageDraw, ImageFont

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Model and component paths")

    parser.add_argument("--model_path", type=str, default="LanguageBind/UniWorld-V1", help="UniWorld-V1模型路径")
    parser.add_argument("--flux_path", type=str, default="black-forest-labs/FLUX.1-dev", help="FLUX.1-dev模型路径")
    parser.add_argument("--siglip_path", type=str, default="google/siglip2-so400m-patch16-512", help="siglip2模型路径")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址")
    parser.add_argument("--server_port", type=int, default=6812, help="端口号")
    parser.add_argument("--share", action="store_true", help="是否公开分享")
    parser.add_argument("--nf4", action="store_true", help="是否NF4量化")

    return parser.parse_args()


def add_plain_text_watermark(
    img: Image.Image,
    text: str,
    margin: int = 50, 
    font_size: int = 30, 
):
    if img.mode != "RGB":
        img = img.convert("RGB")

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    bbox = draw.textbbox((0, 0), text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = img.width - text_width - int(3.3 * margin)
    y = img.height - text_height - margin

    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return img


css = """
.table-wrap table tr td:nth-child(3) > div {
    max-height: 150px;       /* 最多 100px 高度，按需修改 */
    overflow-y: auto;        /* 超出部分显示竖向滚动条 */
    white-space: pre-wrap;   /* 自动换行 */
    word-break: break-all;   /* 长单词内部分行 */
}

.table-wrap table tr td:nth-child(2) > div {
    max-width: 150px;
    white-space: pre-wrap;
    word-break: break-all;
    overflow-x: auto;
}
.table-wrap table tr th:nth-child(2) {
    max-width: 150px;
    white-space: normal;
    word-break: keep-all;
    overflow-x: auto;
}

.table-wrap table tr td:nth-last-child(-n+8) > div {
    max-width: 130px;
    white-space: pre-wrap;
    word-break: break-all;
    overflow-x: auto;
}
.table-wrap table tr th:nth-last-child(-n+8) {
    max-width: 130px;
    white-space: normal;
    word-break: keep-all;
    overflow-x: auto;
}
"""


def img2b64(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"
    return data_uri


def initialize_models(args):
    os.makedirs("tmp", exist_ok=True)
    # Paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    # Load main model and task head
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config if args.nf4 else None,
    ).to(device)
    task_head = nn.Sequential(
        nn.Linear(3584, 10240),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(10240, 2)
    ).to(device)
    task_head.load_state_dict(torch.load(os.path.join(args.model_path, 'task_head_final.pt')))
    task_head.eval()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=448*448,
        max_pixels=448*448,
    )
    if args.nf4:
        text_encoder_2 = T5EncoderModel.from_pretrained(
            args.flux_path,
            subfolder="text_encoder_2",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        pipe = FluxPipeline.from_pretrained(
            args.flux_path,
            transformer=model.denoise_tower.denoiser,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
        ).to(device)
    else:
        pipe = FluxPipeline.from_pretrained(
            args.flux_path,
            transformer=model.denoise_tower.denoiser,
            torch_dtype=torch.bfloat16,
        ).to(device)
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]

    # Optional SigLIP
    siglip_processor, siglip_model = None, None
    siglip_processor = SiglipImageProcessor.from_pretrained(args.siglip_path)
    siglip_model = SiglipVisionModel.from_pretrained(
        args.siglip_path,
        torch_dtype=torch.bfloat16,
    ).to(device)

    return {
        'model': model,
        'task_head': task_head,
        'processor': processor,
        'pipe': pipe,
        'tokenizers': tokenizers,
        'text_encoders': text_encoders,
        'siglip_processor': siglip_processor,
        'siglip_model': siglip_model,
        'device': device,
    }


args = parse_args()
state = initialize_models(args)


def process_large_image(raw_img):
    if raw_img is None:
        return raw_img
    img = Image.open(raw_img).convert("RGB")

    max_side = max(img.width, img.height)
    if max_side > 1024:
        scale = 1024 / max_side
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        print(f'resize img {img.size} to {(new_w, new_h)}')
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)
        save_path = f"tmp/{uuid.uuid4().hex}.png"
        img.save(save_path)
        return save_path
    else:
        return raw_img


def chat_step(image1, image2, text, height, width, steps, guidance,
              ocr_enhancer, joint_with_t5, enhance_generation, enhance_understanding,
              seed, num_imgs, history_state, progress=gr.Progress()):
    
    try:
        convo = history_state['conversation']
        image_paths = history_state['history_image_paths']
        cur_ocr_i = history_state['cur_ocr_i']
        cur_genimg_i = history_state['cur_genimg_i']

        # image1 = process_large_image(image1)
        # image2 = process_large_image(image2)
        # Build content
        content = []
        if text:
            ocr_text = ''
            if ocr_enhancer and content:
                ocr_texts = []
                for img in (image1, image2):
                    if img:
                        ocr_texts.append(get_ocr_result(img, cur_ocr_i))
                        cur_ocr_i += 1
                ocr_text = '\n'.join(ocr_texts)
            content.append({'type':'text','text': text + ocr_text})
        for img in (image1, image2):
            if img:
                content.append({'type':'image','image':img,'min_pixels':448*448,'max_pixels':448*448})
                image_paths.append(img)

        convo.append({'role':'user','content':content})

        # Prepare inputs
        chat_text = state['processor'].apply_chat_template(convo,
                        tokenize=False, add_generation_prompt=True)
        chat_text = '<|im_end|>\n'.join(chat_text.split('<|im_end|>\n')[1:])
        image_inputs, video_inputs = process_vision_info(convo)
        inputs = state['processor'](
            text=[chat_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors='pt'
        ).to(state['device'])

        # Model forward & task head
        with torch.no_grad():
            outputs = state['model'](**inputs, return_dict=True, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = inputs.input_ids == 77091
        vecs = hidden[mask][-1:]
        task_res = state['task_head'](vecs.float())[0]
        print(task_res)
        # Branch decision
        if enhance_generation:
            do_image = True
        elif enhance_understanding:
            do_image = False
        else:
            do_image = (task_res[0] < task_res[1])

        seed = int(seed)
        if seed == -1:
            seed = torch.Generator(device="cpu").seed()
        torch.manual_seed(seed)
        # Generate
        if do_image:
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
                    text if joint_with_t5 else '', 256, state['device'], 1
                )
            emb = torch.concat([lvlm, prm_embeds], dim=1) if joint_with_t5 else lvlm


            def diffusion_to_gradio_callback(_pipeline, step_idx: int, timestep: int, tensor_dict: Dict):
                # 1）更新 Gradio 进度条
                frac = (step_idx + 1) / float(steps)
                progress(frac)

                return tensor_dict

            with torch.no_grad():
                img = state['pipe'](
                    prompt_embeds=emb, pooled_prompt_embeds=pooled,
                    height=height, width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=torch.Generator(device='cuda').manual_seed(seed), 
                    num_images_per_prompt=num_imgs, 
                    callback_on_step_end=diffusion_to_gradio_callback,
                    # callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
                ).images
            # img = [add_plain_text_watermark(im, 'Open-Sora Plan 2.0 Generated') for im in img]
            img = concat_images_adaptive(img)
            save_path = f"tmp/{uuid.uuid4().hex}.png"
            img.save(save_path)
            convo.append({'role':'assistant','content':[{'type':'image','image':save_path}]})
            cur_genimg_i += 1
            progress(1.0)
            bot_msg = (None, save_path)
        else:
            # text generation
            gen_ids = state['model'].generate(**inputs, max_new_tokens=128)
            out = state['processor'].batch_decode(
                [g[len(inputs.input_ids[0]):] for g in gen_ids], skip_special_tokens=True
            )[0]
            convo.append({'role':'assistant','content':[{'type':'text','text':out}]})
            bot_msg = (None, out)

        
        chat_pairs = []
        # print(convo)
        # print()
        # print()
        for msg in convo:
            # print(msg)
            if msg['role']=='user':
                parts = []
                for c in msg['content']:
                    if c['type']=='text': parts.append(c['text'])
                    if c['type']=='image': parts.append(f"![user image]({img2b64(c['image'])})")
                chat_pairs.append(("\n".join(parts), None))
            else:
                parts = []
                for c in msg['content']:
                    if c['type']=='text': parts.append(c['text'])
                    if c['type']=='image': parts.append(f"![assistant image]({img2b64(c['image'])})")
                if msg['content'][-1]['type']=='text':
                    chat_pairs[-1] = (chat_pairs[-1][0], parts[-1])
                else:
                    chat_pairs[-1] = (chat_pairs[-1][0], parts[-1])
        # print()
        # print(chat_pairs)

        # Update state
        history_state.update({
            'conversation': convo,
            'history_image_paths': image_paths,
            'cur_ocr_i': cur_ocr_i,
            'cur_genimg_i': cur_genimg_i
        })
        return chat_pairs, history_state, seed
    except Exception as e:
        # 捕捉所有异常，返回错误提示，建议用户清理历史后重试
        error_msg = f"发生错误：{e}. 请点击 \"Clear History\" 清理对话历史后再试一次。"
        chat_pairs = [(None, error_msg)]
        # 不修改 history_state，让用户自行清理
        return chat_pairs, history_state, seed

def copy_seed_for_user(real_seed):
    # 这个函数会把隐藏的 seed_holder 值，传给真正要显示的 seed Textbox
    return real_seed

def clear_inputs():
    # img1 和 img2 用 None 来清空；text_in 用空字符串清空；seed 同理清空
    return None, None, "", ""

def clear_history():
    # 默认 prompt 和 seed
    default_prompt = "Translate this photo into a Studio Ghibli-style illustration, holding true to the original composition and movement."
    default_seed   = "-1"

    # 1. chatbot 要用 gr.update(value=[]) 清空
    # 2. state 直接给回初始 dict
    # 3. prompt 和 seed 同样用 gr.update()
    return (
        gr.update(value=[]),                             # 清空聊天框
        {'conversation':[],                              # 重置 state
         'history_image_paths':[],
         'cur_ocr_i':0,
         'cur_genimg_i':0},
        gr.update(value=None),                 # 重置 image1
        gr.update(value=None),                 # 重置 image2
        gr.update(value=default_prompt),                 # 重置 prompt 文本框
        gr.update(value=default_seed),                   # 重置 seed 文本框
    )


if __name__ == '__main__':
    # Gradio UI
    with gr.Blocks(
        theme=gr.themes.Soft(), 
        css=css
        ) as demo:

        gr.Markdown(
            """
            <div style="text-align:center;">

            # 🎉 UniWorld-V1 Chat Interface 🎉
            ### Unlock Cutting‑Edge Visual Perception, Feature Extraction, Editing, Synthesis, and Understanding

            **Usage Guide:**

            - It is recommended to perform inference on four images concurrently to offer varied selections.

            - Uploaded images are automatically resized; manually specifying resolutions that differ substantially from the original is not advised.
            </div>
            """,
            elem_classes="header-text",
        )
        with gr.Row():
            with gr.Column(1, min_width=0):
                gr.Markdown(
                    """
                    **🖼️ Visual Perception & Feature Extraction**  
                    - Canny Edge Detection  
                    - Mini-Line Segment Detection  
                    - Normal Map Generation
                    - Sketch Generation  
                    - Holistically-Nested Edge Detection  
                    - Depth Estimation  
                    - Human Pose Estimation
                    - Object Detection (Boxes)  
                    - Semantic Segmentation (Masks)
                    """
                )
            with gr.Column(1, min_width=0):
                gr.Markdown(
                    """
                    **✂️ Image Editing & Manipulation**  
                    - Add Elements  
                    - Adjust Attributes  
                    - Change Background  
                    - Remove Objects  
                    - Replace Regions  
                    - Perform Actions  
                    - Restyle  
                    - Compose Scenes
                    """
                )
            with gr.Column(1, min_width=0):
                gr.Markdown(
                    """
                    **🔄 Cross-Modal Synthesis & Transformation**  
                    - Text→Image Synthesis  
                    - Image‑to‑Image Translation  
                    - Multi‑Image Combination 
                    - Extract IP Features  
                    - IP Feature Composition
                    """ 
                )
            with gr.Column(1, min_width=0):
                gr.Markdown(
                    """
                    **🤖 Visual & Textual QA**  
                    - Image‑Text QA  
                    - Text‑Text QA
                    """
                )

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    img1 = gr.Image(type='filepath', label="Image 1", height=256, width=256)
                    img2 = gr.Image(type='filepath', label="Image 2 (Optional reference)", height=256, width=256, visible=True)
                text_in = gr.Textbox(label="Instruction", value="Translate this photo into a Studio Ghibli-style illustration, holding true to the original composition and movement.")
                seed = gr.Textbox(label="Seed (-1 for random)", value="-1")
                seed_holder = gr.Textbox(visible=False)
                with gr.Row():
                    num_imgs = gr.Slider(1, 4, 4, step=1, label="Num Images")
                
                with gr.Row():
                    height = gr.Slider(256, 2048, 1024, step=64, label="Height")
                    width = gr.Slider(256, 2048, 1024, step=64, label="Width")
                with gr.Row():
                    steps = gr.Slider(8, 50, 30, step=1, label="Inference steps")
                    guidance = gr.Slider(1.0, 10.0, 4.0, step=0.1, label="Guidance scale")
                with gr.Accordion("Advanced Options", open=False, visible=True):
                    with gr.Row():
                        enhance_gen_box = gr.Checkbox(value=False, label="Enhance Generation")
                        enhance_und_box = gr.Checkbox(value=False, label="Enhance Understanding")
                    with gr.Row():
                        ocr_box = gr.Checkbox(value=False, label="Enhance Text Rendering")
                        t5_box = gr.Checkbox(value=False, label="Enhance Current Turn")
            with gr.Column():
                chatbot = gr.Chatbot(
                    max_height=100000, min_height=700, 
                    height=None, 
                    resizable=True, 
                    show_copy_button=True
                    )
        anchor_pixels = 1024*1024
        # Dynamic resize callback
        def update_size(i1, i2):
            shapes = []
            for p in (i1, i2):
                if p:
                    im = Image.open(p)
                    w, h = im.size
                    shapes.append((w, h))
            if not shapes:
                return gr.update(), gr.update()
            if len(shapes) == 1:
                w, h = shapes[0]
            else:
                w = sum(s[0] for s in shapes) / len(shapes)
                h = sum(s[1] for s in shapes) / len(shapes)
            new_h, new_w = dynamic_resize(int(h), int(w), 'any_11ratio', anchor_pixels=anchor_pixels)
            return gr.update(value=new_h), gr.update(value=new_w)
        img1.change(fn=update_size, inputs=[img1, img2], outputs=[height, width])
        img2.change(fn=update_size, inputs=[img1, img2], outputs=[height, width])

        # Mutual exclusivity
        enhance_gen_box.change(
            lambda g: gr.update(value=False) if g else gr.update(),
            inputs=[enhance_gen_box], outputs=[enhance_und_box]
        )
        enhance_und_box.change(
            lambda u: gr.update(value=False) if u else gr.update(),
            inputs=[enhance_und_box], outputs=[enhance_gen_box]
        )
        state_ = gr.State({'conversation':[], 'history_image_paths':[], 'cur_ocr_i':0, 'cur_genimg_i':0})
        with gr.Row():
            submit = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear History", variant="primary")

        progress_bar = gr.Progress()
        click_event = submit.click(
            fn=chat_step,
            inputs=[img1, img2, text_in, height, width, steps, guidance,
                    ocr_box, t5_box, enhance_gen_box, enhance_und_box, seed, num_imgs, state_, 
                    ],
            outputs=[chatbot, state_, seed_holder], 
            scroll_to_output=True
        )
        click_event.then(
            fn=copy_seed_for_user,
            inputs=[seed_holder],    # 输入是隐藏的 seed_holder
            outputs=[seed]           # 输出到真正要显示的 seed Textbox
        )

        clear.click(
            fn=clear_history,
            inputs=[],
            outputs=[chatbot, state_, img1, img2, text_in, seed]
        )

        # ========== 添加 Validation Examples ==========
        example_height, example_width = 1024, 1024
        gr.Examples(
            examples_per_page=100, 
            examples=[
                # text-to-image
                [None, None,
                "Generate an adorable golden retriever puppy playing in a sunny park, "
                "with fluffy fur, big round eyes, and a happy expression. "
                "The background should have green grass, some flowers, and a blue sky with white clouds.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],


                # NIKE color swap
                ["assets/nike_src.jpg", None,
                "Switch the product's color from black, black to white, white, making sure the transition is crisp and clear.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                # style transfer (Ghibli)
                ["assets/gradio/origin.png", None,
                "Translate this photo into a Studio Ghibli-style illustration, holding true to the original composition and movement.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                ["assets/gradio/origin.png", None,
                "Remove the bicycle located in the lower center region of the image.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                # blur
                ["assets/gradio/blur.jpg", None,
                "Remove blur, make it clear.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                # 
                ["assets/gradio/00004614_tgt.jpg", None,
                "Add the ingrid fair isle cashmere turtleneck sweater to the person.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],
                # 
                ["assets/gradio/00006581_tgt.jpg", None,
                "Place the belvoir broderie anglaise linen tank on the person in a way that complements their appearance and style.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],
                # 
                ["assets/gradio/00008153_tgt.jpg", None,
                "Integrate may cashmere tank on body.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],
                # 
                ["assets/gradio/00002315_src.jpg", None,
                "Strip away all context and distractions, leaving the pointelle-trimmed cashmere t-shirt floating on a neutral background.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],
                # 
                ["assets/gradio/00002985_src.jpg", None,
                "Generate an image containing only the henry shearling jacket, free from any other visual elements.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                ["assets/gradio/origin.png", None,
                "Add a cat in the center of image.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                # image+image-to-image (compose)
                ["assets/00182555_target.jpg",
                "assets/00182555_InstantStyle_ref_1.jpg",
                "Adapt Image1's content to fit the aesthetic of Image2.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                # replace object
                ["assets/replace_src.png", None,
                "replace motorcycle located in the lower center region of the image with a black bicycle",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                # segmentation
                ["assets/seg_src.jpg", None,
                "Segment the giraffe from the background.\n",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                # detection
                ["assets/det_src.jpg", None,
                "Please depict the vase accurately",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                # image-to-canny
                ["assets/canny_image.jpg", None,
                "Generate a Canny edge map for this image.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                # image-to-mlsd
                ["assets/mlsd_image.jpg", None,
                "Render an MLSD detection overlay for this input image.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                # image-to-normal
                ["assets/normal_image.jpg", None,
                "Convert the input texture into a tangent-space normal map.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                # image-to-sketch
                ["assets/sketch_image.jpg", None,
                "Transform this image into a hand-drawn charcoal sketch.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                # image-to-hed
                ["assets/hed_image.jpg", None,
                "Produce a holistically-nested boundary probability map of this image.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

                # image-to-depth
                ["assets/depth_image.jpg", None,
                "Estimate depth with a focus on background structure.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],
                
                # image-to-image (reconstruction)
                ["assets/rec.jpg", None,
                "Simply reconstruct the original image with no enhancements.",
                example_height, example_width, 30, 4.0, False, False, False, False, "-1", 4],

            ],
            inputs=[img1, img2, text_in, height, width, steps, guidance,
                    ocr_box, t5_box, enhance_gen_box, enhance_und_box, seed, num_imgs],
        )
    # ==============================================

if __name__ == "__main__": 
    demo.launch(
        allowed_paths=["/"],
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        inbrowser=True,
    )


'''

MODEL_PATH="/mnt/data/lb/Remake/FlowWorld/checkpoints/flux_qwen2p5vl_7b_vlm_mlp_siglip_stage2_ts_1024_bs42x8x1_fa_any_11ratio_ema999_ocr_adamw_t5_0p4_lr1e-5_mask_refstyle_extract_resume_run3/checkpoint-12000/model_ema"
FLUX_PATH="/mnt/data/checkpoints/black-forest-labs/FLUX.1-dev"
SIGLIP_PATH="/mnt/data/checkpoints/google/siglip2-so400m-patch16-512"
CUDA_VISIBLE_DEVICES=2 python -m univa.serve.gradio_web_server \
    --model_path ${MODEL_PATH} \
    --flux_path ${FLUX_PATH} \
    --siglip_path ${SIGLIP_PATH}

'''
