import gradio as gr
import json
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage

import transformers
import copy
import torch
import concurrent.futures

# 常量定义
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 1
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_GEN_IMAGE_TOKEN = "<gen_image>"
DEFAULT_IMAGE_TOKEN = "<image>"

def preprocess_qwen_chatml(
        sources, 
        tokenizer: transformers.PreTrainedTokenizer, 
        system_message: str = "You are a helpful assistant.", 
        ):
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    # im_start, im_end = tokenizer.additional_special_tokens_ids
    im_start, im_end = tokenizer("<|im_start|>").input_ids[0], tokenizer("<|im_end|>").input_ids[0]
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    # print(sources)
    for i, source in enumerate(sources):
        # print(source[0])
        # print(source[0]["from"])
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
                
        # import ipdb;ipdb.set_trace()
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )

def preprocess_multimodal(
    sources,
):
    is_multimodal = True
    if not is_multimodal:
        return sources
    
    is_gen_task = False
    for source in sources:
        len_source = len(source)
        for idx, sentence in enumerate(source):
            # DEFAULT_GEN_IMAGE_TOKEN must be the last image, and will be transform to DEFAULT_IMAGE_TOKEN
            if DEFAULT_GEN_IMAGE_TOKEN in sentence["value"]:
                assert idx + 1 == len_source
                assert sentence['value'].count(DEFAULT_GEN_IMAGE_TOKEN) == 1
                sentence["value"] = sentence["value"].replace(DEFAULT_GEN_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN)
                is_gen_task = True

            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                # sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                # sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

            replace_token = DEFAULT_IMAGE_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources

# —— Gradio 相关函数 —— 
data = []
img_root = ""

def load_json(json_path, image_root):
    global data, img_root
    img_root = image_root.strip()
    try:
        with open(json_path.strip(), 'r', encoding='utf-8') as f:
            data = json.load(f)
        return f"Loaded successfully {len(data)} raw data."
    except Exception as e:
        return f"Error loading JSON file：{e}"

# def check_image_tags(progress=gr.Progress()):
#     global data
#     checked, skipped = [], 0
#     for sample in progress.tqdm(data, desc="检查中"):
#         img_f = sample.get("image", None)
#         conv = sample.get("conversations", [])
#         cnt = sum(turn["value"].count("<image>") + turn["value"].count("<gen_image>") for turn in conv)
#         valid = False
#         if img_f is None:
#             valid = (cnt == 0)
#         elif isinstance(img_f, str):
#             valid = (cnt == 1)
#         elif isinstance(img_f, list):
#             valid = (len(img_f) == cnt)
#         if valid:
#             checked.append(sample)
#         else:
#             skipped += 1
#     data = checked
#     return f"检查完成。有效样本：{len(data)}，跳过：{skipped}"

def check_image_tags(min_images=0, progress=gr.Progress()):
    global data
    if len(data) == 0:
        return "Please enter the JSON file path and click Load."
    checked, skipped = [], 0
    for sample in progress.tqdm(data, desc="Checking"):
        img_f = sample.get("image", None)
        conv = sample.get("conversations", [])
        # 计算该样本中对话里所有出现的 "<image>" 和 "<gen_image>" 的总数
        cnt = sum(turn["value"].count("<image>") + turn["value"].count("<gen_image>") for turn in conv)
        
        # 判断是否满足最少图片数量的要求
        if cnt < min_images:
            skipped += 1
            continue

        # 判断 image 字段与对话中图片符号数量是否匹配
        valid = False
        if img_f is None:
            valid = (cnt == 0)
        elif isinstance(img_f, str):
            valid = (cnt == 1)
        elif isinstance(img_f, list):
            valid = (len(img_f) == cnt)
            
        if valid:
            checked.append(sample)
        else:
            skipped += 1
    exist_pct = (len(checked) / len(data) * 100) if len(data) > 0 else 0.0
    if skipped == 0:
        return (f"✅ Total image path: {len(data)}，"
                f"Ratio: {exist_pct:.2f}%")
    else:
        return (f"❌ Total image path: {len(data)}，"
                f"Success: {len(checked)}，"
                f"Error: {skipped}，"
                f"Ratio: {exist_pct:.2f}%")


def show_random_sample():
    global data
    if len(data) == 0:
        return "Please enter the JSON file path and click Load."
    if len(img_root) == 0:
        return "Please enter the root directory of the image and click Load."
    sample = random.choice(data)
    img_f = sample.get("image", [])
    imgs = [img_f] if isinstance(img_f, str) else (img_f or [])
    fulls = [os.path.join(img_root, p) for p in imgs if os.path.exists(os.path.join(img_root, p))]
    text = ""
    for turn in sample.get("conversations", []):
        sp = "🧑 User: " if turn["from"]=="human" else "🤖 AI: "
        text += f"{sp}{turn['value'].strip()}\n\n"
    return fulls, text

def count_image_distribution_with_plot(progress=gr.Progress()):
    global data
    if len(data) == 0:
        return "Please enter the JSON file path and click Load."
    stats = {"nlp data":0,"1 <image>":0,"2 <image>":0,"more than 2":0}
    for sample in progress.tqdm(data, desc="Checking"):
        img_f = sample.get("image", None)
        if img_f is None:
            stats["nlp data"] += 1
        elif isinstance(img_f, str):
            stats["1 <image>"] += 1
        else:
            L = len(img_f)
            if L==1: stats["1 <image>"] += 1
            elif L==2: stats["2 <image>"] += 1
            else:      stats["more than 2"] += 1
    total = sum(stats.values())
    props = [v/total for v in stats.values()]
    labels = list(stats.keys())
    plt.figure(figsize=(8,6))
    plt.bar(labels, props, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    plt.ylabel('Ratio')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return PILImage.open(buf)

# —— 多进程验证相关 —— 
_tokenizer_global = None

def _init_worker(model_name):
    global _tokenizer_global
    _tokenizer_global = transformers.AutoTokenizer.from_pretrained(model_name)
    _tokenizer_global.add_tokens(["<image>"], special_tokens=True)

def _validate_sample(sample):
    # 使用全局 _tokenizer_global
    sample = [sample]

    # print(copy.deepcopy([e["conversations"] for e in sample]))
    sources = preprocess_multimodal(
        copy.deepcopy([e["conversations"] for e in sample])
        )
    preprocess_qwen_chatml(sources, _tokenizer_global)
    return True

def validate_format(model_name, progress=gr.Progress()):
    global data
    if len(data) == 0:
        return "Please enter the JSON file path and click Load."
    if len(img_root) == 0:
        return "Please enter the root directory of the image and click Load."
    # _init_worker(model_name)
    # for sample in data:
    #     _validate_sample(sample) 
    try:
        total = len(data)
        # 使用与 CPU 核数相同的进程数
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=os.cpu_count(), 
                # max_workers=1, 
                initializer=_init_worker, 
                initargs=(model_name,)
            ) as executor:
            futures = [executor.submit(_validate_sample, sample) for sample in data]
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                progress((i+1)/total, desc="Checking")
                if fut.exception():
                    # 发现错误，取消剩余
                    executor.shutdown(cancel_futures=True)
                    raise fut.exception()
            return "✅ Data format valid!"
    except Exception as e:
        return f"❌ Invalid data format: {e}"


def _check_paths_sample(sample):
    total_paths = 0
    exist_count = 0
    img_f = sample.get("image", None)
    if isinstance(img_f, str):
        paths = [img_f]
    elif isinstance(img_f, list):
        paths = img_f
    else:
        return 0, 0
    for p in paths:
        total_paths += 1
        full = os.path.join(img_root, p)
        if os.path.exists(full):
            exist_count += 1
    return total_paths, exist_count

def check_image_paths(progress=gr.Progress()):
    global data
    total_paths = 0
    exist_count = 0
    total_samples = len(data)
    if len(data) == 0:
        return "Please enter the JSON file path and click Load."
    if len(img_root) == 0:
        return "Please enter the root directory of the image and click Load."
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(_check_paths_sample, sample) for sample in data]
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            progress((i+1) / total_samples, desc="Checking")
            # try:
            sample_total, sample_exist = fut.result()
            total_paths += sample_total
            exist_count += sample_exist
            # except Exception as e:
            #     return str(e)
    missing_count = total_paths - exist_count
    exist_pct = (exist_count / total_paths * 100) if total_paths > 0 else 0.0
    if exist_pct == 100.0:
        return (f"✅ Total image path: {total_paths}，"
                f"Ratio: {exist_pct:.2f}%")
    else:
        return (f"❌ Total image path: {total_paths}，"
                f"Found: {exist_count}，"
                f"Not Found: {missing_count}，"
                f"Ratio: {exist_pct:.2f}%")


# —— Gradio 界面搭建 —— 
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 UniWorld Data Verification Tool")

    with gr.Row():
        json_path      = gr.Textbox(label="JSON file path")
        image_root     = gr.Textbox(label="Image root directory")
        load_btn       = gr.Button("Load JSON (click here)")
    load_status      = gr.Textbox(label="Loading status", interactive=False)

    with gr.Row():
        check_btn      = gr.Button("🔍 Check the <image> tag (click here)")
        min_images_input = gr.Number(label="Minimum number of images", value=0, precision=0)
        check_status   = gr.Textbox(label="<image> check results", interactive=False)
        
    with gr.Row():
        check_paths_btn = gr.Button("🔍 Check image path (click here)")
        check_paths_status = gr.Textbox(label="Path check results", interactive=False)

    with gr.Row():
        validate_btn   = gr.Button("🔍 Verify data format (click here)")
        tokenizer_name = gr.Textbox(label="Tokenizer HF name or absolute path", value="/mnt/data/checkpoints/Qwen/Qwen2.5-3B-Instruct")
        validate_status= gr.Textbox(label="Verification results", interactive=False)

    count_btn        = gr.Button("📊 Image quantity distribution (click here)")
    count_plot       = gr.Image(type="pil", label="Bar chart showing the distribution of image quantities")

    gallery          = gr.Gallery(label="Image preview", columns=4)
    text_box         = gr.Textbox(label="Conversation content", lines=10, interactive=False)
    random_btn       = gr.Button("Randomly view samples (click here)")

    # 事件绑定
    load_btn.click(load_json, inputs=[json_path, image_root], outputs=load_status)
    check_btn.click(check_image_tags, inputs=min_images_input, outputs=check_status)
    check_paths_btn.click(check_image_paths, outputs=check_paths_status)
    validate_btn.click(validate_format, inputs=tokenizer_name, outputs=validate_status)
    count_btn.click(count_image_distribution_with_plot, outputs=count_plot)
    random_btn.click(show_random_sample, outputs=[gallery, text_box])

# server_port = 7888
demo.launch(
    # server_port=server_port, 
    allowed_paths=['/']
)
