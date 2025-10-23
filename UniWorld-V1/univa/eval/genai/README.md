
The original code is from [GenAI-Bench](https://github.com/linzhiqiu/t2v_metrics).


## Requirements and Installation

```
pip install git+https://github.com/openai/CLIP.git
pip install open-clip-torch
```


## Eval

### Generate samples

We also support `genai1600`, just replace`genai527.yaml` with `genai1600.yaml` and change `$OUTPUT_DIR`.

```bash
# switch to univa env
MODEL_PATH='path/to/model'
OUTPUT_DIR='path/to/eval_output/genai527'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node 8 \
  -m step1_gen_samples \
  genai527.yaml \
  --pretrained_lvlm_name_or_path ${MODEL_PATH} \
  --output_dir ${OUTPUT_DIR}
```

### Evaluation & Summary


Download [zhiqiulin/clip-flant5-xxl](https://huggingface.co/zhiqiulin/clip-flant5-xxl) to `$T5_PATH`.
Download [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) to `$VISION_TOWER`.

```bash
# switch to univa env
META_DIR="eval_prompts/genai527"
IMAGE_DIR=${OUTPUT_DIR}
CUDA_VISIBLE_DEVICES=4 VISION_TOWER=${VISION_TOWER} python -m step2_run_model \
    --model_path ${T5_PATH} \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR} > ${IMAGE_DIR}.txt
cat ${IMAGE_DIR}.txt
```

