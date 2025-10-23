
The original code is from [WISE](https://github.com/PKU-YuanGroup/WISE).

Environment:
```
pip install openai==0.28.0
```


## Eval

### Generate samples

```bash
# switch to univa env
MODEL_PATH='path/to/model'
OUTPUT_DIR='path/to/eval_output/wise'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node 8 \
  -m step1_gen_samples \
  wise.yaml \
  --pretrained_lvlm_name_or_path ${MODEL_PATH} \
  --output_dir ${OUTPUT_DIR}
```

### Evaluation

Evaluate using GPT-4o-2024-05-13:
Write your gpt-api-key to `--api_key`.

```bash
IMAGE_DIR=${OUTPUT_DIR}
python step2_gpt_eval.py \
    --json_path data/cultural_common_sense.json \
    --output_dir ${IMAGE_DIR}/Results/cultural_common_sense \
    --image_dir ${IMAGE_DIR} \
    --api_key "" \
    --model "gpt-4o-2024-05-13" \
    --result_full ${IMAGE_DIR}/Results/cultural_common_sense_full_results.json \
    --result_scores ${IMAGE_DIR}/Results/cultural_common_sense_scores_results.jsonl \
    --max_workers 96

IMAGE_DIR=${OUTPUT_DIR}
python step2_gpt_eval.py \
    --json_path data/spatio-temporal_reasoning.json \
    --output_dir ${IMAGE_DIR}/Results/spatio-temporal_reasoning \
    --image_dir ${IMAGE_DIR} \
    --api_key "" \
    --model "gpt-4o-2024-05-13" \
    --result_full ${IMAGE_DIR}/Results/spatio-temporal_reasoning_results.json \
    --result_scores ${IMAGE_DIR}/Results/spatio-temporal_reasoning_results.jsonl \
    --max_workers 96

IMAGE_DIR=${OUTPUT_DIR}
python step2_gpt_eval.py \
    --json_path data/natural_science.json \
    --output_dir ${IMAGE_DIR}/Results/natural_science \
    --image_dir ${IMAGE_DIR} \
    --api_key "" \
    --model "gpt-4o-2024-05-13" \
    --result_full ${IMAGE_DIR}/Results/natural_science_full_results.json \
    --result_scores ${IMAGE_DIR}/Results/natural_science_scores_results.jsonl \
    --max_workers 96
```

### Summary  

```bash
python step3_wise_cal.py \
    "${IMAGE_DIR}/Results/cultural_common_sense_scores_results.jsonl" \
    "${IMAGE_DIR}/Results/natural_science_scores_results.jsonl" \
    "${IMAGE_DIR}/Results/spatio-temporal_reasoning_results.jsonl" \
    --category all
```
