import json
import os
import base64
import re
import argparse
import openai
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Quality Assessment Tool')
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--api_key', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--result_full', required=True)    # .json
    parser.add_argument('--result_scores', required=True)  # .jsonl
    parser.add_argument('--api_base', default=None, type=str)
    parser.add_argument('--max_workers', type=int, default=10)
    return parser.parse_args()

def get_config(args):
    return {
        "json_path": args.json_path,
        "image_dir": args.image_dir,
        "output_dir": args.output_dir,
        "api_key": args.api_key,
        "api_base": args.api_base,
        "model": args.model,
        "result_files": {"full": args.result_full, "scores": args.result_scores},
        "max_workers": args.max_workers,
    }

# ------------------------- 工具函数 -------------------------
def load_jsonl(path: str) -> Dict[int, Dict]:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return {}
    records = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            records[obj["prompt_id"]] = obj
    return records

def load_json(path: str) -> Dict[int, Dict]:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item["prompt_id"]: item for item in data}

def extract_scores(txt: str) -> Dict[str, float]:
    pat = r"\*{0,2}(Consistency|Realism|Aesthetic Quality)\*{0,2}\s*[::]?\s*(\d)"
    matches = re.findall(pat, txt, re.IGNORECASE)
    out = {}
    for k, v in matches:
        out[k.lower().replace(" ", "_")] = float(v)
    return out

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def load_prompts(path: str) -> Dict[int, Dict[str, Any]]:
    with open(path, 'r') as f:
        data = json.load(f)
    return {item["prompt_id"]: item for item in data}

def build_evaluation_messages(prompt_data: Dict, image_base64: str) -> list:
    return [

        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a professional Vincennes image quality audit expert, please evaluate the image quality strictly according to the protocol."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Please evaluate strictly and return ONLY the three scores as requested.

# Text-to-Image Quality Evaluation Protocol

## System Instruction
You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS. Only images meeting the HIGHEST standards should receive top scores.

**Input Parameters**  
- PROMPT: [User's original prompt to]  
- EXPLANATION: [Further explanation of the original prompt] 
---

## Scoring Criteria

**Consistency (0-2):**  How accurately and completely the image reflects the PROMPT.
* **0 (Rejected):**  Fails to capture key elements of the prompt, or contradicts the prompt.
* **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, or not accurately.  Noticeable deviations from the prompt's intent.
* **2 (Exemplary):**  Perfectly and completely aligns with the PROMPT.  Every single element and nuance of the prompt is flawlessly represented in the image. The image is an ideal, unambiguous visual realization of the given prompt.

**Realism (0-2):**  How realistically the image is rendered.
* **0 (Rejected):**  Physically implausible and clearly artificial. Breaks fundamental laws of physics or visual realism.
* **1 (Conditional):** Contains minor inconsistencies or unrealistic elements.  While somewhat believable, noticeable flaws detract from realism.
* **2 (Exemplary):**  Achieves photorealistic quality, indistinguishable from a real photograph.  Flawless adherence to physical laws, accurate material representation, and coherent spatial relationships. No visual cues betraying AI generation.

**Aesthetic Quality (0-2):**  The overall artistic appeal and visual quality of the image.
* **0 (Rejected):**  Poor aesthetic composition, visually unappealing, and lacks artistic merit.
* **1 (Conditional):**  Demonstrates basic visual appeal, acceptable composition, and color harmony, but lacks distinction or artistic flair.
* **2 (Exemplary):**  Possesses exceptional aesthetic quality, comparable to a masterpiece.  Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating artistic style. Demonstrates a high degree of artistic vision and execution.

---

## Output Format

**Do not include any other text, explanations, or labels.** You must return only three lines of text, each containing a metric and the corresponding score, for example:

**Example Output:**
Consistency: 2
Realism: 1
Aesthetic Quality: 0

---

**IMPORTANT Enforcement:**

Be EXTREMELY strict in your evaluation. A score of '2' should be exceedingly rare and reserved only for images that truly excel and meet the highest possible standards in each metric. If there is any doubt, downgrade the score.

For **Consistency**, a score of '2' requires complete and flawless adherence to every aspect of the prompt, leaving no room for misinterpretation or omission.

For **Realism**, a score of '2' means the image is virtually indistinguishable from a real photograph in terms of detail, lighting, physics, and material properties.

For **Aesthetic Quality**, a score of '2' demands exceptional artistic merit, not just pleasant visuals.

--- 
Here are the Prompt and EXPLANATION for this evaluation:
PROMPT: "{prompt_data['Prompt']}"
EXPLANATION: "{prompt_data['Explanation']}"
Please strictly adhere to the scoring criteria and follow the template format when providing your results."""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
    ]



def evaluate_image(prompt_id: int, prompt: Dict, img_path: str, cfg: Dict):
    try:
        print(f"Evaluating {prompt_id} ...")
        img64 = encode_image(img_path)
        msgs = build_evaluation_messages(prompt, img64)
        resp = openai.ChatCompletion.create(
            model=cfg["model"], messages=msgs, temperature=0.0, max_tokens=2000
        )
        eval_txt = resp['choices'][0]['message']['content']
        scores = extract_scores(eval_txt)

        print(f"\n--- {prompt_id} ---\n{eval_txt}\n--------------\n")

        return (
            {  # full record
                "prompt_id": prompt_id,
                "prompt": prompt["Prompt"],
                "key": prompt["Explanation"],
                "image_path": img_path,
                "evaluation": eval_txt
            },
            {  # score record
                "prompt_id": prompt_id,
                "Subcategory": prompt["Subcategory"],
                "consistency": scores.get("consistency", 0),
                "realism": scores.get("realism", 0),
                "aesthetic_quality": scores.get("aesthetic_quality", 0)
            }
        )
    except Exception as e:
        print(f"[ERR] {prompt_id}: {e}")
        return None  # 失败时返回 None，主线程忽略

def save_results(data: List[Dict], filename: str, cfg: Dict):
    path = os.path.join(cfg["output_dir"], filename)
    if filename.endswith('.jsonl'):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {path}")

def main():
    args = parse_arguments()
    cfg = get_config(args)
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    openai.api_key  = cfg["api_key"]
    if cfg['api_base'] is not None:
        openai.api_base = cfg['api_base']

    prompts = load_prompts(cfg["json_path"])

    # ---- 断点续跑：读取已有结果 ----
    exist_scores = load_jsonl(os.path.join(cfg["output_dir"], cfg["result_files"]["scores"]))
    exist_full   = load_json (os.path.join(cfg["output_dir"], cfg["result_files"]["full"]))
    done_ids = set(exist_scores.keys())

    # ---- 生成待评测任务 ----
    tasks = []
    for pid, pdata in prompts.items():
        if pid in done_ids:
            continue
        img_path = os.path.join(cfg["image_dir"], f"{pid}.png")
        if not os.path.exists(img_path):
            print(f"[WARN] Missing image: {img_path}")
            continue
        tasks.append((pid, pdata, img_path))

    # ---- 多线程评测 ----
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["max_workers"]) as ex:
        future_to_id = {ex.submit(evaluate_image, pid, pd, ip, cfg): pid for pid, pd, ip in tasks}
        for fut in concurrent.futures.as_completed(future_to_id):
            res = fut.result()
            if res is None:
                continue
            full_rec, score_rec = res
            exist_full[full_rec["prompt_id"]]     = full_rec
            exist_scores[score_rec["prompt_id"]]  = score_rec

    # ---- 合并排序保存 ----
    full_sorted  = [exist_full[k]   for k in sorted(exist_full.keys())]
    score_sorted = [exist_scores[k] for k in sorted(exist_scores.keys())]

    save_results(full_sorted,  cfg["result_files"]["full"],   cfg)
    save_results(score_sorted, cfg["result_files"]["scores"], cfg)

if __name__ == "__main__":
    main()