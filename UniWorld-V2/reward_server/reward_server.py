import os
import torch
from typing import List
from vllm import LLM, SamplingParams
import vllm
from PIL import Image
from io import BytesIO
import base64
import pickle
import traceback
from flask import Flask, request
import ray
import asyncio
import prompt_template

if vllm.__version__ != "0.9.2":
    raise ValueError("vLLM version must be 0.9.2")

os.environ["VLLM_USE_V1"] = "0"  # IMPORTANT

app = Flask(__name__)

# Global variables
score_idx = [15, 16, 17, 18, 19, 20]
workers = []  # Ray actors for each GPU
MODEL_PATH = "/mnt/data/checkpoints/Qwen/Qwen2.5-VL-32B-Instruct/"
NUM_GPUS = 8
NUM_TP = 2

def get_base64(image):
    image_data = BytesIO()
    image.save(image_data, format="JPEG")
    image_data_bytes = image_data.getvalue()
    encoded_image = base64.b64encode(image_data_bytes).decode("utf-8")
    return encoded_image


class LogitsSpy:
    def __init__(self):
        self.processed_logits: list[torch.Tensor] = []

    def __call__(self, token_ids: list[int], logits: torch.Tensor):
        self.processed_logits.append(logits)
        return logits


@ray.remote(num_gpus=NUM_TP)
class ModelWorker:
    def __init__(self):
        self.llm = None
        self.load_model()

    def load_model(self):
        """Load the Qwen2-VL model using vLLM on specific GPU"""
        self.llm = LLM(
            MODEL_PATH, limit_mm_per_prompt={"image": 3}, tensor_parallel_size=NUM_TP
        )

    def evaluate_image(
        self, image_bytes, prompt, ref_image_bytes=None, requirement: str = ""
    ):
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes), formats=["jpeg"])
        ref_image = Image.open(BytesIO(ref_image_bytes), formats=["jpeg"])
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": ref_image},
                    {"type": "image_pil", "image_pil": image},
                    {
                        "type": "text",
                        "text": prompt_template.SCORE_LOGIT.format(
                            prompt=prompt, requirement=requirement
                        ),
                    },
                ],
            },
        ]
        return self._vllm_evaluate(conversation)

    def _vllm_evaluate(self, conversation, max_tokens=3, max_score=5):
        logits_spy = LogitsSpy()
        sampling_params = SamplingParams(
            max_tokens=max_tokens, logits_processors=[logits_spy]
        )
        self.llm.chat(conversation, sampling_params=sampling_params)
        try:
            if logits_spy.processed_logits:
                probs = torch.softmax(logits_spy.processed_logits[0][score_idx], dim=-1)
                score_prob = (
                    torch.sum(
                        probs * torch.arange(len(score_idx)).to(probs.device)
                    ).item()
                    / max_score
                )
                print(f"Score: {score_prob:.4f}")
                return score_prob
            else:
                print("No outputs received")
                return 0.0
        except Exception as e:
            print(f"Error in _vllm_evaluate: {e}")
            score = 0.0

        return score


def initialize_ray_workers(num_gpus=8, num_tp=4):
    global workers
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Create workers for each GPU
    workers = []
    for _ in range(num_gpus // num_tp):
        worker = ModelWorker.remote()
        workers.append(worker)

    print(f"Initialized {num_gpus//num_tp} Ray workers")
    return workers


async def evaluate_images_async(
    image_bytes_list, prompts, ref_image_bytes_list=None, requirements: List[str] = []
):
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    if not requirements:
        requirements = [""] * len(prompts)
    if ref_image_bytes_list is None:
        ref_image_bytes_list = [None] * len(prompts)
    for i, (image_bytes, prompt, ref_image_bytes, requirement) in enumerate(
        zip(image_bytes_list, prompts, ref_image_bytes_list, requirements)
    ):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_image.remote(
            image_bytes, prompt, ref_image_bytes, requirement
        )
        tasks.append(task)

    scores = ray.get(tasks)
    return scores


def evaluate_images(
    image_bytes_list, prompts, ref_image_bytes_list=None, requirements=[]
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_images_async(
                image_bytes_list, prompts, ref_image_bytes_list, requirements
            )
        )
        return scores
    finally:
        loop.close()


@app.route("/mode/<mode>", methods=["POST"])
def inference_mode(mode):
    data = request.get_data()

    assert mode in ["logits_non_cot"], "Invalid mode"

    try:
        data = pickle.loads(data)
        image_bytes_list = data["images"]
        ref_image_bytes_list = data.get("ref_images", None)
        prompts = data["prompts"]
        metadatas = data.get("metadatas", [])
        requirements = []
        for metadata in metadatas:
            requirements.append(metadata.get("requirement", ""))

        scores = evaluate_images(
            image_bytes_list, prompts, ref_image_bytes_list, requirements
        )

        response = {"scores": scores}
        response = pickle.dumps(response)
        returncode = 200
    except KeyError as e:
        response = f"KeyError: {str(e)}"
        response = response.encode("utf-8")
        returncode = 500
    except Exception as e:
        response = traceback.format_exc()
        response = response.encode("utf-8")
        returncode = 500

    return response, returncode


if __name__ == "__main__":
    initialize_ray_workers(NUM_GPUS, NUM_TP)
    print(f"Starting Flask server with {NUM_GPUS//NUM_TP} Ray workers...")
    app.run(host="0.0.0.0", port=12341, debug=False)
