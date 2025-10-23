import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()

def _get_config(base_model="kontext", n_gpus=1, gradient_step_per_epoch=1, reward_fn={}, name=""):
    config = base.get_config()

    config.base_model = base_model
    config.dataset = "../edit-r1-dataset"
    
    config.pretrained.model = "black-forest-labs/FLUX.1-Kontext-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 15
    config.sample.guidance_scale = 2.5
    config.resolution = 512
    config.train.beta = 0.0001
    config.sample.noise_level = 0.7
    bsz = 3

    config.sample.num_image_per_prompt = 12

    config.sample.ban_std_thres = 0.05
    config.sample.ban_prompt = False

    num_groups = 24

    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1

    # special design, the test set has a total of 1018/2212/2048 for ocr/geneval/pickscore, to make gpu_num*bs*n as close as possible to it, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.test_batch_size = bsz
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = "geneval"

    config.run_name = f"nft_{base_model}_{name}"
    config.save_dir = f"logs/nft/{base_model}/{name}"
    config.reward_fn = reward_fn

    config.decay_type = 1
    config.beta = 1.0
    config.train.adv_mode = "all"

    # config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    return config

def kontext_mllm_reward():
    reward_fn = {
        "mllm_score_continue": 1.0,
    }
    config = _get_config(
        base_model="kontext",
        n_gpus=24,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn=reward_fn,
        name="mllm_score_continue",
    )
    return config

def kontext_mllm_reward_ban_prompt():
    reward_fn = {
        "mllm_score_continue": 1.0,
    }
    config = _get_config(
        base_model="kontext",
        n_gpus=24,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn=reward_fn,
        name="mllm_score_continue_ban_prompt",
    )
    config.sample.ban_prompt = True
    config.sample.ban_std_thres = 0.05
    return config