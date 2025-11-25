import os
import torch
import logging
import warnings
import types
from ultrai2v.distributed.utils import gather_data_from_all_ranks

GRAD_NORM_INF = 1000.0

class AdaptiveGradClipper:
    def __init__(self, clip_grad_ema_decay: float = 0.99, init_max_grad_norm: float = 1.0, model_parallel_group=None):
        self.clip_grad_ema_decay = clip_grad_ema_decay
        if model_parallel_group is None:
            logging.warning("No model parallel group provided for AdaptiveGradClipper. Using default group.")
            self.model_parallel_group = torch.distributed.group.WORLD
        else:
            logging.info(f"Using model parallel group {model_parallel_group} for AdaptiveGradClipper.")
            self.model_parallel_group = model_parallel_group

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print(f"in AdaptiveGradClipper, rank: {torch.distributed.get_rank()}, device: {self.device}")

        self.moving_avg_max_grad_norm = -float('inf')
        self.moving_avg_max_grad_norm_var = 0.0
        self.grad_norm_before_clip = 0.0
        self.grad_norm_after_clip = 0.0
        self.max_grad_norm = init_max_grad_norm
        self.max_grad_norm_var = 0.0

        self.save_postfix = 'adaptive_grad_clip_states.pt'

    def state_dict(self) -> dict:
        return {
            "clip_grad_ema_decay": self.clip_grad_ema_decay,
            "moving_avg_max_grad_norm": self.moving_avg_max_grad_norm,
            "moving_avg_max_grad_norm_var": self.moving_avg_max_grad_norm_var,
            "grad_norm_before_clip": self.grad_norm_before_clip,
            "grad_norm_after_clip": self.grad_norm_after_clip,
            "max_grad_norm": self.max_grad_norm,
            "max_grad_norm_var": self.max_grad_norm_var,
        }
    
    def load_state_dict(self, state_dict: dict):
        self.clip_grad_ema_decay = state_dict["clip_grad_ema_decay"]
        self.moving_avg_max_grad_norm = state_dict["moving_avg_max_grad_norm"]
        self.moving_avg_max_grad_norm_var = state_dict["moving_avg_max_grad_norm_var"]
        self.grad_norm_before_clip = state_dict["grad_norm_before_clip"]
        self.grad_norm_after_clip = state_dict["grad_norm_after_clip"]
        self.max_grad_norm = state_dict["max_grad_norm"]
        self.max_grad_norm_var = state_dict["max_grad_norm_var"]

        if torch.distributed.get_rank() == 0:
            logs_info = f"""
            {'=' * 20}Adaptive Grad Clipper States Loaded{'=' * 20}
            clip_grad_ema_decay: {self.clip_grad_ema_decay}
            moving_avg_max_grad_norm: {self.moving_avg_max_grad_norm}
            moving_avg_max_grad_norm_var: {self.moving_avg_max_grad_norm_var}
            grad_norm_before_clip: {self.grad_norm_before_clip}
            grad_norm_after_clip: {self.grad_norm_after_clip}
            max_grad_norm: {self.max_grad_norm}
            max_grad_norm_var: {self.max_grad_norm_var}
            {'=' * 20}{'=' * len('Adaptive Grad Clipper States Loaded')}{'=' * 20}
            """
            logging.info(logs_info)

    def load(self, output_dir: str):
        load_path = os.path.join(output_dir, self.save_postfix)
        if not os.path.exists(load_path):
            warnings.warn(f"No adaptive grad clipper state file found at {load_path}. Use initial values.")
            return
        state_dict = torch.load(load_path, map_location='cpu')
        self.load_state_dict(state_dict)

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, self.save_postfix))

    def get_max_grad_norm(self, moving_avg_max_grad_norm, moving_avg_max_grad_norm_var) -> float:
        return moving_avg_max_grad_norm + 3.0 * (moving_avg_max_grad_norm_var ** 0.5)

    def get_max_grad_norm_var(self, moving_avg_max_grad_norm, grad_norm_before_clip) -> float:
        return (moving_avg_max_grad_norm - grad_norm_before_clip) ** 2

    # in first iteration, initialize moving averages
    def init_weights(self, grad_norm_before_clip_first_step):
        self.moving_avg_max_grad_norm = min(self.max_grad_norm, grad_norm_before_clip_first_step * 3)
        self.moving_avg_max_grad_norm_var = 0.0
        self.max_grad_norm = self.get_max_grad_norm(self.moving_avg_max_grad_norm, self.moving_avg_max_grad_norm_var)
        self.max_grad_norm_var = 0.0

    def adaptive_clip(self, parameters):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        else:
            is_generator = isinstance(parameters, types.GeneratorType)
            # prevent generators from being exhausted
            parameters = list(parameters)
            if is_generator and len(parameters) == 0:
                warnings.warn(
                    "`parameters` is an empty generator, no gradient clipping will occur.",
                    stacklevel=3,
                )
        grads_for_norm = [p.grad for p in parameters if p.grad is not None]
        if self.moving_avg_max_grad_norm < -1:
            grad_norm_before_clip_first_step = torch.nn.utils.clip_grad_norm_(parameters, GRAD_NORM_INF, error_if_nonfinite=True).item()
            grad_norm_before_clip_first_step = gather_data_from_all_ranks(torch.tensor([grad_norm_before_clip_first_step ** 2], device=self.device), group=self.model_parallel_group).sum().item() ** 0.5
            self.init_weights(grad_norm_before_clip_first_step)
        else:
           self.max_grad_norm = self.get_max_grad_norm(self.moving_avg_max_grad_norm, self.moving_avg_max_grad_norm_var)
        grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm, error_if_nonfinite=True).item()
        # self.grad_norm_before_clip = grad_norm_before_clip
        self.grad_norm_before_clip = gather_data_from_all_ranks(torch.tensor([grad_norm_before_clip ** 2], device=self.device), group=self.model_parallel_group).sum().item() ** 0.5
        if self.grad_norm_before_clip <= self.max_grad_norm:
            self.moving_avg_max_grad_norm = self.clip_grad_ema_decay * self.moving_avg_max_grad_norm + (1 - self.clip_grad_ema_decay) * self.grad_norm_before_clip
            self.max_grad_norm_var = self.get_max_grad_norm_var(self.moving_avg_max_grad_norm, self.grad_norm_before_clip)
            self.moving_avg_max_grad_norm_var = self.clip_grad_ema_decay * self.moving_avg_max_grad_norm_var + (1 - self.clip_grad_ema_decay) * self.max_grad_norm_var
            self.grad_norm_after_clip = self.grad_norm_before_clip
        else:
            grad_norm_after_clip = torch.nn.utils.get_total_norm(grads_for_norm, error_if_nonfinite=True).item()
            # self.grad_norm_after_clip = grad_norm_after_clip
            self.grad_norm_after_clip = gather_data_from_all_ranks(torch.tensor([grad_norm_after_clip ** 2], device=self.device), group=self.model_parallel_group).sum().item() ** 0.5
        # if torch.distributed.get_rank() == 0:
        #     print(f"Adaptive Grad Clip: grad_norm_before_clip={self.grad_norm_before_clip:.4f}, max_grad_norm={self.max_grad_norm:.4f}, grad_norm_after_clip={self.grad_norm_after_clip:.4f}")
        return torch.tensor(self.grad_norm_after_clip)