import torch
import logging
from accelerate.utils import is_npu_available as accelerate_is_npu_available

def str_to_precision(s):
    if s == "bfloat16" or s == "bf16":
        return torch.bfloat16
    elif s == "float16" or s == "fp16":
        return torch.float16
    elif s == "float32" or s == "float" or s == "fp32":
        return torch.float32
    elif s == "float64" or s == "double" or s == "fp64":
        return torch.float64
    elif s == "int64":
        return torch.int64
    elif s == "int32" or s == "int":
        return torch.int32
    elif s == "uint8":
        return torch.uint8
    else:
        raise ValueError(f"Unsupported precision string: {s}")

def precision_to_str(precision):
    if precision == torch.bfloat16:
        return "bfloat16"
    elif precision == torch.float16:
        return "float16"
    elif precision == torch.float32:
        return "float32"
    elif precision == torch.float64:
        return "float64"
    elif precision == torch.int64:
        return "int64"
    elif precision == torch.int32:
        return "int32"
    elif precision == torch.uint8:
        return "uint8"
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
def params_nums_to_str(params_num):
    if params_num >= 1e9:
        return f"{params_num / 1e9:.2f}B"
    elif params_num >= 1e6:
        return f"{params_num / 1e6:.2f}M"
    elif params_num >= 1e3:
        return f"{params_num / 1e3:.2f}K"
    else:
        return str(params_num)

def get_memory_allocated():
    return f"{torch.cuda.memory_allocated() / 1024**3:.2f}"  # GiB

def is_npu_available():
    return accelerate_is_npu_available(True)

def check_and_import_npu():
    if is_npu_available():
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        torch_npu.npu.config.allow_internal_format = False