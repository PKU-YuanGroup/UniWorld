from .attention import flash_attention
from .t5 import T5Decoder, T5Encoder, T5EncoderModel, T5Model
from .tokenizers import HuggingfaceTokenizer
from .vae import WanVAE

from .want2v import (
    models as wan_models, 
    models_main_block as wan_models_main_block, 
    models_blocks_to_float as wan_models_blocks_to_float,
    models_blocks_to_output_float as wan_models_blocks_to_output_float,
    cp_plans as wan_cp_plans
)
from .flashi2v import (
    models as flashi2v_models, 
    models_main_block as flashi2v_models_main_block, 
    models_blocks_to_float as flashi2v_models_blocks_to_float,
    models_blocks_to_output_float as flashi2v_models_blocks_to_output_float,
    cp_plans as flashi2v_cp_plans
)

models = {}
models.update(wan_models)
models.update(flashi2v_models)

models_main_block = {}
models_main_block.update(wan_models_main_block)
models_main_block.update(flashi2v_models_main_block)

models_blocks_to_float = {}
models_blocks_to_float.update(wan_models_blocks_to_float)
models_blocks_to_float.update(flashi2v_models_blocks_to_float)

models_blocks_to_output_float = {}
models_blocks_to_output_float.update(wan_models_blocks_to_output_float)
models_blocks_to_output_float.update(flashi2v_models_blocks_to_output_float)

models_cp_plans = {}
models_cp_plans.update(wan_cp_plans)
models_cp_plans.update(flashi2v_cp_plans)


__all__ = [
    'WanVAE',
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
    'HuggingfaceTokenizer',
    'models',
    'models_main_block',
    'models_blocks_to_float',
    'models_blocks_to_output_float',
    'models_cp_plans',
    'flash_attention',
]
