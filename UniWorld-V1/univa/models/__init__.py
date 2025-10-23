from .modeling_univa import UnivaQwen2ForCausalLM
from .qwen2vl.modeling_univa_qwen2vl import UnivaQwen2VLForConditionalGeneration
from .qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration

MODEL_TYPE = {
    'llava': UnivaQwen2ForCausalLM, 
    'qwen2vl': UnivaQwen2VLForConditionalGeneration, 
    'qwen2p5vl': UnivaQwen2p5VLForConditionalGeneration
}