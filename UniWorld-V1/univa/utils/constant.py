
SPACIAL_TOKEN = {
    'qwen2vl': {
        'image_token': '<|image_pad|>', 
        'image_begin_token': '<|vision_start|>', 
        'image_end_token': '<|vision_end|>', 
    }, 
    'qwen2p5vl': {
        'image_token': '<|image_pad|>', 
        'image_begin_token': '<|vision_start|>', 
        'image_end_token': '<|vision_end|>', 
    }, 
    'llava': {
        'image_token': '<image>', 
        'image_begin_token': '<im_start>', 
        'image_end_token': '<im_end>', 
    }, 
}
GENERATE_TOKEN = '<gen_image>'