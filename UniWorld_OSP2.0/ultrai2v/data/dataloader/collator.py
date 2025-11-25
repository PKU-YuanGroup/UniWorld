import torch
from ultrai2v.utils.constant import VIDEO, PROMPT, PROMPT_IDS, PROMPT_MASK, START_FRAME, NAME_INDEX, VLM_INPUTS

class WanDataCollator:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        # import ipdb; ipdb.set_trace()
        video = torch.stack([i[VIDEO] for i in batch]) if batch[0][VIDEO] is not None else None # in evaluation mode, we have no video gt.
        prompt_ids = torch.cat([i[PROMPT_IDS] for i in batch])
        prompt_mask = torch.cat([i[PROMPT_MASK] for i in batch]) if batch[0][PROMPT_MASK] is not None else None

        return {
            VIDEO: video,
            PROMPT_IDS: prompt_ids,
            PROMPT_MASK: prompt_mask,
        }
    
class FlashI2VDataCollator:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        # print(len(batch), '!!!!!!!!!!!!')
        # import ipdb; ipdb.set_trace()
        video = torch.stack([i[VIDEO] for i in batch]) if batch[0][VIDEO] is not None else None # in evaluation mode, we have no video gt.
        start_frame = torch.stack([i[START_FRAME] for i in batch])
        prompt_ids = torch.cat([i[PROMPT_IDS] for i in batch])
        prompt_mask = torch.cat([i[PROMPT_MASK] for i in batch]) if batch[0][PROMPT_MASK] is not None else None

        vlm_inputs = {}
        vlm_inputs['input_ids'] = torch.cat([sample[VLM_INPUTS]['input_ids'] for sample in batch])
        vlm_inputs['attention_mask'] = torch.cat([sample[VLM_INPUTS]['attention_mask'] for sample in batch])
        vlm_inputs['pixel_values'] = torch.stack([sample[VLM_INPUTS]['pixel_values'] for sample in batch])


        return {
            VIDEO: video,
            START_FRAME: start_frame,
            PROMPT_IDS: prompt_ids,
            PROMPT_MASK: prompt_mask,
            VLM_INPUTS: vlm_inputs,
        }

class T2VEvalDataCollator:
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, batch):
        prompt = [i[PROMPT] for i in batch]
        name_index = [i[NAME_INDEX] for i in batch]
        return {
            PROMPT: prompt,
            NAME_INDEX: name_index
        }
    
class I2VEvalDataCollator:
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, batch):
        start_frame = [i[START_FRAME] for i in batch]
        prompt = [i[PROMPT] for i in batch]
        name_index = [i[NAME_INDEX] for i in batch]
        return {
            START_FRAME: start_frame,
            PROMPT: prompt,
            NAME_INDEX: name_index
        }

data_collator = {
    'wan_t2v': WanDataCollator,
    'flashi2v': FlashI2VDataCollator,
    't2v_eval': T2VEvalDataCollator,
    'i2v_eval': I2VEvalDataCollator
}