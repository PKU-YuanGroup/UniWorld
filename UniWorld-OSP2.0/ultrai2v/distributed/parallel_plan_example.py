from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    SequenceParallel,
    PrepareModuleInputOutput,
)

from torch.distributed.tensor import Shard, Replicate

base_cp_plan = {
    "cp_input_layer": PrepareModuleInput(
        input_layouts=(Replicate(),),
        desired_input_layouts=(Shard(1),),
    ),
    "cp_output_layer": PrepareModuleOutput(
        output_layouts=(Shard(1),),
        desired_output_layouts=(Replicate(),),
    ),
}

attention_cp_plan = {
    "cp_before_attention_layer": PrepareModuleInput(
        input_layouts=(Shard(2),),
        desired_input_layouts=(Shard(1),),
    ),
    "cp_after_attention_layer": PrepareModuleOutput(
        output_layouts=(Shard(1),),
        desired_output_layouts=(Shard(2),),
    ),
}

base_tp_plan = {
    "tok_embeddings": RowwiseParallel(
        input_layouts=Replicate(),
        output_layouts=Shard(1),
    ),
    "norm": SequenceParallel(),
    "output": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
}

tp_plan = {
    # Now the input and output of SequenceParallel has Shard(1) layouts,
    # to represent the input/output tensors sharded on the sequence dimension
    "attention_norm": SequenceParallel(),
    "attention": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    "attention.wq": ColwiseParallel(use_local_output=False),
    "attention.wk": ColwiseParallel(use_local_output=False),
    "attention.wv": ColwiseParallel(use_local_output=False),
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    "ffn_norm": SequenceParallel(),
    "feed_forward": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    "feed_forward.w3": ColwiseParallel(),
}
