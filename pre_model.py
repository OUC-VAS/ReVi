import torch.nn as nn
from peft import LoraConfig, get_peft_model
from revi.revi import replace_with_parallel_transformer

def setup_lora_for_sam(model, lora_r=8, lora_alpha=32, lora_dropout=0.1):
    # add lora to target layers
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            if hasattr(module, 'groups') and module.groups > 1:
                if lora_r % module.groups == 0:
                    target_modules.add(name)
            else:
                target_modules.add(name)
    #print(len(target_modules)) # number of layers with lora

    # lora config
    lora_config = LoraConfig(
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules)
    )

    # set lora to model
    lora_model = get_peft_model(model, lora_config)

    # print the params info
    # lora_model.print_trainable_parameters()

    return lora_model

def addReVito_model(model):
    model = setup_lora_for_sam(model)

    #add revi to the target attention
    target_layers = [
        "layers.1.blocks.0.attn",
        "layers.1.blocks.1.attn",
        "layers.2.blocks.0.attn",
        "layers.2.blocks.1.attn",
        "layers.2.blocks.2.attn",
        "layers.2.blocks.3.attn",
        "layers.2.blocks.4.attn",
        "layers.2.blocks.5.attn",
        "layers.3.blocks.0.attn",
        "layers.3.blocks.1.attn"
    ]
    for layer_path in target_layers:
        model.image_encoder = replace_with_parallel_transformer(model.image_encoder, layer_path)

    return model