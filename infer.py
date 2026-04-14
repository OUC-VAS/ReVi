import os
import yaml
import torch
import random
import argparse
from tinysam import sam_model_registry
from pre_model import addReVito_model
from vis_result import visualize_results
from metric import caculate_metric

seed = 2026
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def infer(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # load yaml
    model_type = config['model']['type']
    checkpoint_path = config['model']['checkpoint']
    input_folder = config['data']['input_folder']
    output_folder = config['data']['output_folder']
    point = config['inference'].get('point', None)
    gt_folder = config['data']['gt_folder']
    device = config.get('device', None)

    # initialize the backbone
    model = sam_model_registry[model_type]()#checkpoint="./weights/tinysam.pth"
    model = addReVito_model(model)

    # all models name
    # with open('model_modules.txt', 'w', encoding='utf-8') as f:
    #     for name, module in model.named_modules():
    #         f.write(f"{name}: {type(module).__name__}\n")
    #         f.write("-" * 50 + "\n")

    # load your weight
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Check checkpoint content
        if 'model_state_dict' in checkpoint:
            # Load from full checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from checkpoint, epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"Best validation loss: {checkpoint.get('loss', 'unknown'):.4f}")
        else:
            # Directly load model state dict
            model.load_state_dict(checkpoint)
            print("Directly loading model state dict")

        print(f"Model weights loaded from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    print(f"Using device: {device}")
    model.to(device=device)

    # visualize_results(model, "38t.tif", None) # you can infer just one map
    visualize_results(
        model=model,
        input_folder=input_folder,
        output_folder=output_folder,
        point=point,
        device=device
    )

    caculate_metric(
        os.path.join(output_folder, "masks"),
        gt_folder=gt_folder
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with config file")
    parser.add_argument("--config", type=str, default="infer_config.yaml",
                        help="Path to configuration YAML file (default: config.yaml)")
    args = parser.parse_args()

    infer(args.config)