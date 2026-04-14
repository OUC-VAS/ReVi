import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pre_model import addReVito_model
from tinysam import sam_model_registry
from glob import glob
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import random
from edge_generator import EdgeGenerator

seed = 2026
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class TamperDetectionNoPointsDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, mask_dir, transform=None, edge_generator=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.edge_generator = edge_generator
        if transform is not None:
            print("train")
            self.transform = get_simple_transform()
        else:
            self.transform = get_test_transform()

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        self.image_files = []

        for ext in image_extensions:
            pattern = os.path.join(image_dir, ext)
            self.image_files.extend(glob(pattern))

        # 预计算掩码路径映射
        self.mask_mapping = {}
        for img_path in self.image_files:
            img_name = os.path.basename(img_path)

            possible_mask_names = [
                # image.jpg -> image_mask.png
                img_name.replace('.png', '_mask.png').replace('.jpeg', '_mask.png')
                .replace('.jpg', '_mask.png').replace('.bmp', '_mask.png')
                .replace('.tif', '_mask.png'),
                # image.jpg -> image_gt.png
                img_name.replace('.png', '_gt.png').replace('.jpeg', '_gt.png')
                .replace('.jpg', '_gt.png').replace('.bmp', '_gt.png')
                .replace('.tif', '_gt.png'),
                # same name dif ext
                img_name.replace('.png', '.png').replace('.jpeg', '.png')
                .replace('.jpg', '.png').replace('.bmp', '.png')
                .replace('.tif', '.png'),
                # forged.tif
                img_name.replace('t.tif', 'forged.tif')
            ]

            mask_path = None
            for mask_name in possible_mask_names:
                candidate_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(candidate_path):
                    mask_path = candidate_path
                    break

            if mask_path is not None:
                self.mask_mapping[img_path] = mask_path
            else:
                print(f"Warning: No mask found for {img_name} (tried: {possible_mask_names})")
                self.mask_mapping[img_path] = None

        # fliter image without mask
        valid_images = [img for img, mask in self.mask_mapping.items() if mask is not None]
        self.image_files = valid_images

        print(f"Found {len(self.image_files)} valid image-mask pairs")
        print(f"All points will be set to None")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_mapping[image_path]

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot load mask: {mask_path}")
        mask = (mask > 127).astype(np.float32)
        mask_ten = torch.tensor(mask, dtype=torch.float)
        mask_ten = mask_ten.view(1, 1, mask_ten.shape[0], mask_ten.shape[1])

        edge_mask = None
        if self.edge_generator is not None:
            edge_mask = self.edge_generator(mask_ten)
            edge_mask = edge_mask.squeeze().squeeze()
        if self.transform:
            # Apply data augmentation
            if hasattr(self.transform, 'keypoint_params'):
                transformed = self.transform(image=image, mask=mask, edge_mask=edge_mask)
                image = transformed['image']
                mask = transformed['mask']
                edge_mask = transformed['edge_mask']
            else:
                transformed = self.transform(image=image, mask=mask, edge_mask=edge_mask)
                image = transformed['image']
                mask = transformed['mask']
                edge_mask = transformed['edge_mask']

        # convert to tensor
        mask_tensor = mask.unsqueeze(0)
        edge_mask = edge_mask.unsqueeze(0)
        image_tensor = image
        if image.max() > 1.0:
            image_tensor = image / 255.0
        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0
        if edge_mask.max() > 1.0:
            edge_mask = edge_mask / 255.0
            edge_mask = (edge_mask > 0.5).float()
        else:
            edge_mask = (edge_mask > 0.5).float()

        point_tensor = torch.tensor([], dtype=torch.float32)  # Empty tensor indicates no point

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'point': point_tensor,
            'edge': edge_mask,
            'image_path': image_path
        }


def get_simple_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={
    'edge_mask': 'mask',
    })
def get_test_transform():
    return A.Compose([
        A.Resize(256, 256),
        ToTensorV2(),
    ], additional_targets={
    'edge_mask': 'mask',
    })

# trainer
def train_sam_ReVi(model, train_loader, val_loader, edge_generator,num_epochs=500, lr=2e-5, device='cuda', save_best_path = "./"):

    for name, param in model.named_parameters():
        if "odownsample" in name:
            param.requires_grad = True
    model.train()
    lr = float(lr)
    #print(lr)
    #model.print_trainable_parameters()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=2000)

    train_losses = []
    val_losses = []
    val_f1_scores = []

    best_f1 = 0.0
    edge_lambda = 20

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch in train_pbar:
            images = batch['image'].to(device)
            true_masks = batch['mask'].to(device)
            edge_masks = batch['edge'].to(device)
            #points = batch['point'].to(device)

            optimizer.zero_grad()

            image_embeddings = model.image_encoder(images)

            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings
            )
            masks = model.postprocess_masks(low_res_masks, (256, 256), (256, 256))

            single_channel_mask = torch.mean(masks, dim=1, keepdim=True)  # [B, 3, H, W] -> [B, 1, H, W]

            loss = F.binary_cross_entropy_with_logits(
                single_channel_mask, true_masks,
                pos_weight=torch.tensor([4.0], device=low_res_masks.device)
            )
            edge_loss = F.binary_cross_entropy_with_logits(
                                input=single_channel_mask,
                                target=true_masks,
                                weight=edge_masks
                            ) * edge_lambda
            loss += edge_loss

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        epoch_val_loss = 0
        all_true_masks = []
        all_pred_masks = []
        all_f1_scores = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for batch in val_pbar:
                images = batch['image'].to(device)
                true_masks = batch['mask'].to(device)
                edge_masks = batch['edge'].to(device)
                #points = batch['point'].to(device)

                image_embeddings = model.image_encoder(images)

                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )

                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings
                )

                masks = model.postprocess_masks(low_res_masks, (256, 256), (256, 256))
                single_channel_mask = torch.mean(masks, dim=1, keepdim=True)  # [B, 3, H, W] -> [B, 1, H, W]

                loss = F.binary_cross_entropy_with_logits(
                    single_channel_mask, true_masks,
                    pos_weight=torch.tensor([4.0], device=low_res_masks.device)
                )
                edge_loss = F.binary_cross_entropy_with_logits(
                    input=single_channel_mask,
                    target=true_masks,
                    weight=edge_masks
                ) * edge_lambda
                loss += edge_loss

                loss += loss.item()

                pred_masks = (single_channel_mask > 0.5).float()

                pred_np = pred_masks.cpu().numpy().flatten()
                true_np = true_masks.cpu().numpy().flatten()

                TP = np.sum((true_np == 1) & (pred_np == 1))
                FP = np.sum((true_np == 0) & (pred_np == 1))
                FN = np.sum((true_np == 1) & (pred_np == 0))

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0

                batch_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                all_f1_scores.append(batch_f1)

                all_true_masks.append(true_np)
                all_pred_masks.append(pred_np)

                # upload pbar
                val_pbar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'F1': f'{batch_f1:.4f}'
                })

        # avg loss
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # avg f1
        avg_f1 = np.mean(all_f1_scores)
        val_f1_scores.append(avg_f1)

        scheduler.step()

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val F1: {avg_f1:.4f}')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.2e}')

        # save best model
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_f1,
            }, save_best_path)
            print('  Best model saved!')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(val_f1_scores, label='Validation F1', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Validation F1')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss_and_f1.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model, train_losses, val_losses

def main(config_path: str):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # initialization
    model_type = config['model']['type']
    checkpoint = config['model']['checkpoint']
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    sam = addReVito_model(sam)

    resume_run = config['model']['resume_run']
    resume_pth = config['model']['resume_pth']
    if resume_run:
        checkpoint = torch.load(resume_pth, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            sam.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded model from checkpoint, epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"Best validation loss: {checkpoint.get('loss', 'unknown'):.4f}")
        else:
            sam.load_state_dict(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    sam.to(device=device)

    train_image_dir = config['data']['train_image_dir']
    train_mask_dir = config['data']['train_mask_dir']
    val_image_dir = config['data']['val_image_dir']
    val_mask_dir = config['data']['val_mask_dir']

    edge_generator = EdgeGenerator(config['edge_generator']['kernel_size'])

    train_dataset = TamperDetectionNoPointsDataset(train_image_dir, train_mask_dir, "train", edge_generator=edge_generator)
    val_dataset = TamperDetectionNoPointsDataset(val_image_dir, val_mask_dir, edge_generator=edge_generator)

    num_workers = config['training']['num_workers']
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['val_batch_size'], shuffle=False, num_workers=num_workers)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    print("Starting training...")
    save_best_path = config['output']['save_best_path']
    trained_model, train_losses, val_losses = train_sam_ReVi(
        sam, train_loader, val_loader, edge_generator, num_epochs=config['training']['num_epochs'], lr=config['training']['learning_rate'], device=device, save_best_path = save_best_path
    )

    save_path = config['output']['save_final_path']
    torch.save(trained_model.state_dict(), save_path)
    print(f"Training completed. Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAM with LoRA using config file")
    parser.add_argument("--config", type=str, default="train_config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
