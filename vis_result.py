import os
import cv2
import torch
from tqdm import tqdm

def visualize_results(model, input_folder, output_folder, point=None, device='cuda'):
    """
    Visualize prediction results for all images in a folder

    Args:
        model: Trained model
        input_folder: Path to input image folder
        output_folder: Path to output result folder
        point: Optional point prompt; if None, no point prompt is used
        device: Device to run on
    """
    model.eval()

    # Ensure output folders exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "overlays"), exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    image_files = []
    for f in os.listdir(input_folder):
        if os.path.splitext(f)[1].lower() in image_extensions:
            image_files.append(f)

    if not image_files:
        print(f"No image files found in folder {input_folder}")
        return

    print(f"Found {len(image_files)} images, starting processing...")

    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            image_path = os.path.join(input_folder, img_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Cannot read image: {img_file}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image_rgb.shape[:2]  # Save original size
            image_resized = cv2.resize(image_rgb, (256, 256))
            image_normalized = image_resized
            if image_normalized.max() > 1:
                image_normalized = image_resized / 255.0

            # Convert to tensor
            image_tensor = torch.from_numpy(image_normalized).float()
            image_tensor = image_tensor.permute(2, 0, 1)  # From (H, W, C) to (C, H, W)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device=device)

            # Get image encoding
            with torch.no_grad():
                image_embeddings = model.image_encoder(image_tensor)

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

                single_channel_mask = torch.mean(masks, dim=1, keepdim=True)
                # single_channel_mask = torch.sigmoid(single_channel_mask)
                single_channel_mask = single_channel_mask.squeeze(0).squeeze(0)
                single_channel_mask = single_channel_mask.cpu().detach().numpy() * 255

                # Resize mask back to original size
                binary_mask_resized = cv2.resize(single_channel_mask, (original_size[1], original_size[0]))

                # Save results
                base_name = os.path.splitext(img_file)[0]

                # Save mask
                mask_path = os.path.join(output_folder, "masks", f"{base_name}_mask.png")
                cv2.imwrite(mask_path, binary_mask_resized)

        except Exception as e:
            print(f"Error processing image {img_file}: {str(e)}")

    print(f"Processing completed! Results saved in: {output_folder}")
    print(f"Masks saved in: {os.path.join(output_folder, 'masks')}")
    print(f"Overlays saved in: {os.path.join(output_folder, 'overlays')}")
