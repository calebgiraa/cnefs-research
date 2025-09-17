"""
This codebase was made with AI tools, and was inspired by the open-source repository for Grounded SAM.
(https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam_demo.py)
@author Caleb Gira
"""


import os, sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Keep this line if you want to explicitly set GPU 0

# All necessary imports
import argparse
import copy
import io

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv # This import is present but sv is not used in the provided code.

# segment anything
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers (present in original code, not used for the requested task)
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline


from huggingface_hub import hf_hub_download

"""
Loads model in from HuggingFace
"""
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device # Set device for the model args
    
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    # The original code had checkpoint = torch.load(cache_file, map_location='cpu') twice.
    # One is sufficient.
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval() # Set model to evaluation mode
    return model

def show_mask(mask, image, random_color = True):
    """
    Overlays a translucent mask on an image.
    Args:
        mask (torch.Tensor): A boolean mask tensor (H_mask, W_mask).
        image (np.array): The base image (RGB or RGBA) as a NumPy array (H_img, W_img, C).
        random_color (bool): If True, uses a random color for the mask.
    Returns:
        np.array: The image with the mask overlaid.
    """
    H_img, W_img, _ = image.shape # Get dimensions of the target image
    
    # Ensure mask is on CPU and convert to numpy (binary: 0 or 1)
    mask_np = mask.cpu().numpy().astype(np.uint8) 
    
    # Resize mask to match image dimensions if necessary
    if mask_np.shape != (H_img, W_img):
        # cv2.resize expects (width, height) for dsize
        mask_np = cv2.resize(mask_np, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
    
    if random_color:
        # Generate a random color with alpha for transparency (0-1 range for PIL)
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        # Fixed blue color with alpha for transparency
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    # Create an RGBA image from the resized mask and color
    mask_image = mask_np.reshape(H_img, W_img, 1) * color.reshape(1, 1, -1)

    # Convert base image and mask image to PIL RGBA for alpha compositing
    # Assume 'image' input is already RGB, convert to RGBA for compositing
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    # Scale mask_image to 0-255 before converting to uint8
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    # Composite the mask image over the annotated frame
    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def create_only_masked_image(masks, image_shape, random_color=True):
    """
    Creates an image showing only the segmentation masks on a black background.
    Args:
        masks (torch.Tensor): A batch of boolean mask tensors (N, 1, H_mask, W_mask).
        image_shape (tuple): The (H_img, W_img, C) shape of the original image to set dimensions.
        random_color (bool): If True, uses a random color for each mask.
    Returns:
        PIL.Image: The image showing only the masks.
    """
    H_img, W_img, _ = image_shape
    # Start with a black background image (RGB) matching original image dimensions
    masked_image_np = np.zeros((H_img, W_img, 3), dtype=np.uint8) # Default black background
    
    for i in range(masks.shape[0]):
        mask = masks[i][0] # Get the single boolean mask (H_mask, W_mask) from the batch dimension
        
        # Convert mask to numpy and resize to original image dimensions if necessary
        mask_np = mask.cpu().numpy().astype(np.uint8) # Binary (0 or 1)
        if mask_np.shape != (H_img, W_img):
            # cv2.resize expects (width, height) for dsize
            mask_np = cv2.resize(mask_np, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
        
        if random_color:
            # Generate a distinct random color for each mask (0-255 range for NumPy)
            color = (np.random.random(3) * 255).astype(np.uint8)
        else:
            # Fixed color for all masks
            color = np.array([30, 144, 255], dtype=np.uint8) # Example: RGB blue
        
        # Apply the mask: set pixels where mask is 1 (True) to the chosen color
        # Use `mask_np == 1` for explicit boolean indexing to ensure it works after resize
        masked_image_np[mask_np == 1] = color
        
    return Image.fromarray(masked_image_np) # Return as PIL Image

def main():
    parser = argparse.ArgumentParser(description="Grounded-Segment-Anything Image Exporter")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the directory containing input photos.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the directory where processed photos will be saved.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for object detection (e.g., 'pipe').")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="Box confidence threshold.")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text confidence threshold.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run models on (e.g., 'cpu' or 'cuda').")
    # Changed sam_checkpoint to not be required and set a default value
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth",
                        help="Path to SAM checkpoint file (e.g., 'sam_vit_h_4b8939.pth').")

    args = parser.parse_args()

    # --- Configuration for Grounding DINO Model (from your original code) ---
    # These are hardcoded as per your original script's structure.
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    # --- Load Grounding DINO Model (Once) ---
    print("Loading Grounding DINO model...")
    try:
        groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device=args.device)
        print("Grounding DINO model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load Grounding DINO model. Check paths and HuggingFace Hub access: {e}")
        return

    # --- Load SAM Model (Once) ---
    print("Loading SAM model...")
    try:
        sam = build_sam(checkpoint=args.sam_checkpoint)
        sam.to(device=args.device)
        sam_predictor = SamPredictor(sam)
        print("SAM model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load SAM model from '{args.sam_checkpoint}'. Check path and dependencies: {e}")
        return

    # --- Load Stable Diffusion Inpaint Pipeline (Once - from your original code) ---
    # This part is kept as per your request not to change the script,
    # but it's not directly used for the image export task.
    float_type = torch.float32 if args.device == 'cpu' else torch.float16
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=float_type
        )
        if args.device != 'cpu':
            pipe = pipe.to(args.device)
        print("Stable Diffusion Inpaint Pipeline loaded successfully (though not used for export).")
    except Exception as e:
        print(f"WARNING: Could not load Stable Diffusion Inpaint Pipeline. This might be expected if you only need detection/segmentation: {e}")
        # Continue execution even if SD fails, as it's not critical for the main task.

    # Create the output directory if it's doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory '{args.output_dir}' ensured.")

    # Get list of image files in the input directory
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    if not image_files:
        print(f"No image files found in '{args.input_dir}'. Supported formats: png, jpg, jpeg, bmp, tiff.")
        return

    print(f"Found {len(image_files)} image(s) in '{args.input_dir}'.")

    # --- Process Each Image ---
    for image_filename in image_files:
        input_image_path = os.path.join(args.input_dir, image_filename)
        base_filename_no_ext = os.path.splitext(image_filename)[0]

        print(f"\nProcessing '{image_filename}'...")

        try:
            # Load image for Grounding DINO inference
            # `load_image` from GroundingDINO.util.inference returns (image_source_np, image_tensor)
            image_source_np, image_tensor_for_dino = load_image(input_image_path)
            # image_source_np is the original image as a NumPy array (H, W, C) in RGB format

            # 1. Get Grounding DINO detections (bounding boxes, logits, phrases)
            boxes, logits, phrases = predict(
                model=groundingdino_model,
                image=image_tensor_for_dino, # Use the transformed image tensor for DINO
                caption=args.prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=args.device
            )

            # Convert to appropriate format for SAM
            H_orig, W_orig, _ = image_source_np.shape
            # Convert boxes from normalized [cx, cy, w, h] to pixel [x_min, y_min, x_max, y_max]
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W_orig, H_orig, W_orig, H_orig])
            
            if not boxes.size(0) > 0: # Check if any boxes were detected
                print(f"No objects detected for prompt '{args.prompt}' in '{image_filename}'. Skipping SAM segmentation and saving placeholder outputs.")
                
                # Save plain masked image (original without detections - now named _masked)
                masked_output_path = os.path.join(args.output_dir, f"{base_filename_no_ext}_masked.jpg")
                Image.fromarray(image_source_np).save(masked_output_path)
                print(f"Masked image (no detections) saved to '{masked_output_path}'")

                # Save empty convoluted image (all black - now named _convoluted)
                convoluted_output_path = os.path.join(args.output_dir, f"{base_filename_no_ext}_convoluted.jpg")
                empty_convoluted_img = Image.fromarray(np.zeros_like(image_source_np)) # All black
                empty_convoluted_img.save(convoluted_output_path)
                print(f"Convoluted image (empty) saved to '{convoluted_output_path}'")
                continue # Move to the next image

            # 2. Predict masks with SAM
            sam_predictor.set_image(image_source_np) # SAM expects RGB NumPy array
            
            # Apply SAM's internal transformation to the boxes and move to device
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source_np.shape[:2]).to(args.device)
            
            # Predict masks based on transformed boxes
            masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None, 
                boxes = transformed_boxes,
                multimask_output = True, 
            )
            # masks shape: (num_boxes, num_masks_per_box, H, W) - boolean masks

            # --- GENERATE AND SAVE MASKED IMAGE (_masked.jpg) ---
            # This image is the original with annotations (boxes, labels) and translucent masks overlaid.
            annotated_frame = annotate(image_source=image_source_np, boxes=boxes, logits=logits, phrases=phrases)
            annotated_frame = annotated_frame[...,::-1] # BGR to RGB if annotate returns BGR, otherwise RGB is fine

            # Overlay masks on the annotated frame
            annotated_frame_with_mask = annotated_frame.copy()
            for i in range(masks.shape[0]):
                mask_for_overlay = masks[i][0] # Select the first mask for this object
                annotated_frame_with_mask = show_mask(mask_for_overlay, annotated_frame_with_mask)
            
            # Save the masked image - CONVERTING TO RGB BEFORE SAVING AS JPEG
            masked_output_path = os.path.join(args.output_dir, f"{base_filename_no_ext}_masked.jpg")
            Image.fromarray(annotated_frame_with_mask).convert("RGB").save(masked_output_path)
            print(f"Masked image saved to '{masked_output_path}'")

            # --- GENERATE AND SAVE CONVOLUTED IMAGE (_convoluted.jpg) ---
            # This image is a combined binary mask (white on black) of all detected objects.
            # Initialize an empty combined mask with the same dimensions as the image
            # and the same dtype as the individual masks (boolean or uint8)
            # Use masks[0][0].shape for dimensions as it's a representative mask shape
            combined_mask = np.zeros_like(masks[0][0].cpu().numpy(), dtype=np.uint8) # Start with black
            
            for mask in masks:
                # Get the first mask from the multi-mask output for the current object
                current_mask_np = mask[0].cpu().numpy()
                
                # Resize current_mask_np to match the original image dimensions if necessary
                if current_mask_np.shape != (H_orig, W_orig):
                    current_mask_np = cv2.resize(current_mask_np.astype(np.uint8), (W_orig, H_orig), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    current_mask_np = current_mask_np.astype(bool) # Ensure it's boolean for logical_or

                # Combine masks using logical OR operation
                combined_mask = np.logical_or(combined_mask, current_mask_np)
            
            # Convert the combined boolean mask to a 0-255 uint8 image (white mask on black background)
            convoluted_output_image_pil = Image.fromarray((combined_mask * 255).astype(np.uint8))
            
            convoluted_output_path = os.path.join(args.output_dir, f"{base_filename_no_ext}_convoluted.jpg")
            convoluted_output_image_pil.save(convoluted_output_path)
            print(f"Convoluted image saved to '{convoluted_output_path}'")

        except Exception as e:
            print(f"ERROR: Failed to process '{image_filename}': {e}")
            # Continue to the next image even if one fails

    print("\nProcessing complete for all images!")

if __name__ == "__main__":
    main()
