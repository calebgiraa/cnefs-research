import os
import sys
import subprocess
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

# PyTorch and related imports
import torch
from torchvision.ops import box_convert

# Grounding DINO
# Must be imported after the setup_groundingdino() call
# which adds the directory to sys.path.

# segment anything
from segment_anything import build_sam, SamPredictor

# diffusers
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download

# supervision (for annotation)
import supervision as sv

def setup_groundingdino():
    """
    Checks for the GroundingDINO repository. If not found, it clones it.
    It then adds the repository to the Python path.
    """
    grounding_path = "GroundingDINO"
    
    if not os.path.exists(grounding_path):
        print("GroundingDINO repository not found. Cloning...")
        try:
            subprocess.run(["git", "clone", "https://github.com/IDEA-Research/GroundingDINO.git"], check=True)
            print("Cloning complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            sys.exit(1)
    else:
        print("GroundingDINO repository found.")

    sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

setup_groundingdino()

from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from GroundingDINO.groundingdino.util import box_ops

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    """Helper function to load the GroundingDINO model from Hugging Face."""
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} \n => {log}")
    model.eval()
    return model

def apply_mask_to_image(mask, image, color):
    """
    Helper function to apply a mask to an image with a specified color.
    The 'color' should be a NumPy array [R, G, B, Alpha].
    """
    h, w = mask.shape[-2:]
    mask = mask.cpu()
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def main(args):
    """Main execution function."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    print("Loading GroundingDINO model...")
    groundingdino_model = load_model_hf(
        "ShilongLiu/GroundingDINO",
        "groundingdino_swinb_cogcoor.pth",
        "GroundingDINO_SwinB.cfg.py",
        device=DEVICE
    )

    print("Loading Segment Anything Model (SAM)...")
    sam_checkpoint_path = "model/sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_checkpoint_path):
        print(f"SAM checkpoint not found at {sam_checkpoint_path}")
        sys.exit(1)
        
    sam = build_sam(checkpoint=sam_checkpoint_path)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    print(f"Loading image from: {args.input_image}")
    image_source, image = load_image(args.input_image)
    MAX_IMAGE_DIM = 1024
    H, W, _ = image_source.shape

    print(f"Detecting objects with prompt: '{args.text_prompt}'")
    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=DEVICE
    )

    print("Running segmentation on detected objects...")
    sam_predictor.set_image(image_source)
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
    
    masks, iou_pred, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True, # Keep True for multiple mask options
    )

    print("Annotating image with masks and bounding boxes...")
    
    # Start with the original image for applying masks
    annotated_frame_with_mask = Image.fromarray(image_source).convert("RGBA")
    annotated_frame_with_mask = np.array(annotated_frame_with_mask) # Convert back to numpy

    # Define color map for different objects
    color_map = {
        "pipe": np.array([30/255, 144/255, 255/255, 0.6]),      # Blue for pipe
        "defect": np.array([255/255, 0/255, 0/255, 0.7]),        # Red for defect
    }
    default_color = np.array([0/255, 255/255, 0/255, 0.6]) # Green for anything else


    detections_to_draw = []
    if boxes.shape[0] > 0:
        for i in range(boxes.shape[0]):
            phrase = phrases[i]
            
            # Select the best mask out of the 3 options using IoU prediction
            best_mask_idx = torch.argmax(iou_pred[i]).item()
            selected_mask = masks[i, best_mask_idx]
            
            detections_to_draw.append((phrase, i, selected_mask))

        def get_draw_order_priority(phrase):
            if "pipe" in phrase: return 0 # Draw pipe first (bottom layer)
            if "defect" in phrase: return 1 # Draw defect second (on top)
            return 99 # Anything else last

        detections_to_draw.sort(key=lambda x: get_draw_order_priority(x[0]))
        
        for phrase, original_box_idx, selected_mask in detections_to_draw:
            color = color_map.get(phrase, default_color)
            
            # Apply the mask with the chosen color
            annotated_frame_with_mask = apply_mask_to_image(selected_mask, annotated_frame_with_mask, color)
    else:
        print("No objects detected for the given prompt and thresholds.")
    
    # MODIFICATION: Apply bounding box annotation *after* masks have been drawn
    # The 'annotate' function expects a BGR image, so convert before passing.
    # It also outputs BGR, so convert back to RGB for saving.
    annotated_frame_with_mask_bgr = annotated_frame_with_mask[..., ::-1] # RGB to BGR
    final_annotated_image = annotate(
        image_source=annotated_frame_with_mask_bgr, 
        boxes=boxes, 
        logits=logits, 
        phrases=phrases
    )
    final_annotated_image = final_annotated_image[..., ::-1] # BGR to RGB


    os.makedirs(args.output_dir, exist_ok=True)
    
    base_name = os.path.basename(args.input_image)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(args.output_dir, f"{name}_multi_annotated_with_boxes.png") # New filename

    final_image_pil = Image.fromarray(final_annotated_image) # Use the image with boxes
    final_image_pil.save(output_path)
    
    print(f"✅ Annotated image with masks and bounding boxes saved successfully to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("GroundingDINO-SAM Multi-Object Segmentation")
    
    parser.add_argument("input_image", type=str, help="Path to the input image file.")
    parser.add_argument("output_dir", type=str, help="Directory where the annotated image will be saved.")
    
    parser.add_argument(
        "--text_prompt", type=str, default="pipe . defect",
        help="Text prompt for objects to detect, separated by ' . '."
    )
    parser.add_argument("--box_threshold", type=float, default=0.35, help="Box confidence threshold.")
    parser.add_argument("--text_threshold", type=float, default=0.28, help="Text confidence threshold.")
    
    args = parser.parse_args()
    main(args)