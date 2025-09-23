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
            # Using subprocess to run git clone
            subprocess.run(["git", "clone", "https://github.com/IDEA-Research/GroundingDINO.git"], check=True)
            print("Cloning complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            sys.exit(1)
    else:
        print("GroundingDINO repository found.")

    # Add GroundingDINO to the system path
    sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

# Call setup right away to make other imports work
setup_groundingdino()

# Now we can import from GroundingDINO
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

def show_mask(mask, image, random_color=True):
    """Helper function to apply a mask to an image."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    h, w = mask.shape[-2:]
    mask = mask.cpu()
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def main(args):
    """Main execution function."""
    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    DEVICE = torch.device('cpu')

    # --- Load Models ---
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
        print("Please download it and place it in the 'model' directory.")
        sys.exit(1)
        
    sam = build_sam(checkpoint=sam_checkpoint_path)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    # --- Load Image ---
    print(f"Loading image from: {args.input_image}")
    image_source, image = load_image(args.input_image)
    H, W, _ = image_source.shape

    # --- Run Detection ---
    print(f"Detecting objects with prompt: '{args.text_prompt}'")
    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=DEVICE
    )

    # --- Run Segmentation ---
    print("Running segmentation on detected objects...")
    sam_predictor.set_image(image_source)
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
    
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
    )

    # --- Annotate and Save Image ---
    print("Annotating image with masks...")
    # Using supervision to draw boxes
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[..., ::-1] # BGR to RGB

    annotated_frame_with_mask = annotated_frame.copy()
    for mask in masks:
        # We take the first mask from the multimask output
        annotated_frame_with_mask = show_mask(mask[0], annotated_frame_with_mask)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Construct output path
    base_name = os.path.basename(args.input_image)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(args.output_dir, f"{name}_annotated.png")

    # Save the final image
    final_image_pil = Image.fromarray(annotated_frame_with_mask)
    final_image_pil.save(output_path)
    
    print(f"✅ Annotated image saved successfully to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("GroundingDINO-SAM Object Detection and Segmentation")
    
    parser.add_argument(
        "input_image", type=str,
        help="Path to the input image file."
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Directory where the annotated image will be saved."
    )
    parser.add_argument(
        "--text_prompt", type=str, default="hole within pipe",
        help="Text prompt for the object to detect."
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.3,
        help="Box confidence threshold for detection."
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25,
        help="Text confidence threshold for detection."
    )
    
    args = parser.parse_args()
    main(args)