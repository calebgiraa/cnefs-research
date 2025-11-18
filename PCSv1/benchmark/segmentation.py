import os, sys, argparse, copy
import pandas as pd
import cv2
import numpy as np
import torch
from PIL import Image

# Paths setup
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

# Libraries
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from GroundingDINO.groundingdino.models import build_model
from segment_anything import build_sam, SamPredictor 
from huggingface_hub import hf_hub_download

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- Model Loading ---
def load_model_hf(repo_id, filename, ckpt_config_filename):
    cache_config = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config) 
    args.device = DEVICE
    model = build_model(args)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=DEVICE)
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model   

groundingdino_model = load_model_hf("ShilongLiu/GroundingDINO", "groundingdino_swinb_cogcoor.pth", "GroundingDINO_SwinB.cfg.py")
sam_predictor = SamPredictor(build_sam(checkpoint='model/sam_vit_h_4b8939.pth').to(DEVICE))

def draw_mask(mask, image):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    
    # FIX: Move mask to CPU so it can interact with numpy
    mask = mask.cpu()
    
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    annotated_pil = Image.fromarray(image).convert("RGBA")
    
    # mask_image is already on CPU now, so .numpy() works
    mask_pil = Image.fromarray((mask_image.numpy() * 255).astype(np.uint8)).convert("RGBA")
    return np.array(Image.alpha_composite(annotated_pil, mask_pil))

def project_and_label_csv(csv_path, output_dir, binary_mask, image_height, image_width):
    print(f"Processing CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    df.columns = [c.strip().title() for c in df.columns]
    
    if 'Classification' not in df.columns: df['Classification'] = 0

    points = df[['X', 'Y', 'Z']].values
    center = np.mean(points, axis=0)
    translated = points - center

    # Spherical Projection (Must match translation.py)
    xy_dist = np.hypot(translated[:, 0], translated[:, 1])
    xy_dist[xy_dist == 0] = 1e-6 
    theta = np.arctan2(translated[:, 1], translated[:, 0])
    phi = np.arctan2(translated[:, 2], xy_dist)

    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi / 2) / np.pi
    x_px = (u * (image_width - 1)).astype(int)
    y_px = ((1 - v) * (image_height - 1)).astype(int)

    valid = (x_px >= 0) & (x_px < image_width) & (y_px >= 0) & (y_px < image_height)
    
    # Check mask intersection
    is_target = np.zeros(len(points), dtype=bool)
    
    # Only check points that project within image bounds
    # Using numpy advanced indexing
    is_target[valid] = binary_mask[y_px[valid], x_px[valid]]

    count = np.sum(is_target)
    if count > 0:
        PIPE_CLASS_ID = 64
        print(f"Labeling {count} points as Class {PIPE_CLASS_ID}...")
        df.loc[is_target, 'Classification'] = PIPE_CLASS_ID
        
        out_name = f"{os.path.splitext(os.path.basename(csv_path))[0]}_labeled.csv"
        out_path = os.path.join(output_dir, out_name)
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    else:
        print("No points overlapped with the detected masks.")

def main(args):
    # 1. Detection
    image_source, image = load_image(args.input_image)
    H, W, _ = image_source.shape

    boxes, logits, phrases = predict(
        model=groundingdino_model, image=image, caption=args.text_prompt,
        box_threshold=args.box_threshold, text_threshold=args.text_threshold, device=DEVICE
    )
    
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)[...,::-1]

    # 2. Segmentation
    sam_predictor.set_image(image_source)
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
    
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False
    )
    print(f"Detected {len(boxes)} objects.")

    # 3. Visualization & Mask Combination
    binary_mask_np = np.zeros((H, W), dtype=bool)
    
    if len(masks) > 0:
        # Combine all masks into one logical mask
        combined_tensor = torch.any(masks, dim=0)
        binary_mask_np = combined_tensor[0].cpu().numpy()
        
        # Draw masks for visual output
        for i in range(len(masks)):
            annotated_frame = draw_mask(masks[i][0], annotated_frame)

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save Image
    out_image_name = f"{os.path.splitext(os.path.basename(args.input_image))[0]}_masked.png"
    Image.fromarray(annotated_frame).save(os.path.join(args.output_dir, out_image_name))
    print(f"Saved masked image to {os.path.join(args.output_dir, out_image_name)}")

    # 4. CSV Labeling
    if args.input_csv and len(masks) > 0:
        project_and_label_csv(args.input_csv, args.output_dir, binary_mask_np, H, W)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", help="Spherical image.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--input_csv", help="CSV point cloud data.")
    parser.add_argument("--text_prompt", default="pipe", help="What to detect.")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    args = parser.parse_args()
    main(args)