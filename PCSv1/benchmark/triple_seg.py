import os, sys, argparse, copy, math
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
    mask = mask.cpu()
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    annotated_pil = Image.fromarray(image).convert("RGBA")
    mask_pil = Image.fromarray((mask_image.numpy() * 255).astype(np.uint8)).convert("RGBA")
    return np.array(Image.alpha_composite(annotated_pil, mask_pil))

def apply_mask_to_dataframe(df, binary_mask, camera_pos, image_height, image_width):
    """
    Projects the 2D mask back to 3D using the SPECIFIC camera position for this angle.
    Includes Depth Buffering to prevent occlusion/shadow errors.
    """
    points = df[['X', 'Y', 'Z']].values
    
    # 1. Translate points relative to the CURRENT camera position
    translated = points - camera_pos

    # 2. Spherical Projection
    xy_dist = np.hypot(translated[:, 0], translated[:, 1])
    xy_dist[xy_dist == 0] = 1e-6 
    theta = np.arctan2(translated[:, 1], translated[:, 0])
    phi = np.arctan2(translated[:, 2], xy_dist)

    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi / 2) / np.pi
    x_px = (u * (image_width - 1)).astype(int)
    y_px = ((1 - v) * (image_height - 1)).astype(int)
    
    # 3. Calculate Depth (Distance from THIS camera)
    depths = np.linalg.norm(translated, axis=1)

    valid = (x_px >= 0) & (x_px < image_width) & (y_px >= 0) & (y_px < image_height)
    
    # --- VISIBILITY CHECK (Depth Buffer) ---
    # We need to filter out points that are "behind" the visible object.
    
    # Create a DataFrame to find the minimum depth per pixel quickly
    # (Using pandas groupby is cleaner than raw numpy loops for readability)
    temp_df = pd.DataFrame({
        'y': y_px[valid],
        'x': x_px[valid],
        'd': depths[valid]
    })
    
    # Find the closest point (min depth) for every pixel
    min_depths = temp_df.groupby(['y', 'x'])['d'].min().reset_index()
    
    # Create a 2D depth buffer
    depth_buffer = np.full((image_height, image_width), np.inf)
    depth_buffer[min_depths['y'], min_depths['x']] = min_depths['d']
    
    # Check 1: Is the point inside the mask?
    is_in_mask = np.zeros(len(points), dtype=bool)
    is_in_mask[valid] = binary_mask[y_px[valid], x_px[valid]]
    
    # Check 2: Is the point visible? (Depth is close to the buffer min)
    # 0.15 = 15cm tolerance. Adjust if pipes are very thick/thin.
    VISIBILITY_TOLERANCE = 0.15
    point_buffer_depth = depth_buffer[y_px[valid], x_px[valid]]
    
    is_visible = np.zeros(len(points), dtype=bool)
    is_visible[valid] = depths[valid] <= (point_buffer_depth + VISIBILITY_TOLERANCE)

    # Final Selection
    is_target = is_in_mask & is_visible
    
    count = np.sum(is_target)
    if count > 0:
        PIPE_CLASS_ID = 64
        print(f"  -> Marking {count} points as Class {PIPE_CLASS_ID}")
        df.loc[is_target, 'Classification'] = PIPE_CLASS_ID
    else:
        print("  -> No points matched (or all were occluded).")

    return df

def main(args):
    # 1. Load Data ONCE
    print(f"Loading Point Cloud Data: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv)
        df.columns = [c.strip().title() for c in df.columns]
        if 'Classification' not in df.columns: df['Classification'] = 0
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. Recalculate Geometry (Must match translation.py logic!)
    points = df[['X', 'Y', 'Z']].values
    centroid = np.mean(points, axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    robust_max_dist = np.percentile(dists, 90)
    orbit_radius = robust_max_dist * args.radius_mult
    
    print(f"Scene Centroid: {centroid}")
    print(f"Orbit Radius: {orbit_radius:.2f}")

    # 3. Process all 3 Angles
    angles = [0, 120, 240]
    
    # Determine base filename pattern
    # Assumes input_csv is "Lab1_data.csv", images are "Lab1_angle_0.png"
    base_name = os.path.basename(args.input_csv).replace("_data.csv", "")
    img_dir = args.image_dir

    for angle_deg in angles:
        print(f"\n--- Processing Angle {angle_deg} ---")
        
        # A. Construct Image Path
        img_filename = f"{base_name}_angle_{angle_deg}.png"
        img_path = os.path.join(img_dir, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue

        # B. Calculate Camera Position for this Angle
        angle_rad = math.radians(angle_deg)
        offset_x = orbit_radius * math.cos(angle_rad)
        offset_y = orbit_radius * math.sin(angle_rad)
        camera_pos = np.array([centroid[0] + offset_x, centroid[1] + offset_y, centroid[2]])

        # C. Detection (DINO)
        image_source, image = load_image(img_path)
        H, W, _ = image_source.shape

        boxes, logits, phrases = predict(
            model=groundingdino_model, image=image, caption=args.text_prompt,
            box_threshold=args.box_threshold, text_threshold=args.text_threshold, device=DEVICE
        )
        print(f"  Found {len(boxes)} objects.")

        if len(boxes) == 0:
            continue

        # D. Segmentation (SAM)
        sam_predictor.set_image(image_source)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
        
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False
        )

        # E. Combine Masks
        binary_mask_np = np.zeros((H, W), dtype=bool)
        if len(masks) > 0:
            combined_tensor = torch.any(masks, dim=0)
            binary_mask_np = combined_tensor[0].cpu().numpy()
            
            # Save visual check
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)[...,::-1]
            for i in range(len(masks)):
                annotated_frame = draw_mask(masks[i][0], annotated_frame)
            out_check = os.path.join(args.output_dir, f"{base_name}_angle_{angle_deg}_masked.png")
            Image.fromarray(annotated_frame).save(out_check)

        # F. Project to 3D & Update DataFrame
        df = apply_mask_to_dataframe(df, binary_mask_np, camera_pos, H, W)

    # 4. Save Final CSV
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    out_csv_name = f"{base_name}_labeled.csv"
    out_path = os.path.join(args.output_dir, out_csv_name)
    df.to_csv(out_path, index=False)
    print(f"\nSaved Final Labeled Data: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to the point cloud data CSV.")
    parser.add_argument("image_dir", help="Directory containing the angle images.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--text_prompt", default="pipe", help="What to detect.")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--radius_mult", type=float, default=0.5, help="Must match translation.py!")
    args = parser.parse_args()
    main(args)