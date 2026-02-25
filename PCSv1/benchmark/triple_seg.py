import os, sys, argparse, math
import pandas as pd
import numpy as np
import torch
import gc
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

DEVICE = torch.device('cpu')

# --- Model Loading Functions (Called only when needed) ---
def load_dino_model():
    print("-> Loading GroundingDINO to VRAM...")
    repo_id = "ShilongLiu/GroundingDINO"
    filename = "groundingdino_swinb_cogcoor.pth"
    config_filename = "GroundingDINO_SwinB.cfg.py"
    
    cache_config = hf_hub_download(repo_id=repo_id, filename=config_filename)
    args = SLConfig.fromfile(cache_config) 
    args.device = DEVICE
    model = build_model(args)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=DEVICE, weights_only=True)
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model   

def load_sam_model():
    print("-> Loading SAM ViT-H to VRAM...")
    sam = build_sam(checkpoint='model/sam_vit_h_4b8939.pth')
    sam.to(DEVICE)
    return SamPredictor(sam)

def draw_mask(mask, image):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.cpu()
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    annotated_pil = Image.fromarray(image).convert("RGBA")
    mask_pil = Image.fromarray((mask_image.numpy() * 255).astype(np.uint8)).convert("RGBA")
    return np.array(Image.alpha_composite(annotated_pil, mask_pil))

def get_visible_mask_points(points, binary_mask, camera_pos, image_height, image_width):
    translated = points - camera_pos

    xy_dist = np.hypot(translated[:, 0], translated[:, 1])
    xy_dist[xy_dist == 0] = 1e-6 
    theta = np.arctan2(translated[:, 1], translated[:, 0])
    phi = np.arctan2(translated[:, 2], xy_dist)

    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi / 2) / np.pi
    x_px = (u * (image_width - 1)).astype(int)
    y_px = ((1 - v) * (image_height - 1)).astype(int)
    
    depths = np.linalg.norm(translated, axis=1)
    valid = (x_px >= 0) & (x_px < image_width) & (y_px >= 0) & (y_px < image_height)
    
    temp_df = pd.DataFrame({'y': y_px[valid], 'x': x_px[valid], 'd': depths[valid]})
    min_depths = temp_df.groupby(['y', 'x'])['d'].min().reset_index()
    
    depth_buffer = np.full((image_height, image_width), np.inf)
    depth_buffer[min_depths['y'], min_depths['x']] = min_depths['d']
    
    is_in_mask = np.zeros(len(points), dtype=bool)
    is_in_mask[valid] = binary_mask[y_px[valid], x_px[valid]]
    
    VISIBILITY_TOLERANCE = 0.15
    point_buffer_depth = depth_buffer[y_px[valid], x_px[valid]]
    
    is_visible = np.zeros(len(points), dtype=bool)
    is_visible[valid] = depths[valid] <= (point_buffer_depth + VISIBILITY_TOLERANCE)

    return is_in_mask & is_visible

def main(args):
    print(f"Loading Point Cloud Data: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv)
        df.columns = [c.strip().title() for c in df.columns]
        if 'Classification' not in df.columns: df['Classification'] = 0
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    df['Vote_Count'] = 0
    df['Weight_Sum'] = 0.0

    points = df[['X', 'Y', 'Z']].values
    centroid = np.mean(points, axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    robust_max_dist = np.percentile(dists, 90)
    orbit_radius = robust_max_dist * args.radius_mult
    
    print(f"Fusion Method: {args.fusion.upper()}")
    angles = [0, 120, 240]
    base_name = os.path.basename(args.input_csv).replace("_data.csv", "")

    # ---------------------------------------------------------
    # PHASE 1: DETECTION (GroundingDINO Only)
    # ---------------------------------------------------------
    print("\n--- Phase 1: Object Detection ---")
    dino_model = load_dino_model()
    
    detections = [] # Store results to pass to SAM
    
    for angle_deg in angles:
        img_path = os.path.join(args.image_dir, f"{base_name}_angle_{angle_deg}.png")
        if not os.path.exists(img_path):
            continue

        image_source, image = load_image(img_path)
        boxes, logits, phrases = predict(
            model=dino_model, image=image, caption=args.text_prompt,
            box_threshold=args.box_threshold, text_threshold=args.text_threshold, device=DEVICE
        )
        print(f"  Angle {angle_deg}: Found {len(boxes)} objects.")

        if len(boxes) > 0:
            detections.append({
                'angle': angle_deg,
                'img_path': img_path,
                'boxes': boxes,
                'max_logit': logits.max().item()
            })

    # CLEANUP VRAM
    print("-> Unloading GroundingDINO to free VRAM...")
    del dino_model
    torch.cuda.empty_cache()
    gc.collect()

    if not detections:
        print("No objects found in any angle. Exiting.")
        return

    # ---------------------------------------------------------
    # PHASE 2: SEGMENTATION & FUSION (SAM Only)
    # ---------------------------------------------------------
    print("\n--- Phase 2: Segmentation & 3D Projection ---")
    sam_predictor = load_sam_model()

    for det in detections:
        angle_deg = det['angle']
        print(f"\n--- Segmenting Angle {angle_deg} ---")
        
        angle_rad = math.radians(angle_deg)
        camera_pos = np.array([
            centroid[0] + orbit_radius * math.cos(angle_rad), 
            centroid[1] + orbit_radius * math.sin(angle_rad), 
            centroid[2]
        ])

        image_source, _ = load_image(det['img_path'])
        H, W, _ = image_source.shape
        boxes = det['boxes']
        angle_weight = det['max_logit']

        # SAM Prediction
        sam_predictor.set_image(image_source)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
        masks, _, _ = sam_predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False)

        binary_mask_np = np.zeros((H, W), dtype=bool)
        if len(masks) > 0:
            binary_mask_np = torch.any(masks, dim=0)[0].cpu().numpy()

        # Apply to Point Cloud
        is_target = get_visible_mask_points(points, binary_mask_np, camera_pos, H, W)
        hits = np.sum(is_target)
        print(f"  -> Mask intersected with {hits} visible 3D points.")

        if args.fusion == 'or':
            df.loc[is_target, 'Classification'] = 64
        elif args.fusion == 'voting':
            df.loc[is_target, 'Vote_Count'] += 1
        elif args.fusion == 'weighted':
            df.loc[is_target, 'Weight_Sum'] += angle_weight

    # CLEANUP VRAM
    del sam_predictor
    torch.cuda.empty_cache()
    gc.collect()

    # ---------------------------------------------------------
    # FINAL CLASSIFICATION RESOLUTION
    # ---------------------------------------------------------
    print("\n--- Final Resolution ---")
    if args.fusion == 'voting':
        df.loc[df['Vote_Count'] >= args.min_votes, 'Classification'] = 64
        print(f"Applied Voting: Requires {args.min_votes}+ votes.")
    elif args.fusion == 'weighted':
        df.loc[df['Weight_Sum'] >= args.weight_thresh, 'Classification'] = 64
        print(f"Applied Weighted: Requires {args.weight_thresh} cumulative confidence.")

    total_pipes = np.sum(df['Classification'] == 64)
    print(f"Total Pipe Points Extracted: {total_pipes}")

    df = df.drop(columns=['Vote_Count', 'Weight_Sum'])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    out_path = os.path.join(args.output_dir, f"{base_name}_labeled.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved Final Labeled Data: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to the point cloud data CSV.")
    parser.add_argument("image_dir", help="Directory containing the angle images.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--text_prompt", default="pipe", help="What to detect.")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--radius_mult", type=float, default=0.5)
    
    parser.add_argument("--fusion", choices=['or', 'voting', 'weighted'], default='or', 
                        help="Method to combine masks: 'or' (any hit), 'voting' (requires N hits), 'weighted' (accumulates confidence).")
    parser.add_argument("--min_votes", type=int, default=2, help="Used with --fusion voting. Minimum angles needed.")
    parser.add_argument("--weight_thresh", type=float, default=0.7, help="Used with --fusion weighted. Minimum cumulative confidence.")
    
    args = parser.parse_args()
    main(args)