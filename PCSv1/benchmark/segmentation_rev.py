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

PIPE_CLASS_ID = 64


# -----------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------

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


groundingdino_model = load_model_hf(
    "ShilongLiu/GroundingDINO",
    "groundingdino_swinb_cogcoor.pth",
    "GroundingDINO_SwinB.cfg.py"
)
sam_predictor = SamPredictor(
    build_sam(checkpoint='model/sam_vit_h_4b8939.pth').to(DEVICE)
)


# -----------------------------------------------------------------------
# Mask overlay visualisation
# -----------------------------------------------------------------------

def draw_mask(mask, image):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.cpu()
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    annotated_pil = Image.fromarray(image).convert("RGBA")
    mask_pil = Image.fromarray(
        (mask_image.numpy() * 255).astype(np.uint8)
    ).convert("RGBA")
    return np.array(Image.alpha_composite(annotated_pil, mask_pil))


# -----------------------------------------------------------------------
# Reprojection using EXACT same camera transform as translation.py
# -----------------------------------------------------------------------

def reproject_points(points, camera_pos, R, image_height, image_width):
    """
    Reproject 3-D points into pixel coordinates using the saved camera
    position and rotation matrix from translation.py.

    Returns
    -------
    x_px, y_px : int arrays of shape (N,)
    valid      : bool mask – True for points that project inside the image
    """
    # 1. Transform to camera space (matches translation.py exactly)
    cam_points = (points - camera_pos) @ R.T  # (N, 3)

    x_c = cam_points[:, 0]
    y_c = cam_points[:, 1]
    z_c = cam_points[:, 2]

    # 2. Full-sphere equirectangular projection (no forward-only filter)
    theta = np.arctan2(y_c, x_c)
    phi = np.arctan2(z_c, np.sqrt(x_c ** 2 + y_c ** 2))

    u = (theta + np.pi) / (2 * np.pi)
    v = (np.pi / 2 - phi) / np.pi

    x_px = np.clip((u * (image_width - 1)).astype(int), 0, image_width - 1)
    y_px = np.clip((v * (image_height - 1)).astype(int), 0, image_height - 1)

    # All points project somewhere on the equirectangular sphere
    valid = np.ones(len(points), dtype=bool)
    return x_px, y_px, valid


# -----------------------------------------------------------------------
# CSV labelling
# -----------------------------------------------------------------------

def project_and_label_csv(csv_path, output_dir, binary_mask,
                           image_height, image_width,
                           camera_pos, R):
    print(f"Processing CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    df.columns = [c.strip().title() for c in df.columns]
    if 'Classification' not in df.columns:
        df['Classification'] = 0

    points = df[['X', 'Y', 'Z']].values

    x_px, y_px, valid = reproject_points(
        points, camera_pos, R, image_height, image_width
    )

    is_target = np.zeros(len(points), dtype=bool)
    is_target[valid] = binary_mask[y_px[valid], x_px[valid]]

    count = int(np.sum(is_target))
    print(f"  → {count} points overlap with detected mask(s).")

    if count > 0:
        df.loc[is_target, 'Classification'] = PIPE_CLASS_ID
        out_name = (f"{os.path.splitext(os.path.basename(csv_path))[0]}"
                    f"_labeled.csv")
        out_path = os.path.join(output_dir, out_name)
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    else:
        print("No points overlapped with the detected masks.")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main(args):
    # ---- Load camera metadata saved by translation.py ----
    if args.camera_meta:
        meta = np.load(args.camera_meta)
        camera_pos = meta['camera_pos']          # shape (3,)
        R = meta['R']                            # shape (3, 3)
        print(f"Loaded camera meta from {args.camera_meta}")
        print(f"  camera_pos = {camera_pos}")
    else:
        camera_pos = None
        R = None
        print("WARNING: No --camera_meta provided. "
              "CSV labelling will be skipped.")

    # ---- 1. Detection ----
    image_source, image = load_image(args.input_image)
    H, W, _ = image_source.shape

    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=DEVICE,
    )

    annotated_frame = annotate(
        image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
    )[..., ::-1]

    print(f"Detected {len(boxes)} objects with prompt '{args.text_prompt}'.")

    # ---- 2. Segmentation ----
    sam_predictor.set_image(image_source)
    boxes_xyxy = (box_ops.box_cxcywh_to_xyxy(boxes)
                  * torch.Tensor([W, H, W, H]))
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes_xyxy, image_source.shape[:2]
    ).to(DEVICE)

    binary_mask_np = np.zeros((H, W), dtype=bool)

    if len(boxes) > 0:
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # Combine all instance masks into one logical mask
        combined_tensor = torch.any(masks, dim=0)
        binary_mask_np = combined_tensor[0].cpu().numpy()

        for i in range(len(masks)):
            annotated_frame = draw_mask(masks[i][0], annotated_frame)
    else:
        print("No objects detected – skipping SAM segmentation.")

    # ---- 3. Save visualisation ----
    os.makedirs(args.output_dir, exist_ok=True)
    out_image_name = (f"{os.path.splitext(os.path.basename(args.input_image))[0]}"
                      f"_masked.png")
    out_image_path = os.path.join(args.output_dir, out_image_name)
    Image.fromarray(annotated_frame).save(out_image_path)
    print(f"Saved masked image: {out_image_path}")

    # ---- 4. CSV Labelling ----
    if args.input_csv and len(boxes) > 0:
        if camera_pos is not None and R is not None:
            project_and_label_csv(
                args.input_csv,
                args.output_dir,
                binary_mask_np,
                H, W,
                camera_pos,
                R,
            )
        else:
            print("Skipping CSV labelling – no camera metadata supplied.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", help="Equirectangular image from translation.py.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--input_csv",
                        help="CSV point cloud exported by translation.py.")
    parser.add_argument("--camera_meta",
                        help="Path to .npz camera metadata saved by translation.py "
                             "(e.g. basename_cam_0deg_elev+0.00.npz).")
    parser.add_argument("--text_prompt", default="pipe",
                        help="Grounding DINO text prompt.")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    args = parser.parse_args()
    main(args)