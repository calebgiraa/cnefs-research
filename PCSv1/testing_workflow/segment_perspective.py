import os
import sys
import argparse

import numpy as np
import laspy
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "benchmark", "GroundingDINO"))

from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from GroundingDINO.groundingdino.models import build_model
from segment_anything import build_sam, SamPredictor
from huggingface_hub import hf_hub_download

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DEFAULT_SAM_CKPT = os.path.join(SCRIPT_DIR, "..", "benchmark", "model", "sam_vit_h_4b8939.pth")


def load_model_hf(repo_id, filename, ckpt_config_filename):
    cache_config = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    cfg = SLConfig.fromfile(cache_config)
    cfg.device = DEVICE
    model = build_model(cfg)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=DEVICE)
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model


def draw_mask(mask, image):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.cpu()
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    annotated_pil = Image.fromarray(image).convert("RGBA")
    mask_pil = Image.fromarray((mask_image.numpy() * 255).astype(np.uint8)).convert("RGBA")
    return np.array(Image.alpha_composite(annotated_pil, mask_pil))


def label_and_write_las(las_path, index_map, binary_mask, output_path):
    """
    Use index_map to back-project masked pixels directly to point indices.
    Writes LAS 1.4 / point format 6 so classification supports values up to 255.
    (Older point formats cap classification at 31, which can't hold class 64.)
    """
    iy, ix = np.where(binary_mask & (index_map >= 0))
    pipe_indices = np.unique(index_map[iy, ix])
    print(f"Labeling {len(pipe_indices):,} points as class 64...")

    src = laspy.read(las_path)

    # Upgrade to LAS 1.4 point format 6 (full uint8 classification, no 5-bit cap)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = src.header.offsets
    header.scales  = src.header.scales

    out = laspy.LasData(header=header)
    out.x = src.x
    out.y = src.y
    out.z = src.z

    classification = np.zeros(len(src.x), dtype=np.uint8)
    classification[pipe_indices] = 64
    out.classification = classification

    out.write(output_path)
    print(f"Saved labeled LAS : {output_path}")
    return len(pipe_indices)


def main():
    parser = argparse.ArgumentParser(
        description="Segment a perspective image and back-project labels into a LAS file."
    )
    parser.add_argument("input_image",  help="Perspective PNG from partial_translation.py.")
    parser.add_argument("index_map",    help=".npy index map from partial_translation.py.")
    parser.add_argument("input_las",    help="Original .las file used to generate the image.")
    parser.add_argument("output_dir",   help="Directory for labeled LAS and masked image.")
    parser.add_argument("--sam_checkpoint", default=DEFAULT_SAM_CKPT,
                        help="Path to SAM ViT-H checkpoint.")
    parser.add_argument("--text_prompt",    default="pipe")
    parser.add_argument("--box_threshold",  type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading GroundingDINO...")
    gdino = load_model_hf(
        "ShilongLiu/GroundingDINO",
        "groundingdino_swinb_cogcoor.pth",
        "GroundingDINO_SwinB.cfg.py",
    )
    print("Loading SAM...")
    sam_predictor = SamPredictor(build_sam(checkpoint=args.sam_checkpoint).to(DEVICE))

    index_map = np.load(args.index_map)

    # --- Detection ---
    image_source, image = load_image(args.input_image)
    H, W, _ = image_source.shape

    boxes, logits, phrases = predict(
        model=gdino, image=image, caption=args.text_prompt,
        box_threshold=args.box_threshold, text_threshold=args.text_threshold,
        device=DEVICE,
    )
    print(f"Detected {len(boxes)} object(s).")

    annotated_frame = annotate(
        image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
    )[..., ::-1]

    # --- Segmentation ---
    binary_mask = np.zeros((H, W), dtype=bool)
    if len(boxes) > 0:
        sam_predictor.set_image(image_source)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_source.shape[:2]
        ).to(DEVICE)

        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None, point_labels=None,
            boxes=transformed_boxes, multimask_output=False,
        )
        binary_mask = torch.any(masks, dim=0)[0].cpu().numpy()

        for i in range(len(masks)):
            annotated_frame = draw_mask(masks[i][0], annotated_frame)

    # --- Save masked visualization ---
    basename = os.path.splitext(os.path.basename(args.input_image))[0]
    out_img = os.path.join(args.output_dir, f"{basename}_masked.png")
    Image.fromarray(annotated_frame).save(out_img)
    print(f"Saved masked image : {out_img}")

    if not np.any(binary_mask):
        print("No pipe pixels in mask — check thresholds or text prompt.")
        sys.exit(1)

    # --- Back-project and write labeled LAS ---
    las_basename = os.path.splitext(os.path.basename(args.input_las))[0]
    out_las = os.path.join(args.output_dir, f"{las_basename}_labeled.las")
    label_and_write_las(args.input_las, index_map, binary_mask, out_las)


if __name__ == "__main__":
    main()
