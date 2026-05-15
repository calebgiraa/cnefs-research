import os
import sys
import argparse
import gc

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


def write_labeled_las(src, pipe_mask, output_path):
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = src.header.offsets
    header.scales = src.header.scales
    out = laspy.LasData(header=header)
    out.x = src.x
    out.y = src.y
    out.z = src.z
    classification = np.zeros(len(src.x), dtype=np.uint8)
    classification[pipe_mask] = 64
    out.classification = classification
    out.write(output_path)
    print(f"Saved labeled LAS: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-view segmentation with vote fusion. Reads a _views.txt manifest from partial_translation.py."
    )
    parser.add_argument("views_manifest", help="Path to _views.txt manifest listing (image, index_map) pairs.")
    parser.add_argument("input_las",      help="Original .las file used to generate the views.")
    parser.add_argument("output_dir",     help="Directory for labeled LAS and visualizations.")
    parser.add_argument("--sam_checkpoint",  default=DEFAULT_SAM_CKPT)
    parser.add_argument("--text_prompt",     default="pipe")
    parser.add_argument("--box_threshold",   type=float, default=0.25)
    parser.add_argument("--text_threshold",  type=float, default=0.25)
    parser.add_argument("--fusion", choices=["or", "voting", "weighted"], default="voting",
                        help="Fusion method: 'or' (any view hits), 'voting' (requires min_votes), "
                             "'weighted' (accumulates detection confidence).")
    parser.add_argument("--min_votes",    type=int,   default=2,
                        help="Minimum views that must agree (--fusion voting).")
    parser.add_argument("--weight_thresh", type=float, default=0.7,
                        help="Minimum cumulative confidence (--fusion weighted).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse manifest (first line may be a # target header)
    views = []
    target_centroid = None
    cube_half = None
    with open(args.views_manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                parts = line.split()
                if len(parts) == 6 and parts[1] == 'target':
                    target_centroid = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                    cube_half = float(parts[5]) / 2.0
            else:
                parts = line.split()
                views.append((parts[0], parts[1]))

    if not views:
        print("ERROR: Manifest is empty.")
        sys.exit(1)

    print(f"Multi-view fusion: {len(views)} view(s), method={args.fusion.upper()}")

    # Load LAS once — used for accumulator sizing, spatial filter, and final write
    src_las = laspy.read(args.input_las)
    n_points = len(src_las.x)
    xyz = np.column_stack((src_las.x, src_las.y, src_las.z))

    vote_count = np.zeros(n_points, dtype=np.int32)
    weight_sum = np.zeros(n_points, dtype=np.float32)
    any_hit    = np.zeros(n_points, dtype=bool)

    # ---------------------------------------------------------
    # PHASE 1: Detection (GroundingDINO on all views)
    # ---------------------------------------------------------
    print("\n--- Phase 1: Detection ---")
    print("Loading GroundingDINO...")
    gdino = load_model_hf(
        "ShilongLiu/GroundingDINO",
        "groundingdino_swinb_cogcoor.pth",
        "GroundingDINO_SwinB.cfg.py",
    )

    detections = []
    for view_idx, (img_path, idx_map_path) in enumerate(views):
        print(f"  View {view_idx}: {os.path.basename(img_path)}")
        image_source, image = load_image(img_path)
        boxes, logits, phrases = predict(
            model=gdino, image=image, caption=args.text_prompt,
            box_threshold=args.box_threshold, text_threshold=args.text_threshold,
            device=DEVICE,
        )
        print(f"    Detected {len(boxes)} object(s).")

        # Save annotated detection image
        annotated = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)[..., ::-1]
        vis_path = os.path.join(args.output_dir, f"view_{view_idx}_detect.png")
        Image.fromarray(annotated).save(vis_path)

        detections.append({
            "img_path":    img_path,
            "idx_map_path": idx_map_path,
            "image_source": image_source,
            "boxes":       boxes,
            "logits":      logits,
            "max_logit":   logits.max().item() if len(logits) > 0 else 0.0,
        })

    print("Unloading GroundingDINO...")
    del gdino
    torch.cuda.empty_cache()
    gc.collect()

    active_views = [d for d in detections if len(d["boxes"]) > 0]
    if not active_views:
        print("No objects detected in any view — check thresholds or text prompt.")
        sys.exit(1)

    # ---------------------------------------------------------
    # PHASE 2: Segmentation + backprojection (SAM on detections)
    # ---------------------------------------------------------
    print("\n--- Phase 2: Segmentation & Fusion ---")
    print("Loading SAM...")
    sam_predictor = SamPredictor(build_sam(checkpoint=args.sam_checkpoint).to(DEVICE))

    for view_idx, det in enumerate(detections):
        if len(det["boxes"]) == 0:
            print(f"  View {view_idx}: skipped (no detections).")
            continue

        print(f"  View {view_idx}: segmenting...")
        image_source = det["image_source"]
        H, W, _ = image_source.shape
        boxes = det["boxes"]
        index_map = np.load(det["idx_map_path"])

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

        # Save masked visualization
        frame = image_source.copy()
        for m in masks:
            frame = draw_mask(m[0], frame)
        vis_masked = os.path.join(args.output_dir, f"view_{view_idx}_masked.png")
        Image.fromarray(frame).save(vis_masked)

        # Backproject via index_map (direct point indices — no spherical math needed)
        iy, ix = np.where(binary_mask & (index_map >= 0))
        hit_indices = np.unique(index_map[iy, ix])
        print(f"    Backprojected to {len(hit_indices):,} points.")

        if args.fusion == "or":
            any_hit[hit_indices] = True
        elif args.fusion == "voting":
            vote_count[hit_indices] += 1
        elif args.fusion == "weighted":
            weight_sum[hit_indices] += det["max_logit"]

    del sam_predictor
    torch.cuda.empty_cache()
    gc.collect()

    # ---------------------------------------------------------
    # FINAL: Resolve classifications
    # ---------------------------------------------------------
    print("\n--- Fusion Resolution ---")
    if args.fusion == "or":
        pipe_mask = any_hit
    elif args.fusion == "voting":
        pipe_mask = vote_count >= args.min_votes
        print(f"  Voting: {args.min_votes}+ views required.")
    elif args.fusion == "weighted":
        pipe_mask = weight_sum >= args.weight_thresh
        print(f"  Weighted: cumulative confidence >= {args.weight_thresh:.2f} required.")

    # Spatial filter: discard labeled points outside the target cube
    if target_centroid is not None:
        spatial_mask = np.all(
            (xyz >= target_centroid - cube_half) & (xyz <= target_centroid + cube_half),
            axis=1,
        )
        before = int(np.sum(pipe_mask))
        pipe_mask &= spatial_mask
        print(f"  Spatial filter: {before - int(np.sum(pipe_mask)):,} points removed outside target cube.")
    else:
        print("  Spatial filter: no target header found in manifest, skipping.")

    n_pipe = int(np.sum(pipe_mask))
    print(f"  Labeled {n_pipe:,} points as pipe (class 64).")

    if n_pipe == 0:
        print("WARNING: No pipe points after fusion — consider lowering thresholds or min_votes.")

    las_basename = os.path.splitext(os.path.basename(args.input_las))[0]
    out_las = os.path.join(args.output_dir, f"{las_basename}_labeled.las")
    write_labeled_las(src_las, pipe_mask, out_las)


if __name__ == "__main__":
    main()
