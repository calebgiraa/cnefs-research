import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.ops import box_convert

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === Add your helper functions to match grounded_sam_demo.py ===

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

# === Make sure Python can find GroundingDINO & your local SAM ===
sys.path.append("./GroundingDINO")
sys.path.append("./segment_anything")

from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T

from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor import SamPredictor

print("✅ GroundingDINO + SAM imports loaded correctly!")

# === CONFIG ===
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
GROUNDING_DINO_CHECKPOINT = "models/groundingdino_swinb_cogcoor.pth"
SAM_CHECKPOINT = "models/sam_vit_h_4b8939.pth"
TEXT_PROMPT = "pipe"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
DEVICE = "cpu"

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load GroundingDINO ===
print("✅ Loading GroundingDINO...")
dino_model = load_model(CONFIG_PATH, GROUNDING_DINO_CHECKPOINT)
dino_model.to(DEVICE)

# === Load SAM ===
print("✅ Loading SAM...")
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# === Image transforms ===
transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# === Process all images ===
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print(f"🔍 Found {len(image_files)} images in {INPUT_DIR}.")

for idx, filename in enumerate(image_files, 1):
    print(f"\n[{idx}/{len(image_files)}] Processing: {filename}")

    # Load & prepare image
    image_bgr = cv2.imread(os.path.join(INPUT_DIR, filename))
    if image_bgr is None:
        print(f"⚠️ Could not read {filename} — skipping.")
        continue

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tensor, _ = transform(image_pil, None)

    # === Run GroundingDINO ===
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_tensor,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    if len(boxes) == 0:
        print(f"❌ No detections for {filename}")
        continue

    # === Convert boxes to pixel coordinates ===
    W, H = image_pil.size
    boxes = boxes * torch.tensor([W, H, W, H])
    boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # === Run SAM on each box ===
    predictor.set_image(image_rgb)
    masks = []
    for box in boxes_xyxy:
        mask, _, _ = predictor.predict(box=box, multimask_output=False)
        mask = np.squeeze(mask)
        masks.append(mask)

    # === Use Matplotlib for mask & box overlays ===
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)

    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)

    for box, label in zip(boxes_xyxy, phrases):
        show_box(box, plt.gca(), label)

    plt.axis('off')
    result_filename = f"result_{filename}"
    plt.savefig(
        os.path.join(OUTPUT_DIR, result_filename),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    plt.close()
    print(f"✅ Saved overlay: {result_filename}")

    # === Also save black & white convoluted mask ===
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask)

    bw_mask = np.where(combined_mask, 255, 0).astype(np.uint8)
    bw_filename = f"bw_mask_{filename}"
    cv2.imwrite(os.path.join(OUTPUT_DIR, bw_filename), bw_mask)
    print(f"✅ Saved BW mask: {bw_filename}")

print("\n✅✅✅ All images processed! Results saved in:", OUTPUT_DIR)
