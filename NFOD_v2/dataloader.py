# import json

# def print_structure(d, indent=0):
#     # Print the strcture of a dictionary/list
    
#     if isinstance(d, dict):
#         for key, value in d.items():
#             print(' ' * indent + str(key))
#             print_structure(value, indent+1)
#     elif isinstance(d, list):
#         print(' ' * indent + "[List of length{} containing:]".format(len(d)))
#         if d:
#             print_structure(d[0], indent+1)

# print_structure(data)

# import os
# import random
# import json
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import cv2

# def display_images_with_coco_annotations(image_paths, annotations, display_type='both', colors=None):
#     fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
#     for ax, img_path in zip(axs.ravel(), image_paths):
#         # Load image using OpenCV and convert it from BGR to RGB color space
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         ax.imshow(image)
#         ax.axis('off')  # Turn off the axes

#         # Define a default color map if none is provided
#         if colors is None:
#             colors = plt.cm.get_cmap('tab10')

#         # Get image filename to match with annotations
#         img_filename = os.path.basename(img_path)
#         img_id = next(item for item in annotations['images'] if item["file_name"] == img_filename)['id']
        
#         # Filter annotations for the current image
#         img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == img_id]
        
#         for ann in img_annotations:
#             category_id = ann['category_id']
#             color = colors(category_id % 10)
            
#             # Display bounding box
#             if display_type in ['bbox', 'both']:
#                 bbox = ann['bbox']
#                 rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=color, facecolor='none')
#                 ax.add_patch(rect)
            
#             # Display segmentation polygon
#             if display_type in ['seg', 'both']:
#                 for seg in ann['segmentation']:
#                     poly = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
#                     polygon = patches.Polygon(poly, closed=True, edgecolor=color, fill=False)
#                     ax.add_patch(polygon)

#     plt.tight_layout()
#     plt.show()

# # Load COCO annotations
# with open(r'Datasets\pipe\Pipe-Segmentation-2\train\_annotations.coco.json', 'r') as file:
#     annotations = json.load(file)

# # Get all image files
# image_dir = "Datasets/pipe/Pipe-Segmentation-2/train"
# all_image_files = [os.path.join(image_dir, img['file_name']) for img in annotations['images']]
# random_image_files = random.sample(all_image_files, 4)

# # Choose between 'bbox', 'seg', or 'both'
# display_type = 'seg'
# display_images_with_coco_annotations(random_image_files, annotations, display_type)

#https://youtu.be/SQng3eIEw-k
"""

Convert coco json labels to labeled masks and copy original images to place 
them along with the masks. 

Dataset from: https://github.com/sartorius-research/LIVECell/tree/main
Note that the dataset comes with: 
Creative Commons Attribution - NonCommercial 4.0 International Public License
In summary, you are good to use it for research purposes but for commercial
use you need to investigate whether trained models using this data must also comply
with this license - it probably does apply to any derivative work so please be mindful. 

You can directly download from the source github page. Links below.

Training json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json
Validation json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json
Test json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json
Images: Download images.zip by following the link: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip

If these links do not work, follow the instructions on their github page. 


"""

import json
import numpy as np
import skimage.draw # Used for polygon drawing
from PIL import Image # Import Pillow for image handling
import os
import shutil


def create_mask(image_info, annotations, output_folder):
    # Create an empty mask as a numpy array.
    # Use dtype=np.uint8 as typical for grayscale images with classes 0-255.
    # If you have more than 255 classes, you'd need uint16, but PNG/JPEG might not support it directly.
    mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

    # Note: For semantic segmentation, objects of the same class typically get the same pixel value.
    # If 'object_number' is intended to differentiate instances (instance segmentation),
    # then this logic provides unique labels for each instance.
    # If it's pure semantic segmentation, you'd assign a class_id from a mapping.
    # Based on the previous context, your dataset.py was using class_idx (e.g., 1 for pipe).
    # This `create_mask` function appears to be generating an *instance-labeled* mask (each instance has a unique ID).
    # If you need semantic masks (e.g., all pipes are value 1), you'd modify `object_number` assignment.
    # For now, we'll keep the instance labeling as per your original code.
    object_number = 1 

    for ann in annotations:
        if ann['image_id'] == image_info['id']:
            # The 'segmentation' field in COCO can be polygons or RLE.
            # Your code assumes polygons (list of floats).
            # A common COCO polygon structure is [[x1, y1, x2, y2, ...]].
            # skimage.draw.polygon expects (row_coords, col_coords).
            # So, seg[1::2] for y (rows) and seg[0::2] for x (columns).
            for seg in ann['segmentation']: # This loop iterates through parts of a multi-polygon segmentation
                # Ensure seg is not empty or malformed before trying to draw
                if len(seg) >= 6: # A polygon needs at least 3 points (6 coordinates)
                    rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
                    mask_np[rr, cc] = 255
                else:
                    print(f"Warning: Malformed segmentation for annotation {ann.get('id', 'N/A')} in image {image_info['file_name']}. Skipping.")
            object_number += 1 # Increment object number for the next annotation in the image

    # Save the numpy array as a PNG using Pillow (PIL)
    # PNG is lossless and supports grayscale well. JPEG is lossy.
    mask_path = os.path.join(output_folder, image_info['file_name'].replace('.jpg', '_mask.png')) # Assuming original images are .jpg
    
    # Ensure the mask data type is suitable for PIL (typically uint8)
    # If object_number can exceed 255, you need to reconsider the mask_np dtype or scale.
    mask_image = Image.fromarray(mask_np.astype(np.uint8), mode='L') # 'L' for grayscale
    mask_image.save(mask_path)

    print(f"Saved mask for {image_info['file_name']} to {mask_path}")


def main(json_file, mask_output_folder, image_output_folder, original_image_dir):
    # Load COCO JSON annotations
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    # Ensure the output directories exist
    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    for img in images:
        # Create the masks
        create_mask(img, annotations, mask_output_folder)
        
        # Copy original images to the specified folder
        original_image_path = os.path.join(original_image_dir, img['file_name'])
        
        new_image_path = os.path.join(image_output_folder, os.path.basename(original_image_path))
        # Add a check if the original image exists before copying
        if os.path.exists(original_image_path):
            shutil.copy2(original_image_path, new_image_path)
            print(f"Copied original image to {new_image_path}")
        else:
            print(f"Warning: Original image not found at {original_image_path}. Skipping copy.")


if __name__ == '__main__':
    # These paths are relative to where you run the script (NFOD_v2)
    original_image_dir = 'Datasets/pipe/pvc-pipe-defect-2/test'
    json_file = 'Datasets/pipe/pvc-pipe-defect-2/test/_annotations.coco.json'
    mask_output_folder = 'Datasets/pipe/pvc-pipe-defect-2/masks/test'
    image_output_folder = 'Datasets/pipe/pvc-pipe-defect-2/images/test'
    
    print("Starting mask and image processing...")
    main(json_file, mask_output_folder, image_output_folder, original_image_dir)
    print("Processing complete.")

