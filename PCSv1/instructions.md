# Point Cloud Processing Workflow

This document outlines the steps to set up your environment and run the point cloud processing workflow.

The process is two-stage:
- Translation: Convert a .las point cloud into a 2D top-down image (.png) and a data file (.csv).

- Segmentation: Run an object detection model (GroundingDINO + SAM) on the generated image to find and mask specific objects.

# Step 1: Conda Environment Setup

First, create and activate the conda environment using the provided requirements file.
Make sure you are in the project's root directory
<!-- ```conda env create --name PCSv1 --file requirements.txt
conda activate PCSv1``` -->

You will also need to download the model checkpoints. Based on segmentation.py, you need:

    GroundingDINO (will be downloaded by hf_hub_download on first run)

    SAM: Download sam_vit_h_4b8939.pth and place it in a model/ directory, or update the path in segmentation.py.

Step 2: Running the Workflow

Follow these steps in order.

Step 2.1: Generate Image and Data from Point Cloud

Use the translation.py script to create the 2D image and the XYZ-RGB data file.

Command:
Bash

python translation.py [path_to_input_las] [path_to_output_dir] --type ortho --res [resolution] --export_csv

Example: This command reads a .las file, creates a directory named output/, and saves both an image and a CSV file inside it.
Bash

# Create an output directory first
mkdir -p output

# Run the translation script
python translation.py ./input_lidar/Lab1_Scan_125.las ./output/ --type ortho --res 0.10 --export_csv

Expected Output: This will create two files in the output/ directory:

    output/Lab1_Scan_125_ortho.png (The 2D image)

    output/Lab1_Scan_125_data.csv (The XYZ-RGB data)

Step 2.2: Segment Objects in the Image

Next, use the segmentation.py script to find objects in the .png image you just created.

Command:
Bash

python segmentation.py [path_to_generated_image] [path_to_output_dir] --text_prompt "your . objects"

Example: This command reads the image from Step 2.1 and searches for "pipe" and "defect" objects, saving a new masked image.
Bash

python segmentation.py ./output/Lab1_Scan_125_ortho.png ./output/ --text_prompt "pipe . defect"

Expected Output: This will create a new, masked image in the output/ directory:

    output/Lab1_Scan_125_ortho_masked.png

Customizing Segmentation

You can adjust the segmentation.py script with these optional arguments:

    --text_prompt "..."

        Description: The objects to detect.

        Format: Separate multiple objects with . (a space, a dot, a space).

        Example: --text_prompt "person . car . traffic light"

    --box_threshold [number]

        Description: How confident the model must be to draw a box.

        Default: 0.35

        Tip: Lower this (e.g., 0.25) to find more objects, even if the model is less certain.

    --text_threshold [number]

        Description: How well the object must match the text prompt.

        Default: 0.28

        Tip: Lower this if the model is failing to label an object it clearly boxed.