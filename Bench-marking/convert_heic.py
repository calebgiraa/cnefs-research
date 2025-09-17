import os
import pyheif
from PIL import Image

input_dir = "/home/cahurd/CNEFS_Research/Bench-marking/stairwell_images"
output_dir = "/home/cahurd/CNEFS_Research/Bench-marking/stairwell_images"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.lower().endswith(".heic"):
        print(f"Converting: {file}")
        heif_file = pyheif.read(os.path.join(input_dir, file))
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        output_file = os.path.splitext(file)[0] + ".jpg"
        image.save(os.path.join(output_dir, output_file), "JPEG")
        print(f"Converted {file} to {output_file}")
