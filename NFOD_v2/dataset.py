import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class PipesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_filename = self.images[index]
        img_path = os.path.join(self.image_dir, img_filename)

        # CORRECTED LINE: Derive mask filename based on your mask generation script's convention
        # It replaces '.jpg' with '_mask.png'
        # Assuming all your original images are '.jpg'. If they can be other extensions like '.png',
        # you might need a more robust way to handle extensions, e.g., using os.path.splitext
        mask_filename = img_filename.replace('.jpg', '_mask.png')
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Error handling if mask file is not found
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path} for image {img_filename}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        # Ensure mask values are 0 or 1 for training
        mask[mask == 255.0] = 1.0 # This converts white (255) to 1 (foreground class)

        if self.transform is not None:
            # Assuming 'augmentations' is from Albumentations or a similar library
            # that expects image and mask as NumPy arrays and returns a dictionary.
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask