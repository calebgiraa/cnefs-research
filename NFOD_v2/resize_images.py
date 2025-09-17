from PIL import Image
import os

def resize_images_in_directory(directory, size=(640, 640)):
    """
    Resizes all image files (JPG, PNG) in a given directory to the specified size.
    Images are resized in place, overwriting the originals.

    Args:
        directory (str): The path to the directory containing the images.
        size (tuple): A tuple (width, height) for the new image dimensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return

    print(f"Resizing images in: {directory} to {size[0]}x{size[1]}...")
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Check if it's a file and a common image extension
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Open the image
                img = Image.open(file_path)
                
                # Determine interpolation method
                # For masks (which are typically grayscale and have integer pixel values),
                # NEAREST neighbor is crucial to avoid creating new pixel values.
                # For regular images, BICUBIC or BILINEAR are good.
                # We'll use NEAREST for all to be safe, especially if this script is run for masks.
                interpolation_method = Image.Resampling.NEAREST
                
                # Resize the image
                resized_img = img.resize(size, interpolation_method)
                
                # Save the resized image, overwriting the original
                resized_img.save(file_path)
                print(f"  Resized: {filename}")
                
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
        else:
            print(f"  Skipping non-image file: {filename}")

if __name__ == '__main__':
    # Define the base directory where your 'images' and 'masks' folders are located
    # This assumes your script is run from 'NFOD_v2'.
    BASE_DATA_DIR = os.path.join('Datasets', 'pipe', 'Pipe-Segmentation-2')
    
    # Define the specific directories to resize
    image_dir_train = os.path.join(BASE_DATA_DIR, 'images', 'train')
    mask_dir_train = os.path.join(BASE_DATA_DIR, 'masks', 'train')

    # Add validation directories if they also contain images/masks you want to resize
    image_dir_valid = os.path.join(BASE_DATA_DIR, 'images', 'valid')
    mask_dir_valid = os.path.join(BASE_DATA_DIR, 'masks', 'valid')

    # Define the target size
    TARGET_SIZE = (640, 640) # (width, height)

    print("Starting image resizing process...")

    # Resize images and masks for the 'train' split
    resize_images_in_directory(image_dir_train, TARGET_SIZE)
    resize_images_in_directory(mask_dir_train, TARGET_SIZE)

    # Resize images and masks for the 'valid' split
    resize_images_in_directory(image_dir_valid, TARGET_SIZE)
    resize_images_in_directory(mask_dir_valid, TARGET_SIZE)

    print("\nImage resizing process complete.")