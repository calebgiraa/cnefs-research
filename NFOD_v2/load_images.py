import numpy as np
from PIL import Image
import os

def generate_random_images(num_images, img_size=(128, 128), channels=3):
    """
    Generates a dataset of random images.

    Args:
        num_images (int): The number of images to generate.
        img_size (tuple): (height, width) of each image.
        channels (int): Number of color channels (e.g., 1 for grayscale, 3 for RGB).

    Returns:
        numpy.ndarray: A dataset of images with shape (num_images, height, width, channels).
                       Pixel values will be between 0 and 255 (simulating image data).
    """
    height, width = img_size
    data = np.random.randint(0, 256, size=(num_images, height, width, channels), dtype=np.uint8)
    return data

def save_images_to_directory(images, output_dir, prefix="random_image_", format="png"):
    """
    Saves a batch of images to a specified directory.

    Args:
        images (numpy.ndarray): The array of images (e.g., from generate_random_images).
                                Expected shape: (num_images, height, width, channels).
        output_dir (str): The path to the directory where images will be saved.
        prefix (str): A prefix for the filenames (e.g., "random_image_001.png").
        format (str): The image format (e.g., "png", "jpeg", "bmp").
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")

    for i, img_array in enumerate(images):
        # Convert NumPy array to PIL Image
        # If the image is grayscale (1 channel), PIL needs mode 'L'
        if img_array.shape[2] == 1:
            img = Image.fromarray(img_array.squeeze(), mode='L') # .squeeze() removes the channel dimension
        else:
            img = Image.fromarray(img_array, mode='RGB') # Assumes 3 channels for RGB

        # Define the filename
        filename = os.path.join(output_dir, f"{prefix}{i:04d}.{format}") # :04d ensures 4-digit numbering (e.g., 0001, 0002)

        # Save the image
        img.save(filename)
        print(f"Saved {filename}")

# --- How to run this ---

if __name__ == "__main__":
    # Define parameters for image generation
    num_images_to_generate = 50
    image_height = 128
    image_width = 128
    image_channels = 3 # 3 for RGB, 1 for grayscale

    # Define the output directory
    output_filepath = 'Datasets/Examples/'

    print(f"Generating {num_images_to_generate} random {image_height}x{image_width}x{image_channels} images...")
    random_images = generate_random_images(num_images_to_generate,
                                           img_size=(image_height, image_width),
                                           channels=image_channels)

    print(f"\nSaving images to: {output_filepath}")
    save_images_to_directory(random_images, output_filepath)

    print("\nImage generation and saving complete!")