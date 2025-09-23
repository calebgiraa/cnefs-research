# Base Libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy
import argparse
import os

def cloud_to_image(pcd_np, resolution):
    """
    Generates a top-down orthographic image from a point cloud.
    Assumes the last 3 columns of pcd_np are RGB colors.
    """
    minx = np.min(pcd_np[:, 0])
    maxx = np.max(pcd_np[:, 0])
    miny = np.min(pcd_np[:, 1])
    maxy = np.max(pcd_np[:, 1])
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for point in pcd_np:
        x, y, *_ = point
        r, g, b = point[-3:]
        pixel_x = int((x - minx) / resolution)
        pixel_y = int((maxy - y) / resolution)
        if 0 <= pixel_y < height and 0 <= pixel_x < width:
            image[height - 1 - pixel_y, pixel_x] = [r, g, b]
    return image

def generate_spherical_image(center_coordinates, point_cloud, colors, resolution_y=500):
    """
    Generates a spherical (equirectangular) projection image from a point cloud.
    """
    resolution_x = 2 * resolution_y
    image = np.zeros((resolution_y, resolution_x, 3), dtype=np.uint8)
    depth_buffer = np.full((resolution_y, resolution_x), np.inf, dtype=float)

    translated_points = point_cloud - center_coordinates
    xy_dist = np.hypot(translated_points[:, 0], translated_points[:, 1])
    theta = np.arctan2(translated_points[:, 1], translated_points[:, 0])
    phi = np.arctan2(translated_points[:, 2], xy_dist)

    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi / 2) / np.pi
    x_px = (u * (resolution_x - 1)).astype(int)
    y_px = ((1 - v) * (resolution_y - 1)).astype(int)

    distances = np.linalg.norm(translated_points, axis=1)
    for i in range(len(translated_points)):
        ix, iy = x_px[i], y_px[i]
        if distances[i] < depth_buffer[iy, ix]:
            depth_buffer[iy, ix] = distances[i]
            image[iy, ix] = colors[i]
            
    return image, None

def main():
    parser = argparse.ArgumentParser(description="Generate an image from a .las or .laz point cloud file.")
    parser.add_argument("input_file", help="Path to the input .las or .laz file.")
    parser.add_argument("output_dir", help="Directory to save the output image.")
    parser.add_argument("--type", choices=['ortho', 'spherical'], default='ortho', help="Type of image to generate. Default is 'ortho'.")
    # ⭐️ FIX 1: Remove the single default and improve the help text.
    parser.add_argument("--res", type=float, default=None, help="Resolution. For 'ortho': meters/pixel (e.g., 0.1). For 'spherical': vertical pixels (e.g., 500).")
    args = parser.parse_args()

    # ⭐️ FIX 2: Set a smart, type-specific default resolution if none is provided.
    resolution = args.res
    if resolution is None:
        if args.type == 'ortho':
            resolution = 0.1
            print(f"No resolution provided. Using default for 'ortho': {resolution} m/pixel.")
        elif args.type == 'spherical':
            resolution = 500  # A sensible default for spherical images
            print(f"No resolution provided. Using default for 'spherical': {resolution} pixels.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    print(f"Loading point cloud from: {args.input_file}")
    las = laspy.read(args.input_file)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    if hasattr(las, 'red'):
        colors = (np.vstack((las.red, las.green, las.blue)).transpose() / 256).astype(np.uint8)
    else:
        print("Warning: No color data found. Generating grayscale image based on Z-value.")
        z = points[:, 2]
        norm_z = (z - np.min(z)) / (np.max(z) - np.min(z))
        gray_values = (norm_z * 255).astype(np.uint8)
        colors = np.vstack((gray_values, gray_values, gray_values)).transpose()

    image = None
    if args.type == 'ortho':
        # ⭐️ FIX 3: Use the new 'resolution' variable.
        print(f"Generating orthographic image with resolution {resolution} m/pixel...")
        pcd_with_colors = np.hstack((points, colors))
        image = cloud_to_image(pcd_with_colors, resolution=resolution)
    elif args.type == 'spherical':
        # ⭐️ FIX 3: Use the new 'resolution' variable.
        print(f"Generating spherical image with vertical resolution {int(resolution)} pixels...")
        center = np.mean(points, axis=0)
        image, _ = generate_spherical_image(center, points, colors, resolution_y=int(resolution))

    if image is not None:
        basename = os.path.splitext(os.path.basename(args.input_file))[0]
        output_filename = f"{basename}_{args.type}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
        print(f"✅ Successfully saved image to: {output_path}")
    else:
        print("Image generation failed.")

if __name__ == "__main__":
    main()