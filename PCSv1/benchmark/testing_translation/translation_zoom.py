import numpy as np
import cv2
import laspy
import argparse
import os
import csv
import math

"""
Currently looking at different translation methods that create three different images at various angles. This particular script seems
to have issues regarding it's zoom, image examples are too zoomed out.
"""

def cloud_to_image(pcd_np, colors_np, resolution):
    """ Generates a top-down orthographic image. """
    minx, maxx = np.min(pcd_np[:, 0]), np.max(pcd_np[:, 0])
    miny, maxy = np.min(pcd_np[:, 1]), np.max(pcd_np[:, 1])
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    pixel_x = ((pcd_np[:, 0] - minx) / resolution).astype(int)
    pixel_y = ((maxy - pcd_np[:, 1]) / resolution).astype(int)
    
    # Painter's Algorithm: Sort by Z 
    sort_idx = np.argsort(pcd_np[:, 2])
    pixel_x = pixel_x[sort_idx]
    pixel_y = pixel_y[sort_idx]
    colors_sorted = colors_np[sort_idx]

    valid_mask = (pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height)
    image[height - 1 - pixel_y[valid_mask], pixel_x[valid_mask]] = colors_sorted[valid_mask]
    
    return image

def generate_spherical_image(center_coordinates, point_cloud, colors, resolution_y=500):
    """ Generates a spherical (equirectangular) projection image. """
    resolution_x = 2 * resolution_y
    image = np.zeros((resolution_y, resolution_x, 3), dtype=np.uint8)

    translated_points = point_cloud - center_coordinates
    
    xy_dist = np.hypot(translated_points[:, 0], translated_points[:, 1])
    xy_dist[xy_dist == 0] = 1e-6 
    
    theta = np.arctan2(translated_points[:, 1], translated_points[:, 0])
    phi = np.arctan2(translated_points[:, 2], xy_dist)

    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi / 2) / np.pi
    x_px = (u * (resolution_x - 1)).astype(int)
    y_px = ((1 - v) * (resolution_y - 1)).astype(int)

    distances = np.linalg.norm(translated_points, axis=1)
    order = np.argsort(distances)[::-1] 

    x_px = x_px[order]
    y_px = y_px[order]
    colors_sorted = colors[order]

    valid = (x_px >= 0) & (x_px < resolution_x) & (y_px >= 0) & (y_px < resolution_y)
    image[y_px[valid], x_px[valid]] = colors_sorted[valid]
            
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input .las file.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--type", choices=['ortho', 'spherical'], default='ortho')
    parser.add_argument("--res", type=float, default=None)
    parser.add_argument("--radius_mult", type=float, default=1.0, help="Multiplier for camera distance from center.")
    parser.add_argument("--export_csv", action="store_true", help="Export data to CSV.")
    args = parser.parse_args()

    resolution = args.res
    if resolution is None:
        resolution = 0.1 if args.type == 'ortho' else 1000

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Loading {args.input_file}...")
    try:
        las = laspy.read(args.input_file)
    except Exception as e:
        print(f"Error reading LAS: {e}")
        return

    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # --- Colors ---
    if hasattr(las, 'red'):
        raw_colors = np.vstack((las.red, las.green, las.blue)).transpose()
        if np.max(las.red) > 255:
            image_colors = (raw_colors / 256).astype(np.uint8)
        else:
            image_colors = raw_colors.astype(np.uint8)
    else:
        z = points[:, 2]
        norm_z = (z - np.min(z)) / (np.max(z) - np.min(z))
        gray = (norm_z * 255).astype(np.uint8)
        image_colors = np.vstack((gray, gray, gray)).transpose()

    basename = os.path.splitext(os.path.basename(args.input_file))[0]

    # --- Logic ---
    if args.type == 'ortho':
        print("Generating Ortho image...")
        img = cloud_to_image(points, image_colors, resolution)
        out_img = os.path.join(args.output_dir, f"{basename}_ortho.png")
        cv2.imwrite(out_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved {out_img}")

    elif args.type == 'spherical':
        # 1. Find Center
        centroid = np.mean(points, axis=0)
        
        # 2. Determine Scale (Radius)
        # We find the furthest point from the center to know how big the object is.
        # This ensures the camera is placed *outside* the object.
        dists = np.linalg.norm(points - centroid, axis=1)
        max_dist = np.max(dists)
        
        # Multiply by user factor (default 1.0) to move closer or further
        orbit_radius = max_dist * args.radius_mult

        print(f"Object Radius detected: {max_dist:.2f}")
        print(f"Camera Orbital Radius: {orbit_radius:.2f}")

        # 3. Define 3 Angles (0, 120, 240 degrees)
        angles = [0, 120, 240]
        
        for angle_deg in angles:
            # Convert polar to cartesian (X, Y)
            angle_rad = math.radians(angle_deg)
            offset_x = orbit_radius * math.cos(angle_rad)
            offset_y = orbit_radius * math.sin(angle_rad)
            
            # Create Camera Position
            # We keep Z the same as the centroid (eye-level)
            camera_pos = centroid + np.array([offset_x, offset_y, 0])
            
            print(f"Rendering Angle {angle_deg}°...")
            img = generate_spherical_image(camera_pos, points, image_colors, int(resolution))
            
            out_img = os.path.join(args.output_dir, f"{basename}_angle_{angle_deg}.png")
            cv2.imwrite(out_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Saved {out_img}")

if __name__ == "__main__":
    main()