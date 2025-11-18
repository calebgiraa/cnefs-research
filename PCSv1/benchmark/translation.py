import numpy as np
import cv2
import laspy
import argparse
import os
import csv

def cloud_to_image(pcd_np, colors_np, resolution):
    """ Generates a top-down orthographic image. """
    minx, maxx = np.min(pcd_np[:, 0]), np.max(pcd_np[:, 0])
    miny, maxy = np.min(pcd_np[:, 1]), np.max(pcd_np[:, 1])
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    pixel_x = ((pcd_np[:, 0] - minx) / resolution).astype(int)
    pixel_y = ((maxy - pcd_np[:, 1]) / resolution).astype(int)
    
    valid_mask = (pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height)
    image[height - 1 - pixel_y[valid_mask], pixel_x[valid_mask]] = colors_np[valid_mask]
    
    return image

def generate_spherical_image(center_coordinates, point_cloud, colors, resolution_y=500):
    """ Generates a spherical (equirectangular) projection image. """
    resolution_x = 2 * resolution_y
    image = np.zeros((resolution_y, resolution_x, 3), dtype=np.uint8)
    depth_buffer = np.full((resolution_y, resolution_x), np.inf, dtype=float)

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
    
    for i in range(len(translated_points)):
        ix, iy = x_px[i], y_px[i]
        if 0 <= ix < resolution_x and 0 <= iy < resolution_y:
            if distances[i] < depth_buffer[iy, ix]:
                depth_buffer[iy, ix] = distances[i]
                image[iy, ix] = colors[i]
            
    return image, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input .las file.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--type", choices=['ortho', 'spherical'], default='ortho')
    parser.add_argument("--res", type=float, default=None)
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
        # Grayscale fallback
        z = points[:, 2]
        norm_z = (z - np.min(z)) / (np.max(z) - np.min(z))
        gray = (norm_z * 255).astype(np.uint8)
        raw_colors = np.vstack((gray, gray, gray)).transpose() # stored as 8-bit in this case
        image_colors = raw_colors

    # --- Classification ---
    if hasattr(las, 'classification'):
        classification = np.array(las.classification).reshape(-1, 1)
    else:
        classification = np.zeros((len(points), 1), dtype=np.uint8)

    basename = os.path.splitext(os.path.basename(args.input_file))[0]

    # --- CSV Export ---
    if args.export_csv:
        out_csv = os.path.join(args.output_dir, f"{basename}_data.csv")
        print(f"Exporting CSV to {out_csv}...")
        
        header = ['X', 'Y', 'Z', 'Red', 'Green', 'Blue', 'Classification']
        data = np.hstack((points, raw_colors, classification))
        
        try:
            with open(out_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)
        except Exception as e:
            print(f"CSV Error: {e}")

    # --- Image Gen ---
    img = None
    if args.type == 'ortho':
        img = cloud_to_image(points, image_colors, resolution)
    elif args.type == 'spherical':
        center = np.mean(points, axis=0)
        img, _ = generate_spherical_image(center, points, image_colors, int(resolution))

    if img is not None:
        out_img = os.path.join(args.output_dir, f"{basename}_{args.type}.png")
        cv2.imwrite(out_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved image to {out_img}")

if __name__ == "__main__":
    main()