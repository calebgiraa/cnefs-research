import numpy as np
import cv2
import laspy
import argparse
import os
import csv
import math

# ------------------------------------------------------------
# Camera Utilities
# ------------------------------------------------------------

def look_at_rotation(camera_pos, target, up=np.array([0, 0, 1])):
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    # Camera space: X=forward, Y=right, Z=up
    return np.vstack((forward, right, true_up))


# ------------------------------------------------------------
# Equirectangular Projection
# ------------------------------------------------------------

def generate_equirectangular(camera_pos, target, points, colors, resolution_y):
    resolution_x = resolution_y * 2
    image = np.zeros((resolution_y, resolution_x, 3), dtype=np.uint8)
    # Track depth to ensure foreground points cover background points
    depth_buffer = np.full((resolution_y, resolution_x), np.inf)

    R = look_at_rotation(camera_pos, target)
    cam_points = (points - camera_pos) @ R.T

    x, y, z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
    valid = x > 0
    
    # Projection math
    theta = np.arctan2(y[valid], x[valid])
    phi = np.arctan2(z[valid], np.sqrt(x[valid]**2 + y[valid]**2))
    u = (theta + np.pi) / (2 * np.pi)
    v = (np.pi / 2 - phi) / np.pi

    px = (u * (resolution_x - 1)).astype(int)
    py = (v * (resolution_y - 1)).astype(int)
    dist = np.linalg.norm(cam_points[valid], axis=1)
    valid_colors = colors[valid]

    # Splatting Loop: Using radius=3 to ensure a solid surface
    for i in range(len(px)):
        ix, iy = px[i], py[i]
        if 0 <= ix < resolution_x and 0 <= iy < resolution_y:
            if dist[i] < depth_buffer[iy, ix]:
                # Drawing a circle 'splats' the point into a solid disk
                cv2.circle(image, (ix, iy), radius=3, color=valid_colors[i].tolist(), thickness=-1)
                depth_buffer[iy, ix] = dist[i]

    return image


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input .las file.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--res", type=int, default=1500, help="Vertical resolution (height). Width = 2x.")
    parser.add_argument("--radius_mult", type=float, default=0.5)
    parser.add_argument("--export_csv", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.input_file}...")
    las = laspy.read(args.input_file)

    points = np.vstack((las.x, las.y, las.z)).T

    # Colors
    if hasattr(las, "red"):
        colors = np.vstack((las.red, las.green, las.blue)).T
        if colors.max() > 255:
            colors = (colors / 256).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
    else:
        z = points[:, 2]
        gray = ((z - z.min()) / (z.ptp()) * 255).astype(np.uint8)
        colors = np.column_stack((gray, gray, gray))

    basename = os.path.splitext(os.path.basename(args.input_file))[0]

    # Optional CSV
    if args.export_csv:
        csv_path = os.path.join(args.output_dir, f"{basename}_data.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Y", "Z", "R", "G", "B"])
            writer.writerows(np.hstack((points, colors)))
        print(f"Saved CSV: {csv_path}")

    # Compute centroid and camera radius
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    radius = np.percentile(dists, 90) * args.radius_mult

    print(f"Centroid: {centroid}")
    print(f"Camera radius: {radius:.2f}")

    # Three viewpoints
    angles = [0, 120, 240]

    for angle in angles:
        rad = math.radians(angle)
        cam_pos = np.array([
            centroid[0] + radius * math.cos(rad),
            centroid[1] + radius * math.sin(rad),
            centroid[2]
        ])

        print(f"Rendering {angle}° view...")
        img_rgb = generate_equirectangular(
            camera_pos=cam_pos,
            target=centroid,
            points=points,
            colors=colors,
            resolution_y=args.res
        )

        # 1. Convert to BGR for OpenCV
        img_final = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 2. Morphological Closing: Fills in the remaining 'Swiss Cheese' pinholes
        # Using a 5x5 kernel for better gap filling
        kernel = np.ones((5, 5), np.uint8)
        img_final = cv2.morphologyEx(img_final, cv2.MORPH_CLOSE, kernel)

        # 3. Gaussian Blur: Softens the digital noise for the AI
        img_final = cv2.GaussianBlur(img_final, (3, 3), 0)

        out_path = os.path.join(args.output_dir, f"{basename}_equirect_{angle}.png")
        
        # 4. Save the processed BGR image
        cv2.imwrite(out_path, img_final)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()