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
    norm = np.linalg.norm(right)
    # Guard against degenerate case where forward is parallel to up
    if norm < 1e-6:
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        norm = np.linalg.norm(right)
    right /= norm

    true_up = np.cross(right, forward)

    return np.vstack((forward, right, true_up))


# ------------------------------------------------------------
# Equirectangular Projection
# ------------------------------------------------------------

def generate_equirectangular(camera_pos, target, points, colors, resolution_y,
                              point_radius=1):
    resolution_x = resolution_y * 2

    image = np.zeros((resolution_y, resolution_x, 3), dtype=np.uint8)

    # Transform to camera space
    R = look_at_rotation(camera_pos, target)
    cam_points = (points - camera_pos) @ R.T

    x, y, z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]

    # Keep all points (full 360° equirectangular, no behind-camera discard)
    theta = np.arctan2(y, x)        # yaw:   -pi .. pi
    phi = np.arctan2(z, np.sqrt(x**2 + y**2))  # pitch: -pi/2 .. pi/2

    u = (theta + np.pi) / (2 * np.pi)
    v = (np.pi / 2 - phi) / np.pi

    px = np.clip((u * (resolution_x - 1)).astype(int), 0, resolution_x - 1)
    py = np.clip((v * (resolution_y - 1)).astype(int), 0, resolution_y - 1)

    dist = np.linalg.norm(cam_points, axis=1)

    # Vectorised depth-buffer sort: paint far→near so near wins
    order = np.argsort(-dist)   # descending distance
    px, py, colors_sorted, dist_sorted = px[order], py[order], colors[order], dist[order]

    # Build image via sorted vectorised assignment (no Python loop)
    image[py, px] = colors_sorted

    # Optionally thicken points with a small dilation instead of per-point circles
    # (much faster than a Python loop with cv2.circle)
    if point_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (point_radius * 2 + 1, point_radius * 2 + 1))
        image = cv2.dilate(image, kernel, iterations=1)

    return image, R  # also return rotation so segmentation can reuse it


# ------------------------------------------------------------
# Save camera metadata for reprojection in segmentation.py
# ------------------------------------------------------------

def save_camera_meta(out_dir, basename, angle, camera_pos, R, resolution_y):
    """Save camera_pos and rotation matrix so segmentation.py can reproject correctly."""
    meta_path = os.path.join(out_dir, f"{basename}_cam_{angle}.npz")
    np.savez(meta_path, camera_pos=camera_pos, R=R,
             resolution_y=np.array([resolution_y]))
    return meta_path


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input .las file.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--res", type=int, default=1000,
                        help="Vertical resolution (height). Width = 2x.")
    parser.add_argument("--radius_mult", type=float, default=0.5)
    parser.add_argument("--point_radius", type=int, default=1,
                        help="Dilation radius in pixels to thicken sparse points.")
    parser.add_argument("--export_csv", action="store_true")
    parser.add_argument("--angles", type=str, default="0,120,240",
                        help="Comma-separated horizontal camera angles in degrees.")
    parser.add_argument("--elevations", type=str, default="0",
                        help="Comma-separated vertical elevation offsets as fraction "
                             "of point-cloud height, e.g. '-0.3,0,0.3'.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.input_file}...")
    las = laspy.read(args.input_file)

    points = np.vstack((las.x, las.y, las.z)).T

    # ---- Colours ----
    if hasattr(las, "red"):
        colors = np.vstack((las.red, las.green, las.blue)).T
        if colors.max() > 255:
            colors = (colors / 256).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
    else:
        z = points[:, 2]
        z_range = z.ptp()
        if z_range == 0:
            gray = np.full(len(z), 128, dtype=np.uint8)
        else:
            gray = ((z - z.min()) / z_range * 255).astype(np.uint8)
        colors = np.column_stack((gray, gray, gray))

    basename = os.path.splitext(os.path.basename(args.input_file))[0]

    # ---- Optional CSV export ----
    if args.export_csv:
        csv_path = os.path.join(args.output_dir, f"{basename}_data.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Y", "Z", "R", "G", "B"])
            writer.writerows(np.hstack((points, colors)))
        print(f"Saved CSV: {csv_path}")

    # ---- Camera geometry ----
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    radius = np.percentile(dists, 90) * args.radius_mult

    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    z_span = z_max - z_min

    print(f"Centroid: {centroid}")
    print(f"Camera radius: {radius:.2f}")

    angles = [float(a) for a in args.angles.split(",")]
    elevations = [float(e) for e in args.elevations.split(",")]

    for elev_frac in elevations:
        cam_z = centroid[2] + elev_frac * z_span
        for angle in angles:
            rad = math.radians(angle)
            cam_pos = np.array([
                centroid[0] + radius * math.cos(rad),
                centroid[1] + radius * math.sin(rad),
                cam_z
            ])

            label = f"{angle}deg_elev{elev_frac:+.2f}"
            print(f"Rendering {label} view...")

            img, R = generate_equirectangular(
                camera_pos=cam_pos,
                target=centroid,
                points=points,
                colors=colors,
                resolution_y=args.res,
                point_radius=args.point_radius,
            )

            # Post-processing
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 1. Close small gaps (small kernel to avoid blooming)
            kernel = np.ones((2, 2), np.uint8)
            img_bgr = cv2.morphologyEx(img_bgr, cv2.MORPH_CLOSE, kernel)

            # 2. CLAHE per channel – boosts local contrast so pipes stand out
            #    against similarly-coloured backgrounds (the main detection issue)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 3. Mild unsharp mask to sharpen edges for DINO
            blurred = cv2.GaussianBlur(img_bgr, (0, 0), 2)
            img_bgr = cv2.addWeighted(img_bgr, 1.5, blurred, -0.5, 0)

            out_path = os.path.join(args.output_dir,
                                    f"{basename}_equirect_{label}.png")
            cv2.imwrite(out_path, img_bgr)   # save the PROCESSED image
            print(f"Saved {out_path}")

            # Save camera metadata for accurate reprojection in segmentation.py
            meta_path = save_camera_meta(args.output_dir, basename, label,
                                         cam_pos, R, args.res)
            print(f"Saved camera meta: {meta_path}")


if __name__ == "__main__":
    main()