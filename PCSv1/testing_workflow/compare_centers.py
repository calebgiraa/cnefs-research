"""
compare_centers.py — Side-by-side image generation for orbit center comparison.

Generates two sets of equirectangular orbit images:
  geo_<basename>_angle_0.png   / 120 / 240  — original geometric mean center
  smart_<basename>_angle_0.png / 120 / 240  — optimized center from smart_center.py

Usage:
    python compare_centers.py input.las output_dir/ --radius_mult 0.5 --res 500
    python compare_centers.py input_data.csv output_dir/ --radius_mult 0.5
"""

import numpy as np
import cv2
import argparse
import math
import os

from smart_center import find_orbit_center, load_points


def generate_spherical_image(camera_pos, points, colors, resolution_y=500):
    """Equirectangular projection from camera_pos — identical to translation_centroid.py."""
    resolution_x = 2 * resolution_y
    image = np.zeros((resolution_y, resolution_x, 3), dtype=np.uint8)

    translated = points - camera_pos
    xy_dist = np.hypot(translated[:, 0], translated[:, 1])
    xy_dist[xy_dist == 0] = 1e-9

    theta = np.arctan2(translated[:, 1], translated[:, 0])
    phi   = np.arctan2(translated[:, 2], xy_dist)

    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi / 2) / np.pi
    x_px = (u * (resolution_x - 1)).astype(int)
    y_px = ((1 - v) * (resolution_y - 1)).astype(int)

    distances = np.linalg.norm(translated, axis=1)
    order = np.argsort(distances)[::-1]

    x_px_s   = x_px[order]
    y_px_s   = y_px[order]
    colors_s = colors[order]

    valid = (x_px_s >= 0) & (x_px_s < resolution_x) & \
            (y_px_s >= 0) & (y_px_s < resolution_y)
    image[y_px_s[valid], x_px_s[valid]] = colors_s[valid]

    return image


def render_set(label, center, points, colors, orbit_radius, angles_deg,
               resolution_y, output_dir, basename):
    """Render one full set of orbit images and return the saved paths."""
    saved = []
    for angle_deg in angles_deg:
        angle_rad = math.radians(angle_deg)
        cam_pos = np.array([
            center[0] + orbit_radius * math.cos(angle_rad),
            center[1] + orbit_radius * math.sin(angle_rad),
            center[2]
        ])
        print(f"  [{label}] angle {angle_deg:3d}°  camera @ {cam_pos}")
        img = generate_spherical_image(cam_pos, points, colors, resolution_y)
        out = os.path.join(output_dir, f"{label}_{basename}_angle_{angle_deg}.png")
        cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        saved.append(out)
    return saved


def load_las(path):
    import laspy
    las = laspy.read(path)
    points = np.vstack((las.x, las.y, las.z)).T
    if hasattr(las, 'red'):
        colors = np.vstack((las.red, las.green, las.blue)).T
        if colors.max() > 255:
            colors = (colors / 256).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
    else:
        z = points[:, 2]
        gray = ((z - z.min()) / (z.max() - z.min()) * 255).astype(np.uint8)
        colors = np.vstack((gray, gray, gray)).T
    return points, colors


def load_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    df.columns = [c.strip().title() for c in df.columns]
    points = df[['X', 'Y', 'Z']].values
    if {'Red', 'Green', 'Blue'}.issubset(df.columns):
        colors = df[['Red', 'Green', 'Blue']].values.astype(np.uint8)
    else:
        z = points[:, 2]
        gray = ((z - z.min()) / (z.max() - z.min()) * 255).astype(np.uint8)
        colors = np.vstack((gray, gray, gray)).T
    return points, colors


def main():
    parser = argparse.ArgumentParser(
        description="Generate geo-center vs smart-center orbit images for comparison."
    )
    parser.add_argument("input_file", help="Path to .las file or data CSV.")
    parser.add_argument("output_dir", help="Directory to write images into.")
    parser.add_argument("--radius_mult", type=float, default=0.5,
                        help="Orbit radius multiplier — match your triple_seg value (default: 0.5).")
    parser.add_argument("--res", type=int, default=500,
                        help="Vertical resolution of equirectangular images (default: 500).")
    parser.add_argument("--angles", type=int, nargs='+', default=[0, 120, 240],
                        help="Orbit angles in degrees (default: 0 120 240).")
    parser.add_argument("--n_candidates", type=int, default=64,
                        help="Candidate centers to evaluate in smart_center search (default: 64).")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N points for the smart_center scoring step only. "
                             "Full cloud is still used for rendering.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ext = os.path.splitext(args.input_file)[1].lower()
    basename = os.path.splitext(os.path.basename(args.input_file))[0]
    basename = basename.replace("_data", "")  # strip _data suffix if present

    print(f"Loading {args.input_file} ...")
    if ext in ('.las', '.laz'):
        points, colors = load_las(args.input_file)
    else:
        points, colors = load_csv(args.input_file)
    print(f"  {len(points):,} points loaded.")

    # Orbit radius — same formula as translation_centroid.py
    geo_center = np.mean(points, axis=0)
    dists = np.linalg.norm(points - geo_center, axis=1)
    robust_max_dist = np.percentile(dists, 90)
    orbit_radius = robust_max_dist * args.radius_mult

    print(f"\nOrbit radius : {orbit_radius:.4f} units")
    print(f"Geo center   : {geo_center}")

    # --- Smart center search (optionally on a subsample) ---
    if args.subsample and args.subsample < len(points):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), args.subsample, replace=False)
        points_scored = points[idx]
        print(f"\nSubsampled to {args.subsample:,} points for smart_center scoring.")
    else:
        points_scored = points

    print(f"\nFinding optimized center ...")
    smart_center, smart_score, _, geo_score = find_orbit_center(
        points_scored, orbit_radius, args.angles,
        n_candidates=args.n_candidates
    )
    improvement = (smart_score - geo_score) / max(geo_score, 1) * 100
    print(f"\n  Geo center   : {geo_score:,} visible pts  {geo_center}")
    print(f"  Smart center : {smart_score:,} visible pts  {smart_center}")
    print(f"  Improvement  : {improvement:+.1f}%")

    # --- Render both sets ---
    print(f"\nRendering geometric-mean images ...")
    geo_paths = render_set("geo", geo_center, points, colors,
                           orbit_radius, args.angles, args.res, args.output_dir, basename)

    print(f"\nRendering smart-center images ...")
    smart_paths = render_set("smart", smart_center, points, colors,
                             orbit_radius, args.angles, args.res, args.output_dir, basename)

    print(f"\n--- Saved images ---")
    for p in geo_paths + smart_paths:
        print(f"  {p}")
    print("\nDone. Compare geo_* vs smart_* images to evaluate the difference.")


if __name__ == "__main__":
    main()