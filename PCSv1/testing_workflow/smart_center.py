"""
smart_center.py — Optimized orbit center finder for multi-view point cloud segmentation.

Replaces the np.mean(points) centroid used in translation_centroid.py and triple_seg.py
with a center that maximizes how many unique surface points are visible from the three
orbit cameras (0°, 120°, 240°).

Two-stage strategy:
  1. Density-weighted centroid  — pulls the anchor toward geometrically complex regions
                                   (dense voxels = more structure = more to detect)
  2. Visibility-scored search   — samples candidate centers in a disk around the anchor,
                                   simulates orbit cameras at each, counts unique visible
                                   surface points, returns the best.

Usage:
    python smart_center.py input_data.csv --radius_mult 0.5
    python smart_center.py input_data.csv --radius_mult 0.5 --output center.npy
    python smart_center.py input_data.las  --radius_mult 0.5
"""

import numpy as np
import argparse
import os


# ---------------------------------------------------------------------------
# Stage 1: Density-weighted centroid
# ---------------------------------------------------------------------------

def density_weighted_centroid(points, grid_size=50):
    """
    Voxel-based density-weighted centroid (pure numpy, no scipy).

    Divides the XY footprint into a grid, counts points per cell, then weights
    each point by its cell's count. Dense cells (rich geometry) pull the result
    toward them more than sparse outlier regions do.
    """
    x, y = points[:, 0], points[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min + 1e-9
    y_range = y_max - y_min + 1e-9

    xi = np.floor((x - x_min) / x_range * grid_size).astype(int).clip(0, grid_size - 1)
    yi = np.floor((y - y_min) / y_range * grid_size).astype(int).clip(0, grid_size - 1)

    bin_idx = xi * grid_size + yi
    counts = np.bincount(bin_idx, minlength=grid_size * grid_size)

    weights = counts[bin_idx].astype(float)
    weights /= weights.sum()

    return np.average(points, axis=0, weights=weights)


# ---------------------------------------------------------------------------
# Stage 2: Visibility scoring
# ---------------------------------------------------------------------------

def count_visible_points(center, points, orbit_radius, orbit_angles_deg, ray_resolution=360):
    """
    Count how many unique surface points are visible from orbit cameras.

    Each camera position is orbit_radius away from center at a given angle in
    the XY plane (matching translation_centroid.py's camera placement exactly).

    Occlusion is simulated by binning rays into a spherical grid and keeping
    only the closest point per angular bin — a point is "visible" if nothing
    closer occludes it. A small depth tolerance allows near-surface neighbours
    to also count (realistic for discrete point clouds).
    """
    half_res = ray_resolution // 2
    n_bins = ray_resolution * half_res
    visible_flags = np.zeros(len(points), dtype=bool)

    for angle_deg in orbit_angles_deg:
        angle_rad = np.radians(angle_deg)
        cam_pos = center + np.array([
            orbit_radius * np.cos(angle_rad),
            orbit_radius * np.sin(angle_rad),
            0.0
        ])

        vecs = points - cam_pos
        dists = np.linalg.norm(vecs, axis=1)
        safe_dists = np.where(dists == 0, 1e-9, dists)
        norms = vecs / safe_dists[:, np.newaxis]

        theta = np.arctan2(norms[:, 1], norms[:, 0])           # [-pi,  pi]
        phi   = np.arctan2(norms[:, 2],                         # [-pi/2, pi/2]
                           np.hypot(norms[:, 0], norms[:, 1]))

        theta_bin = np.floor(
            (theta + np.pi) / (2 * np.pi) * ray_resolution
        ).astype(int).clip(0, ray_resolution - 1)

        phi_bin = np.floor(
            (phi + np.pi / 2) / np.pi * half_res
        ).astype(int).clip(0, half_res - 1)

        bin_idx = theta_bin * half_res + phi_bin

        # Scatter-reduce: minimum distance per angular bin
        min_dist_per_bin = np.full(n_bins, np.inf)
        np.minimum.at(min_dist_per_bin, bin_idx, dists)

        # Points within tolerance of the closest hit in their bin are "visible"
        DEPTH_TOLERANCE = 0.05
        is_visible = dists <= (min_dist_per_bin[bin_idx] + DEPTH_TOLERANCE)
        visible_flags |= is_visible

    return int(np.sum(visible_flags))


# ---------------------------------------------------------------------------
# Candidate search
# ---------------------------------------------------------------------------

def find_orbit_center(points, orbit_radius, orbit_angles_deg,
                      n_candidates=64, seed=42):
    """
    Search for the orbit center that maximises unique visible surface coverage.

    Candidates are sampled in a uniform disk in XY around the density-weighted
    centroid. Z is fixed to the density-weighted centroid's Z (cameras orbit
    in a horizontal plane, consistent with translation_centroid.py).

    Returns
    -------
    best_center  : (3,) ndarray
    best_score   : int   — visible point count at best_center
    geo_center   : (3,) ndarray — np.mean baseline
    geo_score    : int   — visible point count at geo_center
    """
    geo_center      = np.mean(points, axis=0)
    density_center  = density_weighted_centroid(points)

    search_radius = orbit_radius * 0.4

    rng = np.random.default_rng(seed)
    r     = search_radius * np.sqrt(rng.uniform(0, 1, n_candidates))
    theta = rng.uniform(0, 2 * np.pi, n_candidates)

    candidates = np.column_stack([
        density_center[0] + r * np.cos(theta),
        density_center[1] + r * np.sin(theta),
        np.full(n_candidates, density_center[2])
    ])
    # Prepend the two baselines so their indices are always 0 and 1
    candidates = np.vstack([geo_center, density_center, candidates])

    print(f"  Evaluating {len(candidates)} candidates "
          f"({len(orbit_angles_deg)} orbit angles each) ...")

    scores = []
    for i, cand in enumerate(candidates):
        score = count_visible_points(cand, points, orbit_radius, orbit_angles_deg)
        scores.append(score)

        if i == 0:
            print(f"    [baseline] geometric mean:      {score:,} visible pts  {cand}")
        elif i == 1:
            print(f"    [baseline] density-weighted:    {score:,} visible pts  {cand}")
        elif (i + 1) % 16 == 0:
            print(f"    [{i+1:3d}/{len(candidates)}] best so far: {max(scores):,} visible pts")

    scores     = np.array(scores)
    best_idx   = int(np.argmax(scores))

    return candidates[best_idx], scores[best_idx], geo_center, scores[0]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_points(path):
    """Load XYZ from a .las or a CSV that has X, Y, Z columns."""
    ext = os.path.splitext(path)[1].lower()

    if ext == '.las' or ext == '.laz':
        import laspy
        las = laspy.read(path)
        return np.vstack((las.x, las.y, las.z)).T

    # CSV path
    import pandas as pd
    df = pd.read_csv(path)
    df.columns = [c.strip().title() for c in df.columns]
    return df[['X', 'Y', 'Z']].values


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find an optimized orbit center for triple_seg / translation_centroid workflows."
    )
    parser.add_argument("input_file",
                        help="Path to point cloud (.las) or data CSV with X,Y,Z columns.")
    parser.add_argument("--radius_mult", type=float, default=0.5,
                        help="Orbit radius multiplier — must match the value used in "
                             "translation_centroid.py and triple_seg.py (default: 0.5).")
    parser.add_argument("--angles", type=int, nargs='+', default=[0, 120, 240],
                        help="Orbit angles in degrees to simulate (default: 0 120 240).")
    parser.add_argument("--n_candidates", type=int, default=64,
                        help="Number of random candidates to evaluate (default: 64).")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Randomly subsample to this many points before scoring. "
                             "Speeds up evaluation on large clouds (e.g. --subsample 100000).")
    parser.add_argument("--output", type=str, default=None,
                        help="Save the optimal center as a .npy file at this path.")
    args = parser.parse_args()

    print(f"Loading {args.input_file} ...")
    points = load_points(args.input_file)
    print(f"  {len(points):,} points loaded.")

    if args.subsample and args.subsample < len(points):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), args.subsample, replace=False)
        points_scored = points[idx]
        print(f"  Subsampled to {args.subsample:,} points for scoring.")
    else:
        points_scored = points

    # Compute orbit radius using the same formula as translation_centroid.py
    geo_center = np.mean(points_scored, axis=0)
    dists = np.linalg.norm(points_scored - geo_center, axis=1)
    robust_max_dist = np.percentile(dists, 90)
    orbit_radius = robust_max_dist * args.radius_mult

    print(f"\nOrbit radius : {orbit_radius:.4f} units  (90th-pct dist {robust_max_dist:.4f} x {args.radius_mult})")
    print(f"Orbit angles : {args.angles}")
    print(f"Geo center   : {geo_center}\n")

    best_center, best_score, geo_center, geo_score = find_orbit_center(
        points_scored, orbit_radius, args.angles,
        n_candidates=args.n_candidates
    )

    improvement = (best_score - geo_score) / max(geo_score, 1) * 100

    print(f"\n--- Results ---")
    print(f"  Geometric mean   : {geo_score:>10,} visible pts  {geo_center}")
    print(f"  Optimal center   : {best_score:>10,} visible pts  {best_center}")
    print(f"  Improvement      : {improvement:+.1f}%")

    if args.output:
        np.save(args.output, best_center)
        print(f"\nSaved to {args.output}")
    else:
        coords = best_center.tolist()
        print(f"\nTo use this center in translation_centroid.py / triple_seg.py,")
        print(f"replace the centroid line with:")
        print(f"  centroid = np.array({coords})")


if __name__ == "__main__":
    main()
