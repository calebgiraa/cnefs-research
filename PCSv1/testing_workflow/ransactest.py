"""
ransactest.py  (testing_workflow version)

RANSAC cylinder fitting on class-64 labeled LAS files.

Pipeline:
  1. Load LAS, filter to class 64 (pipe) points only.
  2. Voxel downsample.
  3. DBSCAN — select largest cluster.
  4. Visualize cluster in Open3D.
  5. RANSAC cylinder fit → radius, diameter, orientation.

Usage:
    python ransactest.py --file path/to/labeled.las
    python ransactest.py --file labeled.las --epsilon 0.5 --ransac-thresh 0.01
"""

import argparse
import sys
import time

import laspy
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_VOXEL_SIZE    = 0.01
DEFAULT_EPSILON       = 0.08
DEFAULT_MIN_POINTS    = 100
DEFAULT_RANSAC_THRESH = 0.005
DEFAULT_RANSAC_ITERS  = 1000
DEFAULT_FIT_SAMPLES   = 8_000
TARGET_CLASS          = 64


# ---------------------------------------------------------------------------
# Step 1 — Load and filter to class 64
# ---------------------------------------------------------------------------

def load_point_cloud(file_path: str) -> o3d.geometry.PointCloud:
    print(f"\n[1/4] Loading '{file_path}' ...")
    t0 = time.perf_counter()

    las = laspy.read(file_path)
    classifications = np.asarray(las.classification)

    pipe_mask = classifications == TARGET_CLASS
    n_pipe = pipe_mask.sum()

    if n_pipe == 0:
        print(f"      No points with class {TARGET_CLASS} found.")
        print("      Check that segment_perspective.py ran correctly and wrote the LAS.")
        sys.exit(1)

    xyz = np.column_stack((las.x, las.y, las.z))[pipe_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color([0.2, 0.6, 1.0])

    print(f"      Total points in file : {len(las.x):,}")
    print(f"      Class {TARGET_CLASS} points     : {n_pipe:,}")
    print(f"      Loaded in {time.perf_counter()-t0:.2f}s.")
    return pcd


# ---------------------------------------------------------------------------
# Step 2 — Downsample
# ---------------------------------------------------------------------------

def downsample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    print(f"\n[2/4] Voxel downsampling (voxel_size={voxel_size} m) ...")
    t0 = time.perf_counter()

    pcd_down = pcd.voxel_down_sample(voxel_size)

    ratio = len(pcd_down.points) / len(pcd.points) * 100
    print(f"      {len(pcd.points):,} → {len(pcd_down.points):,} points "
          f"({ratio:.1f}% retained) in {time.perf_counter()-t0:.2f}s.")
    return pcd_down


# ---------------------------------------------------------------------------
# Step 3 — DBSCAN: largest cluster
# ---------------------------------------------------------------------------

def find_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    epsilon: float,
    min_points: int,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    print(f"\n[3/4] DBSCAN clustering (eps={epsilon}, min_points={min_points}) ...")
    t0 = time.perf_counter()

    labels = np.array(
        pcd.cluster_dbscan(eps=epsilon, min_points=min_points, print_progress=True)
    )
    print(f"      Finished in {time.perf_counter()-t0:.2f}s.")

    if labels.max() < 0:
        print(
            "\n      DBSCAN found NO clusters — all points classified as noise.\n"
            "      Try a larger --epsilon (e.g. 0.5, 5.0, 50.0) to match your\n"
            "      point cloud's coordinate units."
        )
        sys.exit(1)

    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    print(f"      Found {len(unique_labels):,} cluster(s) + noise.")

    largest_label = unique_labels[np.argmax(counts)]
    indices = np.where(labels == largest_label)[0]
    cluster_pcd = pcd.select_by_index(indices)
    cluster_pts = np.asarray(cluster_pcd.points)

    print(f"      Largest cluster: {len(cluster_pts):,} points.")
    return cluster_pcd, cluster_pts


# ---------------------------------------------------------------------------
# Step 4 — Visualize cluster
# ---------------------------------------------------------------------------

def visualize_cluster(full_pcd: o3d.geometry.PointCloud,
                      cluster_pcd: o3d.geometry.PointCloud) -> None:
    print("\n      Opening Open3D viewer (Q or Esc to close and continue) ...")
    full_display = o3d.geometry.PointCloud(full_pcd)
    full_display.paint_uniform_color([0.6, 0.6, 0.6])

    cluster_display = o3d.geometry.PointCloud(cluster_pcd)
    cluster_display.paint_uniform_color([1.0, 0.2, 0.2])

    o3d.visualization.draw_geometries(
        [full_display, cluster_display],
        window_name="RANSAC: full labeled cloud (grey) + selected cluster (red)",
        width=1280, height=720,
    )


# ---------------------------------------------------------------------------
# Step 5 — RANSAC cylinder fit
# ---------------------------------------------------------------------------

def _point_to_cylinder_distance(pts, axis_pt, axis_dir, radius):
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    diff = pts - axis_pt
    proj = diff - np.outer(diff @ axis_dir, axis_dir)
    return np.abs(np.linalg.norm(proj, axis=1) - radius)


def _fit_cylinder_from_sample(pts):
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    axis_dir = Vt[0] / np.linalg.norm(Vt[0])
    proj = centred - np.outer(centred @ axis_dir, axis_dir)
    radius = float(np.median(np.linalg.norm(proj, axis=1)))
    if radius < 1e-9:
        return None
    return centroid, axis_dir, radius


def fit_cylinder(points, thresh, n_iterations, fit_samples):
    print(f"\n[4/4] Fitting RANSAC cylinder ({len(points):,} points) ...")

    if len(points) > fit_samples:
        rng = np.random.default_rng(42)
        fit_pts = points[rng.choice(len(points), size=fit_samples, replace=False)]
        print(f"      Subsampled to {fit_samples:,} points for fitting.")
    else:
        fit_pts = points

    print(f"      Iterations: {n_iterations}   Threshold: {thresh} m")

    t0 = time.perf_counter()
    best_count = -1
    best_model = None
    rng = np.random.default_rng(0)
    n = len(fit_pts)

    for i in range(n_iterations):
        sample = fit_pts[rng.choice(n, size=10, replace=False)]
        model = _fit_cylinder_from_sample(sample)
        if model is None:
            continue
        dists = _point_to_cylinder_distance(fit_pts, *model)
        count = int((dists < thresh).sum())
        if count > best_count:
            best_count = count
            best_model = model
        if (i + 1) % 100 == 0:
            print(f"      iter {i+1:4d}/{n_iterations}  best inliers: {best_count:,}", flush=True)

    if best_model is None:
        print("\n      RANSAC failed — try --ransac-thresh or --ransac-iters.")
        sys.exit(1)

    center, direction, radius = best_model
    all_dists = _point_to_cylinder_distance(points, center, direction, radius)
    n_inliers = int((all_dists < thresh).sum())

    results = {
        "center":       np.round(center, 4),
        "direction":    np.round(direction, 4),
        "radius_m":     radius,
        "diameter_m":   radius * 2,
        "radius_cm":    radius * 100,
        "diameter_cm":  radius * 2 * 100,
        "n_inliers":    n_inliers,
        "inlier_pct":   n_inliers / len(points) * 100,
        "fit_time_s":   time.perf_counter() - t0,
    }

    print(f"\n{'─'*44}")
    print(f"  Pipe Measurements")
    print(f"{'─'*44}")
    print(f"  Center        : {results['center']}")
    print(f"  Orientation   : {results['direction']}")
    print(f"  Radius        : {results['radius_cm']:.2f} cm")
    print(f"  Diameter      : {results['diameter_cm']:.2f} cm")
    print(f"  Inliers       : {n_inliers:,} / {len(points):,} ({results['inlier_pct']:.1f}%)")
    print(f"  Fit time      : {results['fit_time_s']:.2f}s")
    print(f"{'─'*44}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="RANSAC cylinder fit on class-64 labeled LAS files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file",          required=True,               help="Labeled LAS file.")
    p.add_argument("--voxel-size",    type=float, default=DEFAULT_VOXEL_SIZE)
    p.add_argument("--epsilon",       type=float, default=DEFAULT_EPSILON)
    p.add_argument("--min-points",    type=int,   default=DEFAULT_MIN_POINTS)
    p.add_argument("--ransac-thresh", type=float, default=DEFAULT_RANSAC_THRESH)
    p.add_argument("--ransac-iters",  type=int,   default=DEFAULT_RANSAC_ITERS)
    p.add_argument("--fit-samples",   type=int,   default=DEFAULT_FIT_SAMPLES)
    p.add_argument("--no-downsample", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    t0 = time.perf_counter()

    pcd = load_point_cloud(args.file)

    if args.no_downsample:
        print("\n[2/4] Skipping downsampling.")
        working = pcd
    else:
        working = downsample(pcd, args.voxel_size)

    cluster_pcd, cluster_pts = find_largest_cluster(
        working, args.epsilon, args.min_points
    )

    visualize_cluster(working, cluster_pcd)

    fit_cylinder(
        cluster_pts,
        thresh=args.ransac_thresh,
        n_iterations=args.ransac_iters,
        fit_samples=args.fit_samples,
    )

    print(f"Total elapsed: {time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    main()
