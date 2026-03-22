"""
ransactest.py


RANSAC cylinder fitting benchmark for large-scale LiDAR point clouds.
Optimized for 80M+ point datasets via voxel downsampling and cluster subsampling.


Key changes vs pyransac3d:
  - pyransac3d is pure-Python and hangs on 300K+ points.
  - This script uses a vectorised NumPy RANSAC that runs in seconds at any scale.
  - Cluster points are randomly subsampled to `--fit-samples` before fitting
    (RANSAC needs geometric coverage, not density — 8K points is plenty).


Usage:
    python ransactest.py --file path/to/file.las [options]


    python ransactest.py --file ./segmented_lidar/Lab1total_labeled_weight.las --epsilon 0.5
"""


import argparse
import sys
import time


import laspy
import numpy as np
import open3d as o3d




# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------


DEFAULT_VOXEL_SIZE    = 0.05    # Voxel size (m) for scene downsampling.
DEFAULT_EPSILON       = 0.5     # DBSCAN neighbourhood radius (scene units). Updated from 0.05.
DEFAULT_MIN_POINTS    = 100     # DBSCAN minimum cluster size.
DEFAULT_RANSAC_THRESH = 0.01    # Max point-to-cylinder distance (m) for inliers.
DEFAULT_RANSAC_ITERS  = 1000    # RANSAC iterations.
DEFAULT_FIT_SAMPLES   = 8_000   # Points subsampled from the cluster before fitting.
DEFAULT_VISUALIZE     = False




# ---------------------------------------------------------------------------
# Step 1 - Load
# ---------------------------------------------------------------------------


def load_point_cloud(file_path: str) -> o3d.geometry.PointCloud:
    """Read a LAS/LAZ file and return an Open3D PointCloud (full resolution)."""
    print(f"\n[1/4] Loading '{file_path}' ...")
    t0 = time.perf_counter()


    las = laspy.read(file_path)
    xyz = np.column_stack((las.x, las.y, las.z))


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)


    print(f"      Loaded {len(pcd.points):,} points in {time.perf_counter()-t0:.2f}s.")
    return pcd




# ---------------------------------------------------------------------------
# Step 2 - Downsample
# ---------------------------------------------------------------------------


def downsample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    """
    Voxel-grid downsample.


    Preserves the spatial distribution of the cloud (unlike stride sampling).
    At voxel_size=0.05 m on an 80M-point cloud you typically retain ~300-500K
    well-distributed points — enough for reliable DBSCAN clustering.
    """
    print(f"\n[2/4] Voxel downsampling (voxel_size={voxel_size} m) ...")
    t0 = time.perf_counter()


    pcd_down = pcd.voxel_down_sample(voxel_size)


    ratio = len(pcd_down.points) / len(pcd.points) * 100
    print(f"      {len(pcd.points):,} -> {len(pcd_down.points):,} points "
          f"({ratio:.1f}% retained) in {time.perf_counter()-t0:.2f}s.")
    return pcd_down




# ---------------------------------------------------------------------------
# Step 3 - DBSCAN clustering
# ---------------------------------------------------------------------------


def find_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    epsilon: float,
    min_points: int,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Run DBSCAN and return the largest cluster as a PointCloud and point array.
    Exits with an informative message if no clusters are found.
    """
    print(f"\n[3/4] DBSCAN clustering (eps={epsilon}, min_points={min_points}) ...")
    t0 = time.perf_counter()


    labels = np.array(
        pcd.cluster_dbscan(eps=epsilon, min_points=min_points, print_progress=True)
    )


    print(f"      Clustering finished in {time.perf_counter()-t0:.2f}s.")
    max_label = labels.max()


    if max_label < 0:
        print(
            "\n DBSCAN found NO clusters - all points classified as noise.\n"
            "    Your 'epsilon' value is likely too small for the coordinate units\n"
            "    of your LAS file (millimetres vs metres vs survey feet).\n"
            "    Suggested fixes:\n"
            "      - Try --epsilon 0.5, 5.0, or 50.0 and re-run.\n"
            "      - Inspect las.header.offsets / las.header.scales to check units."
        )
        sys.exit(1)


    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    print(f"      Found {len(unique_labels):,} clusters (+ noise).")


    largest_label = unique_labels[np.argmax(counts)]
    indices = np.where(labels == largest_label)[0]
    cluster_pcd = pcd.select_by_index(indices)
    cluster_pts = np.asarray(cluster_pcd.points)


    print(f"      Largest cluster: label={largest_label}, {len(cluster_pts):,} points.")
    return cluster_pcd, cluster_pts




# ---------------------------------------------------------------------------
# Step 4 - Fast vectorised RANSAC cylinder fit
# ---------------------------------------------------------------------------


def _point_to_cylinder_distance(pts: np.ndarray,
                                 axis_pt: np.ndarray,
                                 axis_dir: np.ndarray,
                                 radius: float) -> np.ndarray:
    """
    Radial distance from each point to the cylinder surface.
    Fully vectorised over (N, 3) — no Python loops.
    """
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    diff = pts - axis_pt
    proj = diff - np.outer(diff @ axis_dir, axis_dir)  # perp component
    dist_to_axis = np.linalg.norm(proj, axis=1)
    return np.abs(dist_to_axis - radius)




def _fit_cylinder_from_sample(pts: np.ndarray):
    """
    Minimal cylinder fit from a small sample via PCA axis + median radius.
    Returns (axis_point, axis_direction, radius) or None if degenerate.
    """
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    axis_dir = Vt[0]  # direction of greatest variance ~ cylinder axis


    axis_dir_n = axis_dir / np.linalg.norm(axis_dir)
    proj = centred - np.outer(centred @ axis_dir_n, axis_dir_n)
    radius = float(np.median(np.linalg.norm(proj, axis=1)))


    if radius < 1e-9:
        return None
    return centroid, axis_dir_n, radius




def fit_cylinder(
    points: np.ndarray,
    thresh: float,
    n_iterations: int,
    fit_samples: int,
) -> dict:
    """
    RANSAC cylinder fit using a fast vectorised NumPy implementation.


    Why not pyransac3d?
      pyransac3d iterates over individual points in Python loops and hangs
      on 300K+ point clusters (20+ minutes in testing). This implementation
      evaluates all N points against the model in a single NumPy call,
      making each RANSAC iteration roughly 1000x faster.


    fit_samples:
      RANSAC needs geometric diversity, not density. Subsampling the cluster
      to 5-10K points before fitting preserves shape while cutting per-iteration
      cost dramatically. Inliers are then recounted on the full cluster.
    """
    print(f"\n[4/4] Fitting RANSAC cylinder ...")
    print(f"      Cluster size   : {len(points):,} points")


    if len(points) > fit_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=fit_samples, replace=False)
        fit_pts = points[idx]
        print(f"      Subsampled to  : {fit_samples:,} points for fitting")
    else:
        fit_pts = points
        print(f"      (cluster smaller than fit_samples, using all points)")


    print(f"      RANSAC iters   : {n_iterations}")
    print(f"      Inlier thresh  : {thresh} m")
    print(f"      Fitting ...", flush=True)


    t0 = time.perf_counter()


    best_inlier_count = -1
    best_model = None
    rng = np.random.default_rng(0)
    n = len(fit_pts)
    SAMPLE_SIZE = 10  # points sampled per RANSAC iteration


    for i in range(n_iterations):
        sample = fit_pts[rng.choice(n, size=SAMPLE_SIZE, replace=False)]
        model = _fit_cylinder_from_sample(sample)
        if model is None:
            continue


        axis_pt, axis_dir, radius = model
        dists = _point_to_cylinder_distance(fit_pts, axis_pt, axis_dir, radius)
        inlier_count = int((dists < thresh).sum())


        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_model = model


        # Progress indicator every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"      iter {i+1:4d}/{n_iterations}  best inliers: {best_inlier_count:,}", flush=True)


    elapsed = time.perf_counter() - t0


    if best_model is None:
        print("\n  RANSAC failed to find a valid cylinder model.")
        print("    Try increasing --ransac-thresh or --ransac-iters.")
        sys.exit(1)


    center, direction, radius = best_model


    # Recount inliers on the FULL cluster using the best model
    all_dists = _point_to_cylinder_distance(points, center, direction, radius)
    n_inliers = int((all_dists < thresh).sum())
    inlier_ratio = n_inliers / len(points) * 100


    results = {
        "center":       np.round(center, 4),
        "direction":    np.round(direction, 4),
        "radius_m":     float(radius),
        "diameter_m":   float(radius * 2),
        "radius_cm":    float(radius * 100),
        "diameter_cm":  float(radius * 2 * 100),
        "n_inliers":    n_inliers,
        "inlier_ratio": inlier_ratio,
        "fit_time_s":   elapsed,
    }


    print(f"\n{'─'*42}")
    print(f"  Pipe Measurements")
    print(f"{'─'*42}")
    print(f"  Center (X, Y, Z)   : {results['center']}")
    print(f"  Orientation vector : {results['direction']}")
    print(f"  Radius             : {results['radius_cm']:.2f} cm")
    print(f"  Diameter           : {results['diameter_cm']:.2f} cm")
    print(f"  Inliers (full clus): {n_inliers:,} / {len(points):,} ({inlier_ratio:.1f}%)")
    print(f"  Fit time           : {elapsed:.2f}s")
    print(f"{'─'*42}\n")


    return results




# ---------------------------------------------------------------------------
# Optional visualisation
# ---------------------------------------------------------------------------


def visualize(full_pcd: o3d.geometry.PointCloud,
              cluster_pcd: o3d.geometry.PointCloud) -> None:
    print("Opening 3D viewer - drag to rotate, scroll to zoom, Q to quit.")
    full_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    cluster_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([full_pcd, cluster_pcd])




# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fast RANSAC cylinder benchmark for large LiDAR point clouds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", required=True,
                   help="Path to the input LAS/LAZ file.")
    p.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE,
                   help="Voxel size (m) for scene downsampling. Larger = faster.")
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                   help="DBSCAN neighbourhood radius (same units as point cloud).")
    p.add_argument("--min-points", type=int, default=DEFAULT_MIN_POINTS,
                   help="DBSCAN minimum points per cluster.")
    p.add_argument("--ransac-thresh", type=float, default=DEFAULT_RANSAC_THRESH,
                   help="RANSAC inlier distance threshold (m).")
    p.add_argument("--ransac-iters", type=int, default=DEFAULT_RANSAC_ITERS,
                   help="Number of RANSAC iterations.")
    p.add_argument("--fit-samples", type=int, default=DEFAULT_FIT_SAMPLES,
                   help="Points subsampled from the cluster before RANSAC fitting.")
    p.add_argument("--visualize", action="store_true", default=DEFAULT_VISUALIZE,
                   help="Open interactive Open3D viewer after fitting.")
    p.add_argument("--no-downsample", action="store_true",
                   help="Skip voxel downsampling (very slow on large clouds).")
    return p




def main() -> None:
    args = build_parser().parse_args()
    total_t0 = time.perf_counter()


    full_pcd = load_point_cloud(args.file)


    if args.no_downsample:
        print("\n[2/4] Skipping downsampling (--no-downsample).")
        working_pcd = full_pcd
    else:
        working_pcd = downsample(full_pcd, args.voxel_size)


    cluster_pcd, cluster_pts = find_largest_cluster(
        working_pcd, args.epsilon, args.min_points
    )


    fit_cylinder(
        cluster_pts,
        thresh=args.ransac_thresh,
        n_iterations=args.ransac_iters,
        fit_samples=args.fit_samples,
    )


    print(f"Total elapsed: {time.perf_counter()-total_t0:.2f}s")


    if args.visualize:
        visualize(working_pcd, cluster_pcd)




if __name__ == "__main__":
    main()
"""
ransactest.py


RANSAC cylinder fitting benchmark for large-scale LiDAR point clouds.
Optimized for 80M+ point datasets via voxel downsampling and cluster subsampling.


Key changes vs pyransac3d:
  - pyransac3d is pure-Python and hangs on 300K+ points.
  - This script uses a vectorised NumPy RANSAC that runs in seconds at any scale.
  - Cluster points are randomly subsampled to `--fit-samples` before fitting
    (RANSAC needs geometric coverage, not density — 8K points is plenty).


Usage:
    python ransactest.py --file path/to/file.las [options]


    python ransactest.py --file ./segmented_lidar/Lab1total_labeled_weight.las --epsilon 0.5
"""


import argparse
import sys
import time


import laspy
import numpy as np
import open3d as o3d




# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------


DEFAULT_VOXEL_SIZE    = 0.05    # Voxel size (m) for scene downsampling.
DEFAULT_EPSILON       = 0.5     # DBSCAN neighbourhood radius (scene units). Updated from 0.05.
DEFAULT_MIN_POINTS    = 100     # DBSCAN minimum cluster size.
DEFAULT_RANSAC_THRESH = 0.01    # Max point-to-cylinder distance (m) for inliers.
DEFAULT_RANSAC_ITERS  = 1000    # RANSAC iterations.
DEFAULT_FIT_SAMPLES   = 8_000   # Points subsampled from the cluster before fitting.
DEFAULT_VISUALIZE     = False




# ---------------------------------------------------------------------------
# Step 1 - Load
# ---------------------------------------------------------------------------


def load_point_cloud(file_path: str) -> o3d.geometry.PointCloud:
    """Read a LAS/LAZ file and return an Open3D PointCloud (full resolution)."""
    print(f"\n[1/4] Loading '{file_path}' ...")
    t0 = time.perf_counter()


    las = laspy.read(file_path)
    xyz = np.column_stack((las.x, las.y, las.z))


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)


    print(f"      Loaded {len(pcd.points):,} points in {time.perf_counter()-t0:.2f}s.")
    return pcd




# ---------------------------------------------------------------------------
# Step 2 - Downsample
# ---------------------------------------------------------------------------


def downsample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    """
    Voxel-grid downsample.


    Preserves the spatial distribution of the cloud (unlike stride sampling).
    At voxel_size=0.05 m on an 80M-point cloud you typically retain ~300-500K
    well-distributed points — enough for reliable DBSCAN clustering.
    """
    print(f"\n[2/4] Voxel downsampling (voxel_size={voxel_size} m) ...")
    t0 = time.perf_counter()


    pcd_down = pcd.voxel_down_sample(voxel_size)


    ratio = len(pcd_down.points) / len(pcd.points) * 100
    print(f"      {len(pcd.points):,} -> {len(pcd_down.points):,} points "
          f"({ratio:.1f}% retained) in {time.perf_counter()-t0:.2f}s.")
    return pcd_down




# ---------------------------------------------------------------------------
# Step 3 - DBSCAN clustering
# ---------------------------------------------------------------------------


def find_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    epsilon: float,
    min_points: int,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Run DBSCAN and return the largest cluster as a PointCloud and point array.
    Exits with an informative message if no clusters are found.
    """
    print(f"\n[3/4] DBSCAN clustering (eps={epsilon}, min_points={min_points}) ...")
    t0 = time.perf_counter()


    labels = np.array(
        pcd.cluster_dbscan(eps=epsilon, min_points=min_points, print_progress=True)
    )


    print(f"      Clustering finished in {time.perf_counter()-t0:.2f}s.")
    max_label = labels.max()


    if max_label < 0:
        print(
            "\n DBSCAN found NO clusters - all points classified as noise.\n"
            "    Your 'epsilon' value is likely too small for the coordinate units\n"
            "    of your LAS file (millimetres vs metres vs survey feet).\n"
            "    Suggested fixes:\n"
            "      - Try --epsilon 0.5, 5.0, or 50.0 and re-run.\n"
            "      - Inspect las.header.offsets / las.header.scales to check units."
        )
        sys.exit(1)


    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    print(f"      Found {len(unique_labels):,} clusters (+ noise).")


    largest_label = unique_labels[np.argmax(counts)]
    indices = np.where(labels == largest_label)[0]
    cluster_pcd = pcd.select_by_index(indices)
    cluster_pts = np.asarray(cluster_pcd.points)


    print(f"      Largest cluster: label={largest_label}, {len(cluster_pts):,} points.")
    return cluster_pcd, cluster_pts




# ---------------------------------------------------------------------------
# Step 4 - Fast vectorised RANSAC cylinder fit
# ---------------------------------------------------------------------------


def _point_to_cylinder_distance(pts: np.ndarray,
                                 axis_pt: np.ndarray,
                                 axis_dir: np.ndarray,
                                 radius: float) -> np.ndarray:
    """
    Radial distance from each point to the cylinder surface.
    Fully vectorised over (N, 3) — no Python loops.
    """
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    diff = pts - axis_pt
    proj = diff - np.outer(diff @ axis_dir, axis_dir)  # perp component
    dist_to_axis = np.linalg.norm(proj, axis=1)
    return np.abs(dist_to_axis - radius)




def _fit_cylinder_from_sample(pts: np.ndarray):
    """
    Minimal cylinder fit from a small sample via PCA axis + median radius.
    Returns (axis_point, axis_direction, radius) or None if degenerate.
    """
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    axis_dir = Vt[0]  # direction of greatest variance ~ cylinder axis


    axis_dir_n = axis_dir / np.linalg.norm(axis_dir)
    proj = centred - np.outer(centred @ axis_dir_n, axis_dir_n)
    radius = float(np.median(np.linalg.norm(proj, axis=1)))


    if radius < 1e-9:
        return None
    return centroid, axis_dir_n, radius




def fit_cylinder(
    points: np.ndarray,
    thresh: float,
    n_iterations: int,
    fit_samples: int,
) -> dict:
    """
    RANSAC cylinder fit using a fast vectorised NumPy implementation.


    Why not pyransac3d?
      pyransac3d iterates over individual points in Python loops and hangs
      on 300K+ point clusters (20+ minutes in testing). This implementation
      evaluates all N points against the model in a single NumPy call,
      making each RANSAC iteration roughly 1000x faster.


    fit_samples:
      RANSAC needs geometric diversity, not density. Subsampling the cluster
      to 5-10K points before fitting preserves shape while cutting per-iteration
      cost dramatically. Inliers are then recounted on the full cluster.
    """
    print(f"\n[4/4] Fitting RANSAC cylinder ...")
    print(f"      Cluster size   : {len(points):,} points")


    if len(points) > fit_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=fit_samples, replace=False)
        fit_pts = points[idx]
        print(f"      Subsampled to  : {fit_samples:,} points for fitting")
    else:
        fit_pts = points
        print(f"      (cluster smaller than fit_samples, using all points)")


    print(f"      RANSAC iters   : {n_iterations}")
    print(f"      Inlier thresh  : {thresh} m")
    print(f"      Fitting ...", flush=True)


    t0 = time.perf_counter()


    best_inlier_count = -1
    best_model = None
    rng = np.random.default_rng(0)
    n = len(fit_pts)
    SAMPLE_SIZE = 10  # points sampled per RANSAC iteration


    for i in range(n_iterations):
        sample = fit_pts[rng.choice(n, size=SAMPLE_SIZE, replace=False)]
        model = _fit_cylinder_from_sample(sample)
        if model is None:
            continue


        axis_pt, axis_dir, radius = model
        dists = _point_to_cylinder_distance(fit_pts, axis_pt, axis_dir, radius)
        inlier_count = int((dists < thresh).sum())


        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_model = model


        # Progress indicator every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"      iter {i+1:4d}/{n_iterations}  best inliers: {best_inlier_count:,}", flush=True)


    elapsed = time.perf_counter() - t0


    if best_model is None:
        print("\n  RANSAC failed to find a valid cylinder model.")
        print("    Try increasing --ransac-thresh or --ransac-iters.")
        sys.exit(1)


    center, direction, radius = best_model


    # Recount inliers on the FULL cluster using the best model
    all_dists = _point_to_cylinder_distance(points, center, direction, radius)
    n_inliers = int((all_dists < thresh).sum())
    inlier_ratio = n_inliers / len(points) * 100


    results = {
        "center":       np.round(center, 4),
        "direction":    np.round(direction, 4),
        "radius_m":     float(radius),
        "diameter_m":   float(radius * 2),
        "radius_cm":    float(radius * 100),
        "diameter_cm":  float(radius * 2 * 100),
        "n_inliers":    n_inliers,
        "inlier_ratio": inlier_ratio,
        "fit_time_s":   elapsed,
    }


    print(f"\n{'─'*42}")
    print(f"  Pipe Measurements")
    print(f"{'─'*42}")
    print(f"  Center (X, Y, Z)   : {results['center']}")
    print(f"  Orientation vector : {results['direction']}")
    print(f"  Radius             : {results['radius_cm']:.2f} cm")
    print(f"  Diameter           : {results['diameter_cm']:.2f} cm")
    print(f"  Inliers (full clus): {n_inliers:,} / {len(points):,} ({inlier_ratio:.1f}%)")
    print(f"  Fit time           : {elapsed:.2f}s")
    print(f"{'─'*42}\n")


    return results




# ---------------------------------------------------------------------------
# Optional visualisation
# ---------------------------------------------------------------------------


def visualize(full_pcd: o3d.geometry.PointCloud,
              cluster_pcd: o3d.geometry.PointCloud) -> None:
    print("Opening 3D viewer - drag to rotate, scroll to zoom, Q to quit.")
    full_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    cluster_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([full_pcd, cluster_pcd])




# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fast RANSAC cylinder benchmark for large LiDAR point clouds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", required=True,
                   help="Path to the input LAS/LAZ file.")
    p.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE,
                   help="Voxel size (m) for scene downsampling. Larger = faster.")
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                   help="DBSCAN neighbourhood radius (same units as point cloud).")
    p.add_argument("--min-points", type=int, default=DEFAULT_MIN_POINTS,
                   help="DBSCAN minimum points per cluster.")
    p.add_argument("--ransac-thresh", type=float, default=DEFAULT_RANSAC_THRESH,
                   help="RANSAC inlier distance threshold (m).")
    p.add_argument("--ransac-iters", type=int, default=DEFAULT_RANSAC_ITERS,
                   help="Number of RANSAC iterations.")
    p.add_argument("--fit-samples", type=int, default=DEFAULT_FIT_SAMPLES,
                   help="Points subsampled from the cluster before RANSAC fitting.")
    p.add_argument("--visualize", action="store_true", default=DEFAULT_VISUALIZE,
                   help="Open interactive Open3D viewer after fitting.")
    p.add_argument("--no-downsample", action="store_true",
                   help="Skip voxel downsampling (very slow on large clouds).")
    return p




def main() -> None:
    args = build_parser().parse_args()
    total_t0 = time.perf_counter()


    full_pcd = load_point_cloud(args.file)


    if args.no_downsample:
        print("\n[2/4] Skipping downsampling (--no-downsample).")
        working_pcd = full_pcd
    else:
        working_pcd = downsample(full_pcd, args.voxel_size)


    cluster_pcd, cluster_pts = find_largest_cluster(
        working_pcd, args.epsilon, args.min_points
    )


    fit_cylinder(
        cluster_pts,
        thresh=args.ransac_thresh,
        n_iterations=args.ransac_iters,
        fit_samples=args.fit_samples,
    )


    print(f"Total elapsed: {time.perf_counter()-total_t0:.2f}s")


    if args.visualize:
        visualize(working_pcd, cluster_pcd)




if __name__ == "__main__":
    main()

