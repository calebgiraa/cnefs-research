
### IMPORTS ###
import laspy
import numpy as np
import open3d as o3d
import os
import cv2
import argparse

### FUNCTIONS ### 

def detect_floor_and_align_open3d(las_path):
    """
    Loads a .las file, detects the floor plane using Open3D RANSAC,
    and aligns the cloud so that the detected floor normal = +Z axis.

    Returns:
        aligned_points : (N, 3) numpy array
        aligned_colors : (N, 3) uint8 RGB
        R : 3x3 rotation matrix
        t : 3-element translation vector
        floor_normal : unit 3-vector
    """

    # --- Load LAS ---
    las = laspy.read(las_path)

    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

    # --- Extract colors if present ---
    if hasattr(las, "red"):
        colors = np.vstack((las.red, las.green, las.blue)).T.astype(np.float64)

        # Normalize LAS colors, convert to 255 format
        if colors.max() > 255:
            colors = (colors / 65535.0 * 255.0).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
    else:
        # fallback (Grayscale)
        colors = np.zeros((points.shape[0], 3), dtype=np.uint8)

    # --- Convert to Open3D point cloud (for plane detection only) ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Optional: attach colors for debugging/visualization
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # --- Downsample for plane fitting ---
    pcd_small = pcd.voxel_down_sample(voxel_size=0.05)

    # --- RANSAC Floor Plane Detection ---
    plane_model, inliers = pcd_small.segment_plane(
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=1000
    )

    a, b, c, d = plane_model
    floor_normal = np.array([a, b, c], dtype=float)
    floor_normal /= np.linalg.norm(floor_normal)

    # Ensure upward facing normal
    if floor_normal[2] < 0:
        floor_normal = -floor_normal

    # --- Build Orthogonal Frame ---
    z_axis = floor_normal

    world_x = np.array([1, 0, 0], dtype=float)
    x_axis = world_x - np.dot(world_x, z_axis) * z_axis

    if np.linalg.norm(x_axis) < 1e-6:
        world_x = np.array([0, 1, 0], dtype=float)
        x_axis = world_x - np.dot(world_x, z_axis) * z_axis

    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Rotation matrix
    R = np.vstack([x_axis, y_axis, z_axis]).T

    # --- Compute translation (center of floor points) ---
    floor_points = np.asarray(pcd_small.points)[inliers]
    t = floor_points.mean(axis=0)

    # --- Apply transform to ALL points ---
    aligned_points = (points - t) @ R

    

    return aligned_points, colors, R, t, floor_normal

def load_las_as_points_and_colors(las_path):
    """
    Load LAS file and return:
        points: (N, 3) float64 array of XYZ
        colors: (N, 3) uint8 array of RGB (0-255)
    Falls back to grayscale if RGB not present.
    """
    las = laspy.read(las_path)

    # XYZ
    points = np.vstack((las.x, las.y, las.z)).T  # (N, 3)

    # RGB handling 
    if hasattr(las, "red"): # If a color exists
        raw_colors = np.vstack((las.red, las.green, las.blue)).T

        # Typical LAS: 16-bit color, scale down
        if np.max(las.red) > 255:
            colors = (raw_colors / 256).astype(np.uint8)
        else:
            colors = raw_colors.astype(np.uint8)
    else:
        # Grayscale fallback 
        z = points[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-12)
        gray = (z_norm * 255).astype(np.uint8)
        colors = np.stack((gray, gray, gray), axis=1)

    return points, colors

def reset_open3d():
    """
    - Fix open3d window if error occurs
    - Error occurs typically after closing open3d manually
    """
    try:
        o3d.visualization.Visualizer().destroy_window()
    except:
        pass
    return()


def define_target_cube(points, seed, cube_size=2.0):
    """
    Define a target point, region and centroid of the target mask.
    """

    if points.shape[0] == 0:
        raise ValueError("Point cloud is empty.")


    # ---  Snap seed to nearest surface point ---
    dists = np.linalg.norm(points - seed, axis=1)
    seed = points[np.argmin(dists)]

    # ---  Build cube mask around snapped seed ---
    half = cube_size / 2.0
    lower = seed - half
    upper = seed + half

    target_mask = np.all(
        (points >= lower) & (points <= upper),
        axis=1
    )

    if not np.any(target_mask):
        raise RuntimeError(
            "Target mask is empty after snapping seed to surface. "
            "Increase cube_size or inspect point cloud density."
        )

    # ---  Compute centroid from target points ---
    target_centroid = np.mean(points[target_mask], axis=0)

    return target_centroid, target_mask


def create_poses(target,n_poses,radius=5.0,elev_range=(0.0, np.deg2rad(45.0)),include_topdown=True):
    """
    Generate camera poses around a target point using azimuth/elevation sampling.

    Important note - this is geometry only, including camera view distance vector and rotation matrix.

    Parameters
    ----------
    target : (3,) array-like
        Target point (x, y, z)
    n_poses : int
        Number of desired camera poses
    radius : float
        Distance from camera to target
        Center of circle is target
    elev_range : (float, float)
        Min/max elevation angles in radians
        Default 0 to 45
    include_topdown : bool
        Whether to include an explicit top-down pose
        Default is true

    Returns
    -------
    poses : list of dict
        Each pose contains:
        - position : (3,) np.ndarray
        - view_dir : (3,) np.ndarray (unit vector)
        - R: Camera rotation
        - azimuth  : float (rad)
        - elevation: float (rad)
    """
    
    target = np.asarray(target, dtype=float)
    poses = []
    
    # --- Reserve one slot for top-down if requested ---
    n_remaining = n_poses
    if include_topdown and n_poses > 0:
        pos = target + np.array([0.0, 0.0, radius])
        view_dir, R = compute_view_and_R(pos, target)

        poses.append({
            "position": pos,
            "view_dir": view_dir,
            "R":R,
            "azimuth": 0.0,
            "elevation": np.pi / 2,
        })
        n_remaining -= 1

    if n_remaining <= 0:
        return poses

    # --- Azimuth sampling (uniform ring) ---
    azimuths = np.linspace(0, 2 * np.pi, n_remaining, endpoint=False)

    # ---  Elevation sampling  ---
    elev_min, elev_max = elev_range
    elevations = np.linspace(elev_min, elev_max, n_remaining)

    # --- Generate poses ---
    for theta, phi in zip(azimuths, elevations):
        direction = np.array([
            np.cos(theta) * np.cos(phi),
            np.sin(theta) * np.cos(phi),
            np.sin(phi),
        ])

        pos = target + radius * direction
        view_dir, R = compute_view_and_R(pos, target)
        # Insert poses into list with dict values
        poses.append({
            "position": pos,
            "view_dir": view_dir,
            "R": R,
            "azimuth": theta,
            "elevation": phi,
        })

    return poses


def compute_view_and_R(
    cam_pos,
    target,
    world_up=np.array([0.0, 0.0, 1.0]),
    eps=1e-6,
):
    """
    Second version. Edited to reduce camera roll and pass on camera rotation vector.
    """
    view_vec = target - cam_pos
    norm = np.linalg.norm(view_vec)
    if norm < eps:
        raise ValueError("Camera position and target coincide.")

    z = view_vec / norm  # forward

    up = world_up / np.linalg.norm(world_up)

    # --- Compute right axis ---
    x = np.cross(up, z)
    
    if np.linalg.norm(x) < eps:
        # Degenerate case (looking straight up/down)
        up = np.array([1.0, 0.0, 0.0])
        x = np.cross(up, z)

    x /= np.linalg.norm(x)

    # --- Recompute true orthogonal up ---
    y = np.cross(z, x)
    y /= np.linalg.norm(y)

    # Re‐orthogonalize
    x = np.cross(y, z)

    # --- Construct rotation ---
    R = np.stack([x, y, z], axis=0)

    return z, R



def generate_depth_map(
    pose,
    points,
    target_mask_3d,
    img_size=(64, 64),
    fov_deg=70.0,
    z_near=0.1,
    z_far=50.0,
):
    """
    Generate coarse depth map and target hits from a camera pose.

    Given the camera poses and target mask, generate occlusion and perspective
    cost parameters. 

    Keep in mind that for an image ray generation size of (H,W)  = (64,64) 4096 rays will be created.
    """

    H, W = img_size

    # --- Initialize map array --- #
    depth_map = np.full((H, W), np.inf)
    target_hits = np.zeros((H, W), dtype=bool)

    cam_pos = pose["position"]
    R = pose["R"]

    # --- Intrinsics ---
    fov = np.deg2rad(fov_deg)
    fx = fy = 0.5 * W / np.tan(0.5 * fov)
    cx, cy = W / 2, H / 2

    # --- Transform points to camera frame ---
    pts_cam = (R @ (points - cam_pos).T).T

    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    # --- Keep valid depths ---
    valid = (z > z_near) & (z < z_far)

    x = x[valid]
    y = y[valid]
    z = z[valid]
    tgt = target_mask_3d[valid]

    # --- Project to image ---
    u = (fx * (x / z) + cx).astype(int)
    v = (fy * (y / z) + cy).astype(int)

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    u, v, z, tgt = u[in_bounds], v[in_bounds], z[in_bounds], tgt[in_bounds]

    # --- Z-buffer ---
    for ui, vi, zi, is_target in zip(u, v, z, tgt):
        if zi < depth_map[vi, ui]:
            depth_map[vi, ui] = zi
            target_hits[vi, ui] = is_target

    # Replace "no hit" pixels
    depth_map[np.isinf(depth_map)] = 0.0

    return depth_map, target_hits

def compute_total_cost(depth_map, target_hits, weights=[.4,.4]):
    """
    Compute total cost for a pose.

    Weights is by default a list where 0= occ and 1 = persp

    J_occ and J_persp output as unweighted value
    """

    # --- Occlusion cost ---
    total_pixels = depth_map > 0

    if np.sum(total_pixels) == 0:
        return np.inf

    target_pixels = target_hits
    non_target_pixels = total_pixels & (~target_pixels)

    visibility = np.sum(target_pixels) / np.sum(total_pixels)
    occlusion = np.sum(non_target_pixels) / np.sum(total_pixels)

    J_occ = (1 - visibility) + occlusion

    # --- Perspective cost ---
    target_depths = depth_map[target_hits]

    if len(target_depths) < 5:
        J_persp = np.inf
    else:
        sigma_z = np.std(target_depths)
        mean_z = np.mean(target_depths)

        A_t = np.mean(target_hits)

        depth_term = sigma_z / (mean_z + 1e-6)
        area_term = abs(A_t - 0.5)

        J_persp = depth_term + area_term

    # --- Total ---
    J_total = (
        weights[0] * J_occ +
        weights[1] * J_persp
    )

    return J_total, J_occ, J_persp

def evaluate_poses(poses, points, target_mask_3d, weights=[.4,.4]):
    """
    Results is initialized as a list. Then, data is stored as a dictionary within each list entry.
    """
    results = []
    # --- Depth map and total cost from each pose --- 
    for pose in poses:

        depth_map, target_mask_2d = generate_depth_map(
            pose,
            points,
            target_mask_3d
        )

        J_total, J_occ, J_persp = compute_total_cost(
            depth_map,
            target_mask_2d,
            weights
        )
        # --- Results --- #
        results.append({
            "pose": pose,
            "J_total": J_total,
            "J_occ": J_occ,
            "J_persp": J_persp,
        })

    return results

def select_best_pose(results):
    """
    Brief function to select the best pose. I will also write one to select the top three, but for now just stick with one.
    """
    results = sorted(results, key=lambda r: r["J_total"])
    return results[0]

def select_top_k_poses(results, k):
    """
    Return the k lowest-cost poses from evaluate_poses results.
    """
    sorted_results = sorted(results, key=lambda r: r["J_total"])
    return sorted_results[:min(k, len(sorted_results))]

def project_pinhole(
    points,
    colors,
    pose,
    img_h=800,
    img_w=None,
    fov_deg=60.0,
    
):
    """
    Core pinhole projection + rasterization + index tracking.

    Returns:
        image: (H, W, 3)
        depth_map: (H, W)
        index_map: (H, W)  # index of point contributing to each pixel
    """

    points = np.asarray(points, dtype=float)
    colors = np.asarray(colors, dtype=np.uint8)

    if img_w is None:
        img_w = int(img_h * 16.0 / 9.0)

    H, W = img_h, img_w

    # --- Cam Properties ---
    fov_rad = np.deg2rad(fov_deg)
    fy = 0.5 * H / np.tan(0.5 * fov_rad)
    fx = fy
    cx, cy = W / 2.0, H / 2.0


    # --- Rotation ---
    # Pass on pose information 
    cam_pos = pose["position"]
    R_cam = pose["R"]

    # --- Transform to camera frame ---
    P_cam = (R_cam @ (points-cam_pos).T).T

    x_cam = P_cam[:, 0]
    y_cam = P_cam[:, 1]
    z_cam = P_cam[:, 2]

    # --- Keep points in front ---
    in_front = z_cam > 0

    x_cam = x_cam[in_front]
    y_cam = y_cam[in_front]
    z_cam = z_cam[in_front]
    colors_cam = colors[in_front]

    # --- Track original indices
    original_indices = np.arange(len(points))[in_front]

    if x_cam.size == 0:
        return (
            np.zeros((H, W, 3), dtype=np.uint8),
            np.zeros((H, W)),
            np.full((H, W), -1, dtype=int)
        )

    # --- Projection ---
    u = np.round(fx * (x_cam / z_cam) + cx).astype(int)
    v = np.round(-fy * (y_cam / z_cam) + cy).astype(int)

    # --- Buffers ---
    image = np.zeros((H, W, 3), dtype=np.uint8)
    depth_map = np.full((H, W), np.inf)
    index_map = np.full((H, W), -1, dtype=int) 

    # --- Rasterization ---
    for i in range(len(u)):
        ix, iy = u[i], v[i]

        if 0 <= ix < W and 0 <= iy < H:
            z = z_cam[i]

            if z < depth_map[iy, ix]:
                depth_map[iy, ix] = z
                image[iy, ix] = colors_cam[i]
                index_map[iy, ix] = original_indices[i]  

    # Replace inf depths
    depth_map[np.isinf(depth_map)] = 0.0

    return image, depth_map, index_map


### ---  MAIN WRAPPER FUNCTION --- COST FUNCTION --- #

def main():
    parser = argparse.ArgumentParser(description="NBV Generation and Pinhole Camera Projection")
    parser.add_argument("input_file", help="Path to input .las file.")
    parser.add_argument("output_image", help="Path to output image (e.g., output.png).")
    parser.add_argument("number_cam_poses",type=int,help="Number of poses generated.")
    parser.add_argument("radius", type=float, help="Distance from the target point to the camera pose.")
    parser.add_argument("--elev_range",nargs=2, type=float, metavar=("MIN","MAX"), help="Elevation range in degrees. Ex. 10 45")
    parser.add_argument("--include_top_down",action="store_true", help="Include a top down view of the object, True of False.")
    
    
    #--- Optional user-defined target ---
    parser.add_argument("--target", nargs=3, type=float,
                        help="Optional target point (x y z)")

    # --- Camera Parameters ---
    parser.add_argument("--height", type=int, default=800, help="Image height in pixels (default: 800).")
    parser.add_argument("--width", type=int, default=None, help="Image width in pixels (default: 16:9 from height).")
    parser.add_argument("--fov", type=float, default=60.0, help="Vertical field of view in degrees (default: 60).")



    args = parser.parse_args()

    # --- Reset open3d
    reset_open3d()

    # Load data
    print(f"Loading {args.input_file}...")

    # -- Align Cloud Points ---
    """
    RANSAC alignment not currently deterministic, will need to adjust parameters.
    """
    #aligned_points, colors, R, t, floor_normal = detect_floor_and_align_open3d(args.input_file)
    aligned_points, colors = load_las_as_points_and_colors(args.input_file)


    # --- Target Module ---
    # Can add later, pose will by default select centroid

    if args.target is not None:
        seed = np.array(args.target)
    else:
        seed = np.mean(aligned_points, axis=0)

    target_centroid, target_mask= define_target_cube(aligned_points,seed,2) # in meters


    


    # --- Generate Poses ---

    # Double check elevation angles 
    if args.elev_range is not None:
        elev_range = np.deg2rad(args.elev_range)
    else:
        elev_range = (0.0, np.deg2rad(45.0))

    poses = create_poses(target_centroid,args.number_cam_poses,args.radius,elev_range=elev_range, include_topdown=args.include_top_down)

    # --- Evaluate Poses and Generate Cost Values ---

    results = evaluate_poses(poses,aligned_points,target_mask,weights=[.4,.4])

    print("Poses generated.")

    # -- Order poses by cost and choose best ---
    
    best = select_best_pose(results)
    best_pose = best["pose"]

    # --- Generate pinhole projection image with index information ---

    best_image, best_depth_map, best_index = project_pinhole(points=aligned_points,colors=colors,pose=best_pose,img_h=args.height,img_w=args.width,fov_deg=args.fov)

    # --- Return the best image and its index. Save the image ---
    cv2.imwrite(args.output_image,best_image)

    print("Done! Best pose returned.")

    return best_image,best_index

if __name__ == "__main__":
    import sys

    
    sys.argv = [
        "script_name.py",
        './input_perspective_lidar/Lab1total.las',          # input file
        './output_perspective_image/testing3.jpg',          # output image
        "5",                   # number_cam_poses
        "3.0",                 # radius

        "--elev_range", "0","45",                # Elevation range
                                                # Include Top Down or not. --include_top_down means true
        "--fov", "60"
    ]

    main()

