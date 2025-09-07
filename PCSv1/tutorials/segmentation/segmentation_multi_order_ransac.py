# --- Environment Setup --- #
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
# ------------------------- #


# --- Point Cloud Data Preperation --- #
DATANAME = "apartment.ply"
pcd = o3d.io.read_point_cloud("segmentation/DATA/" + DATANAME)
print("Data Preperation Successful!")
# ------------------------------------ #


# --- Data Pre-Processing --- #
pcd_center = pcd.get_center()
pcd.translate(-pcd_center)
# --------------------------- #


# --- Statistical outlier filter --- #
nn = 16
std_multiplier = 10

filtered_pcd = pcd.remove_statistical_outlier(nn, std_multiplier)

outliers = pcd.select_by_index(filtered_pcd[1], invert=True)
outliers.paint_uniform_color([1, 0, 0])
filtered_pcd = filtered_pcd[0]

o3d.visualization.draw_geometries([filtered_pcd, outliers])
# ---------------------------------- #


# --- Voxel downsampling --- #
voxel_size = 0.01
pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size = voxel_size)
o3d.visualization.draw_geometries([pcd_downsampled])
# -------------------------- #


# --- Estimating Normals --- #
nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())

radius_normals = nn_distance*4

pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16), fast_normal_computation=True)

pcd_downsampled.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([pcd_downsampled, outliers])
# -------------------------- #


# --- Extracting and Setting Parameters --- #
front = [ -0.24904880349948727, -0.66306507082651667, 0.7059174210382545 ] # Gotten from pressing ctrl+c at o3d output from the visualization above
lookat = [ -0.21062663734925158, 0.53264195607544895, 0.25312096983257781 ]
up = [ -0.11032698333934068, 0.7435627514322698, 0.65950162352318265 ]
zoom = 0.69999999999999996

pcd = pcd_downsampled
o3d.visualization.draw_geometries([pcd], zoom=zoom, front=front, lookat=lookat, up=up)
# ----------------------------------------- #


# --- RANSAC Planar Segmentation --- #
pt_to_plane_dist = 0.02 # 2 centimeters

plane_model, inliers = pcd.segment_plane(distance_threshold=pt_to_plane_dist, ransac_n=3, num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], zoom=zoom, front=front, lookat=lookat, up=up)
# ---------------------------------- #


# --- Multi-order RANSAC --- #
max_plane_idx = 7
pt_to_plane_dist = 0.02

segment_models = {}
segments = {}
rest = pcd

for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=pt_to_plane_dist, ransac_n=3, num_iterations=1000)
    segments[i]=rest.select_by_index(inliers)
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i,"/",max_plane_idx,"done.")

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest], zoom=zoom, front=front, lookat=lookat, up=up)
# -------------------------- #


# --- DBSCAN --- #
labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
rest.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest], zoom=zoom, front=front, lookat=lookat, up=up)
# -------------- #