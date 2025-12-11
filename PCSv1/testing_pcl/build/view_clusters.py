import open3d as o3d
import glob
import numpy as np
import matplotlib.pyplot as plt

# 1. Find all cluster files
files = glob.glob("cluster_*.pcd")
print(f"Found {len(files)} clusters.")

geometries = []
# Create a color map (e.g., Jet or Tab20)
cmap = plt.get_cmap("tab20")

for i, filename in enumerate(files):
    # Load Cloud
    pcd = o3d.io.read_point_cloud(filename)
    
    # Assign a unique color based on ID
    # cmap returns (R,G,B,A), we need just (R,G,B)
    color = cmap(i / len(files))[:3] 
    pcd.paint_uniform_color(color)
    
    geometries.append(pcd)

# Visualize everything together
o3d.visualization.draw_geometries(geometries, window_name="Euclidean Clustering Results")