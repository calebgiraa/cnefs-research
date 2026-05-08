import numpy as np
import cv2
import laspy
import argparse
import os
import csv
import math
from dataclasses import dataclass


"""
Camera Pose
"""
@dataclass
class CameraPose:
  position: np.ndarray  # (3,) world-space camera origin
  target: np.ndarray    # (3,) world-space look-at point


def get_camera_pose(points: np.ndarray, colors: np.ndarray) -> CameraPose:
  """
  Stub pose selector — swap in colleague's implementation here.

  Must keep this signature: (points: ndarray, colors: ndarray) -> CameraPose
  """
  centroid = points.mean(axis=0)
  dists = np.linalg.norm(points - centroid, axis=1)
  radius = np.percentile(dists, 90) * .6
  cam_pos = centroid + np.array([radius, 0.0, 0.0])

  angle_rad = math.radians(90)
  cam_pos = centroid + np.array([
    radius * math.cos(angle_rad),
    radius * math.sin(angle_rad),
    -1.0
  ])

  target = centroid.copy()
  target[2] -= 1.5
  return CameraPose(position=cam_pos, target=target)


"""
Camera Method
"""
def look_at_rotation(camera_pos: np.ndarray, target: np.ndarray,
                     up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
  forward = target - camera_pos
  forward /= np.linalg.norm(forward)

  right = np.cross(forward, up)
  right /= np.linalg.norm(right)

  true_up = np.cross(right, forward)

  # Rows: [forward, right, true_up] — matches new_translation.py convention
  return np.vstack((forward, right, true_up))


"""
Perspective Projection
"""
def project_perspective(pose: CameraPose, points: np.ndarray, colors: np.ndarray,
                        width: int, height: int, fov_h_deg: float):
  """
  Project the point cloud onto a perspective (pinhole) image.

  Returns
  -------
  image        : (H, W, 3) uint8 RGB
  visible_idx  : (M,) indices into `points` of in-frustum points
  pixel_coords : (M, 2) int array of (pixel_x, pixel_y) per visible point
  """
  fov_h = math.radians(fov_h_deg)
  fov_v = 2.0 * math.atan(math.tan(fov_h / 2.0) * height / width)

  tan_h = math.tan(fov_h / 2.0)
  tan_v = math.tan(fov_v / 2.0)

  R = look_at_rotation(pose.position, pose.target)
  cam_points = (points - pose.position) @ R.T  # (N, 3): cols = [depth, right, up]

  depth = cam_points[:, 0]

  # Normalized device coordinates: [-1, 1] within the frustum
  ndc_x = cam_points[:, 1] / np.where(depth > 0, depth * tan_h, np.inf)
  ndc_y = cam_points[:, 2] / np.where(depth > 0, depth * tan_v, np.inf)

  in_frustum = (depth > 0) & (np.abs(ndc_x) <= 1.0) & (np.abs(ndc_y) <= 1.0)
  visible_idx = np.where(in_frustum)[0]

  px = ((ndc_x[visible_idx] + 1.0) / 2.0 * (width - 1)).astype(int)
  py = ((1.0 - ndc_y[visible_idx]) / 2.0 * (height - 1)).astype(int)

  image = np.zeros((height, width, 3), dtype=np.uint8)
  depth_buffer = np.full((height, width), np.inf)

  dist = depth[visible_idx]
  visible_colors = colors[visible_idx]

  for i in range(len(visible_idx)):
    ix, iy = px[i], py[i]
    if dist[i] < depth_buffer[iy, ix]:
      cv2.circle(image, (ix, iy), radius=2,
                 color=visible_colors[i].tolist(), thickness=-1)
      depth_buffer[iy, ix] = dist[i]

  return image, visible_idx, np.column_stack((px, py))



"""
Main method
"""
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("input_file", help="Path to input .las file.")
  parser.add_argument("output_dir", help="Output directory.")
  parser.add_argument("--width", type=int, default=1920, help="Image width in pixels.")
  parser.add_argument("--height", type=int, default=1080, help="Image height in pixels.")
  parser.add_argument("--fov", type=float, default=90.0,
                      help="Horizontal field of view in degrees.")
  parser.add_argument("--export_csv", action="store_true",
                      help="Export point-to-pixel mapping CSV.")
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

  print(f"Loading {args.input_file}...")
  las = laspy.read(args.input_file)

  points = np.vstack((las.x, las.y, las.z)).T

  if hasattr(las, "red"):
    colors = np.vstack((las.red, las.green, las.blue)).T
    if colors.max() > 255:
      colors = (colors / 256).astype(np.uint8)
    else:
      colors = colors.astype(np.uint8)
  else:
    z = points[:, 2]
    gray = ((z - z.min()) / z.ptp() * 255).astype(np.uint8)
    colors = np.column_stack((gray, gray, gray))

  pose = get_camera_pose(points, colors)
  print(f"Camera position : {pose.position}")
  print(f"Camera target   : {pose.target}")

  img_rgb, visible_idx, pixel_coords = project_perspective(
    pose, points, colors, args.width, args.height, args.fov
  )
  print(f"Visible points  : {len(visible_idx):,}")

  img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
  kernel = np.ones((3, 3), np.uint8)
  img_bgr = cv2.morphologyEx(img_bgr, cv2.MORPH_CLOSE, kernel)
  img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)

  basename = os.path.splitext(os.path.basename(args.input_file))[0]
  out_img = os.path.join(args.output_dir, f"{basename}_partial.png")
  cv2.imwrite(out_img, img_bgr)
  print(f"Saved image     : {out_img}")

  if args.export_csv:
    csv_path = os.path.join(args.output_dir, f"{basename}_partial_mapping.csv")
    visible_points = points[visible_idx]
    visible_colors = colors[visible_idx]
    rows = np.column_stack((
      visible_idx.reshape(-1, 1),
      visible_points,
      visible_colors,
      pixel_coords
    ))
    with open(csv_path, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(["point_idx", "X", "Y", "Z", "R", "G", "B", "pixel_x", "pixel_y"])
      writer.writerows(rows)
    print(f"Saved CSV       : {csv_path}")


if __name__ == "__main__":
  main()
