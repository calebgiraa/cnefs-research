import numpy as np
import cv2
import argparse
import os
import csv

from perspective import (
    load_las_as_points_and_colors,
    define_target_cube,
    create_poses,
    evaluate_poses,
    select_best_pose,
    project_pinhole,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input .las file.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("n_poses", type=int, help="Number of camera poses to evaluate.")
    parser.add_argument("radius", type=float, help="Camera distance from target.")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Target seed point (x y z). Defaults to cloud centroid.")
    parser.add_argument("--elev_range", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Elevation range in degrees (e.g. 0 45).")
    parser.add_argument("--include_topdown", action="store_true",
                        help="Include an explicit top-down camera pose.")
    parser.add_argument("--width", type=int, default=None,
                        help="Image width in pixels (default: 16:9 from height).")
    parser.add_argument("--height", type=int, default=800, help="Image height in pixels.")
    parser.add_argument("--fov", type=float, default=60.0, help="Vertical FOV in degrees.")
    parser.add_argument("--export_csv", action="store_true",
                        help="Export point-to-pixel mapping CSV.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.input_file}...")
    points, colors = load_las_as_points_and_colors(args.input_file)

    seed = np.array(args.target) if args.target is not None else np.mean(points, axis=0)
    target_centroid, target_mask = define_target_cube(points, seed)

    elev_range = (
        np.deg2rad(args.elev_range) if args.elev_range is not None
        else (0.0, np.deg2rad(45.0))
    )
    poses = create_poses(
        target_centroid, args.n_poses, args.radius,
        elev_range=elev_range, include_topdown=args.include_topdown,
    )

    results = evaluate_poses(poses, points, target_mask)
    best = select_best_pose(results)
    best_pose = best["pose"]
    print(f"Best pose J_total={best['J_total']:.4f}  J_occ={best['J_occ']:.4f}  J_persp={best['J_persp']:.4f}")
    print(f"Camera position : {best_pose['position']}")

    image, _, index_map = project_pinhole(
        points=points, colors=colors, pose=best_pose,
        img_h=args.height, img_w=args.width, fov_deg=args.fov,
    )

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    basename = os.path.splitext(os.path.basename(args.input_file))[0]
    out_img = os.path.join(args.output_dir, f"{basename}_partial.png")
    cv2.imwrite(out_img, image)
    print(f"Saved image     : {out_img}")

    out_npy = os.path.join(args.output_dir, f"{basename}_index_map.npy")
    np.save(out_npy, index_map)
    print(f"Saved index map : {out_npy}")

    if args.export_csv:
        csv_path = os.path.join(args.output_dir, f"{basename}_partial_mapping.csv")
        iy, ix = np.where(index_map >= 0)
        point_indices = index_map[iy, ix]
        rows = np.column_stack((
            point_indices,
            points[point_indices],
            colors[point_indices],
            ix.reshape(-1, 1),
            iy.reshape(-1, 1),
        ))
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["point_idx", "X", "Y", "Z", "R", "G", "B", "pixel_x", "pixel_y"])
            writer.writerows(rows)
        print(f"Saved CSV       : {csv_path}")


if __name__ == "__main__":
    main()
