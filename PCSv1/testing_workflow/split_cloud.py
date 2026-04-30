import numpy as np
import laspy
import argparse
import os


def factor_grid(n):
    """Return (rows, cols) that multiply to n and are as square as possible."""
    best = (1, n)
    for r in range(1, int(n ** 0.5) + 1):
        if n % r == 0:
            c = n // r
            if abs(r - c) < abs(best[0] - best[1]):
                best = (r, c)
    return best


def split_cloud(input_file, output_dir, n_tiles):
    print(f"Loading {input_file}...")
    las = laspy.read(input_file)

    x = np.array(las.x)
    y = np.array(las.y)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    rows, cols = factor_grid(n_tiles)
    print(f"Splitting into {rows} row(s) x {cols} col(s) = {n_tiles} tiles")
    print(f"XY extent: X=[{x_min:.2f}, {x_max:.2f}]  Y=[{y_min:.2f}, {y_max:.2f}]")

    x_edges = np.linspace(x_min, x_max, cols + 1)
    y_edges = np.linspace(y_min, y_max, rows + 1)

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_file))[0]

    tile_num = 0
    for row in range(rows):
        for col in range(cols):
            x0, x1 = x_edges[col], x_edges[col + 1]
            y0, y1 = y_edges[row], y_edges[row + 1]

            # Include the upper boundary only on the last tile to avoid gaps
            if col < cols - 1:
                x_mask = (x >= x0) & (x < x1)
            else:
                x_mask = (x >= x0) & (x <= x1)

            if row < rows - 1:
                y_mask = (y >= y0) & (y < y1)
            else:
                y_mask = (y >= y0) & (y <= y1)

            mask = x_mask & y_mask
            count = mask.sum()

            out_path = os.path.join(output_dir, f"{basename}_tile_{tile_num:03d}.las")

            if count == 0:
                print(f"  Tile {tile_num:03d} (row={row}, col={col}): 0 points — skipping")
                tile_num += 1
                continue

            subset = las.points[mask]
            new_las = laspy.LasData(header=las.header)
            new_las.points = subset
            new_las.write(out_path)

            print(f"  Tile {tile_num:03d} (row={row}, col={col}): {count} points -> {out_path}")
            tile_num += 1

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Split a .las point cloud into N equal spatial tiles.")
    parser.add_argument("input_file", help="Path to input .las file.")
    parser.add_argument("output_dir", help="Directory to write tile files into.")
    parser.add_argument("--n_tiles", type=int, required=True,
                        help="Number of tiles to produce. The cloud is divided into a grid "
                             "whose dimensions are the most square factoring of n_tiles "
                             "(e.g. 10 -> 2x5, 9 -> 3x3, 6 -> 2x3).")
    args = parser.parse_args()
    split_cloud(args.input_file, args.output_dir, args.n_tiles)


if __name__ == "__main__":
    main()
