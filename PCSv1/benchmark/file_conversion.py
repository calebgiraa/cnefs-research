import numpy as np
import laspy
from plyfile import PlyData, PlyElement
import argparse
import os

def convert_ply_to_las(ply_path, las_path):
    """
    Converts a .ply point cloud file to a .las file.

    This function reads vertex data (X, Y, Z, and optionally Red, Green, Blue)
    from a PLY file and writes it to a LAS file, handling the necessary
    scaling for color values.

    Args:
        ply_path (str): The full path to the input .ply file.
        las_path (str): The full path for the output .las file.
    """
    # --- 1. Read the PLY file ---
    print(f"Reading data from {ply_path}...")
    try:
        ply_data = PlyData.read(ply_path)
        # ⭐️ FIX: Get the numpy array from the '.data' attribute
        vertices = ply_data['vertex'].data
    except Exception as e:
        print(f"Error: Could not read PLY file. {e}")
        return

    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).transpose()

    # --- 2. Check for color data and set the LAS point format ---
    # This check will now work correctly on the numpy array's dtype
    has_color = all(prop in vertices.dtype.names for prop in ['red', 'green', 'blue'])
    
    if has_color:
        print("Color information found. Creating LAS file with RGB data (Point Format 3).")
        # LAS format requires 16-bit color, PLY is often 8-bit (0-255)
        # We scale the 8-bit colors to 16-bit by multiplying by 256.
        red = (vertices['red']).astype(np.uint16) * 256
        green = (vertices['green']).astype(np.uint16) * 256
        blue = (vertices['blue']).astype(np.uint16) * 256
        point_format_id = 3
    else:
        print("No color information found. Creating LAS file without RGB data (Point Format 0).")
        point_format_id = 0

    # --- 3. Create the LAS file ---
    print("Creating LAS file header...")
    header = laspy.LasHeader(point_format=point_format_id, version="1.2")
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    if has_color:
        las.red = red
        las.green = green
        las.blue = blue

    # --- 4. Write the LAS file to disk ---
    print(f"Writing data to {las_path}...")
    las.write(las_path)
    print(f"✅ Successfully converted file to {las_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert a .ply file to a .las file.")
    parser.add_argument("input_ply", help="Path to the input .ply file.")
    parser.add_argument("output_las", help="Path for the output .las file.")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_ply):
        print(f"Error: Input file not found at {args.input_ply}")
        return
        
    convert_ply_to_las(args.input_ply, args.output_las)


if __name__ == "__main__":
    main()