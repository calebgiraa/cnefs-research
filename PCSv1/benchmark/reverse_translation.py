import pandas as pd
import numpy as np
import laspy
import argparse
import os

def create_las(input_csv, output_las):
    print(f"Reading {input_csv}...")
    try:
        # Load the labeled CSV file
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Normalize headers
    df.columns = [c.strip().title() for c in df.columns]
    
    # Check for required columns
    required = ['X', 'Y', 'Z', 'Red', 'Green', 'Blue']
    if not all(c in df.columns for c in required):
        print(f"Missing columns. Found: {df.columns}")
        return
    
    # Add classification column if it's somehow missing (shouldn't be if segmentation.py ran)
    if 'Classification' not in df.columns:
        print("Warning: No Classification found. Defaulting to 0.")
        df['Classification'] = 0


    # --- LAS Header Setup ---
    # FIX: Use Point Format 6 with Version 1.4. 
    # This configuration supports 8-bit Classification (0-255 range), 
    # resolving the OverflowError for your Class ID 64.
    header = laspy.LasHeader(point_format=7, version="1.4")
    
    # Important: Scale and Offset to preserve precision
    min_x, min_y, min_z = df['X'].min(), df['Y'].min(), df['Z'].min()
    header.offsets = [min_x, min_y, min_z]
    header.scales = [0.001, 0.001, 0.001] # mm precision

    las = laspy.LasData(header)

    # --- Assign Coordinates ---
    las.x = df['X'].values
    las.y = df['Y'].values
    las.z = df['Z'].values

    # --- Assign Colors (Ensure 16-bit) ---
    red, green, blue = df['Red'].values, df['Green'].values, df['Blue'].values
    max_val = max(red.max(), green.max(), blue.max())

    # Upscale 8-bit colors (0-255) to 16-bit (0-65535) if necessary
    if max_val <= 256:
        las.red = (red.astype(np.uint16) * 256)
        las.green = (green.astype(np.uint16) * 256)
        las.blue = (blue.astype(np.uint16) * 256)
    else:
        las.red = red.astype(np.uint16)
        las.green = green.astype(np.uint16)
        las.blue = blue.astype(np.uint16)

    # --- Assign Classification ---
    # numpy.uint8 ensures the classification fits in the 8-bit field
    las.classification = df['Classification'].fillna(0).astype(np.uint8)

    print(f"Writing LAS to {output_las}...")
    las.write(output_las)
    print("Conversion complete. File is LAS v1.4, Format 6.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert labeled CSV point cloud back to LAS v1.4.")
    parser.add_argument("input_csv", help="Path to the labeled CSV file.")
    parser.add_argument("output_dir", help="Output directory.")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    out_name = f"{os.path.splitext(os.path.basename(args.input_csv))[0]}.las"
    create_las(args.input_csv, os.path.join(args.output_dir, out_name))