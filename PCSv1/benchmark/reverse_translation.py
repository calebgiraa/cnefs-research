import pandas as pd
import laspy
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to the labeled CSV file.")
    parser.add_argument("output_dir", help="Directory to save the .las file.")
    args = parser.parse_args()

    print(f"Reading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    # Map columns dynamically
    col_map = {c.upper(): c for c in df.columns}
    
    try:
        x, y, z = df[col_map['X']], df[col_map['Y']], df[col_map['Z']]
        r_col = col_map.get('R') or col_map.get('RED')
        g_col = col_map.get('G') or col_map.get('GREEN')
        b_col = col_map.get('B') or col_map.get('BLUE')
        class_col = col_map.get('CLASSIFICATION')
    except KeyError:
        print(f"Error: Missing columns. Found: {list(df.columns)}")
        return

    # Create LAS 1.4 Header with Point Format 7 (Supports RGB + 8-bit Class)
    header = laspy.LasHeader(point_format=7, version="1.4")
    las = laspy.LasData(header)

    las.x, las.y, las.z = x, y, z

    # RGB in LAS is 16-bit (0-65535). Scale 8-bit CSV values (0-255).
    las.red = (df[r_col].values * 256).astype(np.uint16)
    las.green = (df[g_col].values * 256).astype(np.uint16)
    las.blue = (df[b_col].values * 256).astype(np.uint16)

    # Assign Classifications (now supports 64 without overflow)
    las.classification = df[class_col].values.astype(np.uint8)

    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.basename(args.input_csv).replace("_labeled.csv", "_segmented.las")
    out_path = os.path.join(args.output_dir, basename)
    
    las.write(out_path)
    print(f"Successfully created: {out_path}")
    print(f"Total points with Class 64: {np.sum(las.classification == 64)}")

if __name__ == "__main__":
    main()