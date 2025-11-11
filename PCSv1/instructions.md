# Instructions to deploy PCSv1 locally
## Step 1: Conda environment set-up
```
conda env create --name PCSv1 --file requirements.txt
```
(make sure your cwd is cnefs-research)
- After this, you should change directories to the 'benchmark' directory within PCSv1
## Step 2: Running translation.py
- To run translation.py, make sure you are in the benchmark directory and run the following command:
```
python translation.py [path_to_input_lidar] [path_to_image] --type ortho --res 0.10
```

- For instance, when I run translation, I put it as so:
```
python translation.py ./input_lidar/Lab1_Scan_125.las ./input_images/ --type ortho --res 0.10
```

