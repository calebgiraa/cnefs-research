# cnefs-research
This repository contains all of the different models the CNEFS lab has been creating in effort to create NRC guidelines for AI implementation. Please note that all models may be subject to change as we are in early stages of what will be years of research.
## HTTP_Flask
This simple server keeps track of images transferred from one device to another in real-time. We hope to one day deploy this onto the robot, ensuring that images taken by the robot are transferred to another, more powerful machine that's capable of running the models for segmentation and defect detection.
## Benchmark - Fall 2025
As of 9/26/2025, use the requirements.txt and download.txt files to create your environment and model necessary for the benchmark. The benchmark has been tested with CUDA 12.8 on CPU-mode due to current hardware constraints. Grounded SAM is being used for segmentation, and various algorithms are in use to translate a point cloud into a equirectangular image.
