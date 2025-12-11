#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h> // REQUIRED for centering
#include <pcl/common/transforms.h> // REQUIRED for shifting
#include <iostream>

int main(int argc, char** argv)
{
    // 1. Load the converted PCD file
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // Change this filename if needed, or pass as argv[1]
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("output_file.pcd", *cloud) == -1) {
        PCL_ERROR("Couldn't read file \n");
        return -1;
    }
    std::cout << "Loaded " << cloud->points.size() << " points." << std::endl;

    // --- NEW: CENTER THE CLOUD TO FIX VOXEL GRID OVERFLOW ---
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    
    // Create a translation matrix to move cloud to (0,0,0)
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << -centroid[0], -centroid[1], -centroid[2];
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *cloud_centered, transform);
    std::cout << "Cloud centered. Moved from " << centroid[0] << ", " << centroid[1] << " to 0,0,0." << std::endl;
    // --------------------------------------------------------

    // 2. Downsample (VoxelGrid Filter)
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud(cloud_centered); // Use the centered cloud!
    
    // ADJUST THIS: If units are Meters, 0.05 is 5cm. 
    // If units are Feet, 0.05 is 0.6 inches.
    // If units are Millimeters (common in some LAS), 0.05 is microscopic -> ERROR.
    vg.setLeafSize(0.1f, 0.1f, 0.1f); 
    vg.filter(*cloud_filtered);
    
    std::cout << "Downsampled to " << cloud_filtered->points.size() << " points." << std::endl;

    // Safety Check: If VoxelGrid failed, stop here.
    if (cloud_filtered->points.empty()) {
        std::cerr << "Error: VoxelGrid returned 0 points. Check leaf size or units." << std::endl;
        return -1;
    }

    // 3. Remove Ground Plane (RANSAC)
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.3); 
    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers, *coefficients);

    // Extract Objects (Remove Ground)
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud_filtered);
    std::cout << "Ground removed. Points remaining: " << cloud_filtered->points.size() << std::endl;

    // 4. Euclidean Clustering
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    
    ec.setClusterTolerance(0.5); 
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(50000); // Increased max size for your large file
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);

    // 5. Save Clusters
    int j = 0;
    for (const auto& indices : cluster_indices)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& index : indices.indices)
            cloud_cluster->points.push_back(cloud_filtered->points[index]); 

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        std::string filename = "cluster_" + std::to_string(j) + ".pcd";
        pcl::io::savePCDFileASCII(filename, *cloud_cluster);
        j++;
    }
    std::cout << "Found " << j << " clusters." << std::endl;

    return 0;
}