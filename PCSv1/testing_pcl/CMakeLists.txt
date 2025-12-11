cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(euclidean_cluster)

# Find the PCL packages
find_package(PCL 1.2 REQUIRED)

# Include directories
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Create the executable
add_executable(euclidean_cluster euclidean_cluster.cpp)

# Link the executable to PCL libraries
target_link_libraries(euclidean_cluster ${PCL_LIBRARIES})