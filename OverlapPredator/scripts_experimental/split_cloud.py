"""
This script was developed to experiment with our own generated point clouds. 
It loads a point cloud file ("assets/park1.ply"), splits it into multiple overlapping fragments 
based on a specified grid size and overlap ratio, and then saves each fragment as a separate file.
The purpose is to test our model's performance on registration using these smaller fragments.
 
Usage:
    Run this script to generate fragment files (e.g., fragment_1.ply, fragment_2.ply, ...) for further experiments.
    
Note:
    Adjust grid size and overlap ratio as needed.
    
Author: WhiteTrafficLight
Last modified: 23.02.2025    
"""

import open3d as o3d
import numpy as np

def load_point_cloud(filename):
    return o3d.io.read_point_cloud(filename)

def save_point_cloud(point_cloud, filename):
    o3d.io.write_point_cloud(filename, point_cloud)

def split_point_cloud_for_registration(pcd, grid_size, overlap_ratio):
    # Calculate grid steps based on point cloud size and desired grid size and overlap
    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    
    step_x = (max_bound[0] - min_bound[0]) / grid_size[0]
    step_y = (max_bound[1] - min_bound[1]) / grid_size[1]
    step_z = (max_bound[2] - min_bound[2]) / grid_size[2]
    
    overlap_x = step_x * overlap_ratio
    overlap_y = step_y * overlap_ratio
    overlap_z = step_z * overlap_ratio
    
    fragments = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for k in range(grid_size[2]):
                # Compute overlapping bounds
                x_min = max(min_bound[0] + i * step_x - overlap_x, min_bound[0])
                x_max = min(min_bound[0] + (i + 1) * step_x + overlap_x, max_bound[0])
                y_min = max(min_bound[1] + j * step_y - overlap_y, min_bound[1])
                y_max = min(min_bound[1] + (j + 1) * step_y + overlap_y, max_bound[1])
                z_min = max(min_bound[2] + k * step_z - overlap_z, min_bound[2])
                z_max = min(min_bound[2] + (k + 1) * step_z + overlap_z, max_bound[2])
                
                # Crop point cloud
                box = o3d.geometry.AxisAlignedBoundingBox([x_min, y_min, z_min], [x_max, y_max, z_max])
                fragment = pcd.crop(box)
                fragments.append(fragment)
    
    return fragments

# Main program
if __name__ == "__main__":
    pcd = load_point_cloud("assets/park1.ply")
    # Define grid size (number of fragments in x, y, z) and overlap ratio
    fragments = split_point_cloud_for_registration(pcd, (2, 2, 2), 0.2)
    for index, fragment in enumerate(fragments):
        save_point_cloud(fragment, f"fragment_{index+1}.ply")
