#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>        // For converting between ROS and PCL
#include <pcl/point_types.h>            // Point type definitions
#include <pcl/filters/voxel_grid.h>     // VoxelGrid filter
#include <pcl_conversions/pcl_conversions.h> // for pcl::fromROSMsg, pcl::toROSMsg

// Define a point cloud type; use PointXYZRGB if you have color information
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

ros::Publisher pub;

// Callback function to process the incoming point cloud
void cloudCallback (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    // Convert the ROS message to a PCL point cloud
    PointCloud::Ptr cloud (new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    ROS_INFO("Received cloud with %lu points", cloud->points.size());

    // Create a new point cloud to hold the filtered data
    PointCloud::Ptr cloud_filtered (new PointCloud);

    // Create the filtering object and set parameters
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    // Set the voxel (leaf) size. Adjust the values for your specific use-case.
    voxel_filter.setLeafSize(0.05f, 0.05f, 0.05f);
    voxel_filter.filter(*cloud_filtered);

    ROS_INFO("Filtered cloud has %lu points", cloud_filtered->points.size());

    // Convert the filtered PCL point cloud back to ROS message format
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud_filtered, output);
    output.header = cloud_msg->header; // preserve the header from the input message
    pub.publish(output);
}

int main (int argc, char** argv)
{
    ros::init(argc, argv, "point_cloud_voxelizer");
    ros::NodeHandle nh;

    // Subscribe to the input point cloud topic (adjust the topic name if needed)
    ros::Subscriber sub = nh.subscribe("/camera/depth/points", 1, cloudCallback);

    // Advertise the output (filtered) point cloud topic
    pub = nh.advertise<sensor_msgs::PointCloud2>("/camera/depth/points_voxelized", 1);

    ros::spin();
    return 0;
}
