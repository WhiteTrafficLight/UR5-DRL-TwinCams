#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;

void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    // Convert ROS message to PCL point cloud
    PointCloudRGB::Ptr cloud(new PointCloudRGB);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Statistical Outlier Removal 
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(20);               
    sor.setStddevMulThresh(2.0);
    sor.filter(*cloud);

    ROS_INFO("Processed cloud with %zu points", cloud->points.size());
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "point_cloud_processor");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe("/camera/depth/points", 1, pointCloudCallback);

    ros::spin();
    return 0;
}
