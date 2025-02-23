#!/usr/bin/env python

import rospy
import torch
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

def load_point_cloud(pth_file):
    """
    Load a point cloud from a .pth file.
    """
    point_cloud = torch.load(pth_file).astype(np.float32)
    return point_cloud

def point_cloud_to_pointcloud2(points, frame_id="map"):
    """
    Convert a numpy array of points to a PointCloud2 message.
    """
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]

    point_cloud2_msg = pc2.create_cloud(header, fields, points)
    return point_cloud2_msg

def publish_point_clouds(pth_file1, pth_file2):
    """
    Load and publish point clouds from two .pth files.
    """
    rospy.init_node('pointcloud_publisher', anonymous=True)
    pub1 = rospy.Publisher('/camera/depth/points', PointCloud2, queue_size=10)
    pub2 = rospy.Publisher('/camera2/depth/points', PointCloud2, queue_size=10)

    point_cloud1_np = load_point_cloud(pth_file1)
    point_cloud2_np = load_point_cloud(pth_file2)

    #point_cloud1_np = point_cloud1.cpu().numpy()
    #point_cloud2_np = point_cloud2.cpu().numpy()

    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        point_cloud2_msg1 = point_cloud_to_pointcloud2(point_cloud1_np)
        point_cloud2_msg2 = point_cloud_to_pointcloud2(point_cloud2_np)

        pub1.publish(point_cloud2_msg1)
        pub2.publish(point_cloud2_msg2)

        rate.sleep()

if __name__ == '__main__':
    try:
        pth_file1 = "assets/cloud_bin_21.pth"
        pth_file2 = "assets/cloud_bin_34.pth"
        publish_point_clouds(pth_file1, pth_file2)
    except rospy.ROSInterruptException:
        pass

