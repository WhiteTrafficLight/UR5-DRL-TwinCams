import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import numpy as np
import os

class PointCloudSaver:
    def __init__(self, max_distance=2.0):
        self.received_pcd1 = False
        self.received_pcd2 = False
        self.points_raw1 = None
        self.points_raw2 = None
        self.max_distance = max_distance

        rospy.init_node('pointcloud_saver', anonymous=True)
        rospy.Subscriber("/camera/depth/points", PointCloud2, self.cb_get_pointcloud1)
        rospy.Subscriber("/camera2/depth/points", PointCloud2, self.cb_get_pointcloud2)
        
        self.save_path = "assets"  # Change to your desired save path

    def cb_get_pointcloud1(self, data):
        np_data = ros_numpy.numpify(data)
        points = np.zeros((np_data.shape[0] * np_data.shape[1], 3))
        points[:, 0] = np_data['x'].flatten()
        points[:, 1] = np_data['y'].flatten()
        points[:, 2] = np_data['z'].flatten()
        
        # Filter points by distance
        distances = np.linalg.norm(points, axis=1)
        points = points[distances <= self.max_distance]
        
        self.points_raw1 = points
        self.received_pcd1 = True

    def cb_get_pointcloud2(self, data):
        np_data = ros_numpy.numpify(data)
        points = np.zeros((np_data.shape[0] * np_data.shape[1], 3))
        points[:, 0] = np_data['x'].flatten()
        points[:, 1] = np_data['y'].flatten()
        points[:, 2] = np_data['z'].flatten()
        
        # Filter points by distance
        distances = np.linalg.norm(points, axis=1)
        points = points[distances <= self.max_distance]
        
        self.points_raw2 = points
        self.received_pcd2 = True

    def save_pcd_files(self):
        while not self.received_pcd1 or not self.received_pcd2:
            rospy.sleep(0.1)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(self.points_raw1)
        o3d.io.write_point_cloud(os.path.join(self.save_path, "camera1.ply"), pcd1)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(self.points_raw2)
        o3d.io.write_point_cloud(os.path.join(self.save_path, "camera2.ply"), pcd2)

        print("Saved camera1.ply and camera2.ply")

if __name__ == '__main__':
    saver = PointCloudSaver(max_distance=2.0)  # Set the maximum distance threshold as needed
    saver.save_pcd_files()
    rospy.spin()





