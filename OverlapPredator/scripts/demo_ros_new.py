import os
import torch
import time
import shutil
import json
import glob
import sys
import copy
import argparse
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch import optim, nn
import open3d as o3d
import rospy
import ros_numpy
import threading
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

cwd = os.getcwd()
sys.path.append(cwd)
from datasets.indoor import IndoorDataset
from datasets.dataloader import get_dataloader
from models.architectures import KPFCNN
from lib.utils import load_obj, setup_seed, natural_key, load_config
from lib.benchmark_utils import ransac_pose_estimation, to_o3d_pcd, get_blue, get_yellow, to_tensor
from lib.trainer import Trainer
from lib.loss import MetricLoss
import sensor_msgs.point_cloud2 as pc2


setup_seed(0)

class ThreeDMatchDemo(Dataset):
    def __init__(self, config, src_pcd, tgt_pcd):
        super(ThreeDMatchDemo, self).__init__()
        self.config = config
        self.src_pcd = src_pcd
        self.tgt_pcd = tgt_pcd

    def __len__(self):
        return 1

    def __getitem__(self, item):
        src_pcd = self.src_pcd
        tgt_pcd = self.tgt_pcd

        # Downsample point clouds
        src_pcd = src_pcd.voxel_down_sample(0.025)
        tgt_pcd = tgt_pcd.voxel_down_sample(0.025)
        src_pcd = np.array(src_pcd.points).astype(np.float32)
        tgt_pcd = np.array(tgt_pcd.points).astype(np.float32)

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)

        # Fake the ground truth information
        rot = np.eye(3).astype(np.float32)
        trans = np.ones((3, 1)).astype(np.float32)
        correspondences = torch.ones(1, 2).long()

        return src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)

class PoseEstimatorNode:
    def __init__(self, config):
        self.config = config
        self.received_pcd1 = False
        self.received_pcd2 = False
        self.points_raw1 = None
        self.points_raw2 = None
        self.points_raw1_uncut = None
        self.points_raw2_uncut = None
        self.static = True
        self.tsfm = None


        rospy.init_node('pointcloud_matcher', anonymous=True)
        rospy.Subscriber("/camera/depth/points", PointCloud2, self.cb_get_pointcloud1)
        rospy.Subscriber("/camera2/depth/points", PointCloud2, self.cb_get_pointcloud2)
        self.pub = rospy.Publisher('/combined_pointcloud', PointCloud2, queue_size=10)

        # Load pretrained weights
        assert self.config.pretrain is not None
        state = torch.load(config.pretrain, map_location=torch.device('cuda'))
        config.model = KPFCNN(config).to(config.device)
        config.model.load_state_dict(state['state_dict'])

        # Create dataset and dataloader
        self.info_train = load_obj(config.train_info)
        self.train_set = IndoorDataset(self.info_train, config, data_augmentation=True)
        _, self.neighborhood_limits = get_dataloader(dataset=self.train_set,
                                                     batch_size=config.batch_size,
                                                     shuffle=True,
                                                     num_workers=config.num_workers)
        self.demo_loader = None

        # Start the processing thread
        self.processing_thread = threading.Thread(target=self.process_pointclouds)
        self.processing_thread.start()

    def cb_get_pointcloud1(self, data):
        np_data = ros_numpy.numpify(data)
        points = np.ones((np_data.shape[0]*np_data.shape[1], 3))
        if len(points) == 0:
            print("[cbGetPointcloud] No point cloud data received, check the camera and its ROS program!")
            return    
        points[:, 0] = np_data['x'].flatten()
        points[:, 1] = np_data['y'].flatten()
        points[:, 2] = np_data['z'].flatten()

        # Filter out NaN values
        #mask = ~np.isnan(points).any(axis=1)
        #points = points[mask] 
        
        #points = []
        #for point in ros_numpy.point_cloud2.pointcloud2_to_array(data):
        #    points.append([point[0], point[1], point[2]])

        #points = np.array(points)
        self.points_raw1_uncut = points
        distances = np.linalg.norm(points, axis=1)
        points = points[distances <= 2.5]

        self.points_raw1 = points
        self.received_pcd1 = True

    def cb_get_pointcloud2(self, data):
        np_data = ros_numpy.numpify(data)
        points = np.ones((np_data.shape[0]*np_data.shape[1], 3))
        if len(points) == 0:
            print("[cbGetPointcloud] No point cloud data received, check the camera and its ROS program!")
            return    
        points[:, 0] = np_data['x'].flatten()
        points[:, 1] = np_data['y'].flatten()
        points[:, 2] = np_data['z'].flatten()

        #mask = ~np.isnan(points).any(axis=1)
        #points = points[mask] 
        #points = []
        #for point in ros_numpy.point_cloud2.pointcloud2_to_array(data):
        #    points.append([point[0], point[1], point[2]])

        #points = np.array(points)
        self.points_raw2_uncut = points
        distances = np.linalg.norm(points, axis=1)
        points = points[distances <= 2.5]

        self.points_raw2 = points
        self.received_pcd2 = True

    def process_pointclouds(self):
        while not rospy.is_shutdown():
            if self.received_pcd1 and self.received_pcd2:
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(self.points_raw1)
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(self.points_raw2)

                demo_set = ThreeDMatchDemo(self.config, pcd2, pcd1)
                self.demo_loader, _ = get_dataloader(dataset=demo_set,
                                                     batch_size=self.config.batch_size,
                                                     shuffle=False,
                                                     num_workers=1,
                                                     neighborhood_limits=self.neighborhood_limits)

                self.estimate_pose()

                # Reset flags
                self.received_pcd1 = False
                self.received_pcd2 = False

    def estimate_pose(self):
        if self.static and self.tsfm is not None:
            self.publish_combined_pointcloud(self.points_raw2, self.points_raw1, self.tsfm)
            return
        self.config.model.eval()
        c_loader_iter = iter(self.demo_loader)
        with torch.no_grad():
            inputs = next(c_loader_iter)
            for k, v in inputs.items():
                if isinstance(v, list):
                    inputs[k] = [item.to(self.config.device) for item in v]
                else:
                    inputs[k] = v.to(self.config.device)

            feats, scores_overlap, scores_saliency = self.config.model(inputs)
            pcd = inputs['points'][0]
            len_src = inputs['stack_lengths'][0][0]
            src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
            src_feats, tgt_feats = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
            src_overlap, src_saliency = scores_overlap[:len_src].detach().cpu(), scores_saliency[:len_src].detach().cpu()
            tgt_overlap, tgt_saliency = scores_overlap[len_src:].detach().cpu(), scores_saliency[len_src:].detach().cpu()

            src_scores = src_overlap * src_saliency
            tgt_scores = tgt_overlap * tgt_saliency

            if src_pcd.size(0) > self.config.n_points:
                idx = np.arange(src_pcd.size(0))
                probs = (src_scores / src_scores.sum()).numpy().flatten()
                idx = np.random.choice(idx, size=self.config.n_points, replace=False, p=probs)
                src_pcd, src_feats = src_pcd[idx], src_feats[idx]
            if tgt_pcd.size(0) > self.config.n_points:
                idx = np.arange(tgt_pcd.size(0))
                probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
                idx = np.random.choice(idx, size=self.config.n_points, replace=False, p=probs)
                tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

            tsfm = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False)
            #print(tsfm)
            self.tsfm = tsfm
            self.publish_combined_pointcloud(self.points_raw2, self.points_raw1, tsfm)

    def publish_combined_pointcloud(self, src_pcd, tgt_pcd, tsfm):
        # Apply transformation to src_pcd
        src_pcd_hom = np.hstack((src_pcd, np.ones((src_pcd.shape[0], 1))))
        src_pcd_transformed = (tsfm @ src_pcd_hom.T).T[:, :3]

        # Combine src_pcd and tgt_pcd
        combined_pcd = np.vstack((src_pcd_transformed, tgt_pcd))

        # Convert to PointCloud2 message
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'camera_depth_optical_frame'
        combined_msg = pc2.create_cloud_xyz32(header, combined_pcd.tolist())

        # Publish the combined point cloud
        self.pub.publish(combined_msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file.')
    args = parser.parse_args()

    config = load_config(args.config)
    config = edict(config)
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    # Model initialization
    config.architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers - 1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
        config.architecture.append('resnetb')
    for i in range(config.num_layers - 2):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
    config.architecture.append('nearest_upsample')
    config.architecture.append('last_unary')
    config.model = KPFCNN(config).to(config.device)

    pose_estimator = PoseEstimatorNode(config)
    rospy.spin()



