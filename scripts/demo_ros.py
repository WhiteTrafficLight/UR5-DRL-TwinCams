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

setup_seed(0)

class PointCloudMatcher:
    def __init__(self, config):
        self.config = config
        self.received_pcd1 = False
        self.received_pcd2 = False
        self.points_raw1 = None
        self.points_raw2 = None

        rospy.init_node('pointcloud_matcher', anonymous=True)
        rospy.Subscriber("/camera/depth/points", PointCloud2, self.cb_get_pointcloud1)
        rospy.Subscriber("/camera2/depth/points", PointCloud2, self.cb_get_pointcloud2)
        self.pub = rospy.Publisher('/combined_pointcloud', PointCloud2, queue_size=10)


    def cb_get_pointcloud1(self, data):
        np_data = ros_numpy.numpify(data)
        points = np.zeros((np_data.shape[0] * np_data.shape[1], 3))
        points[:, 0] = np_data['x'].flatten()
        points[:, 1] = np_data['y'].flatten()
        points[:, 2] = np_data['z'].flatten()
        
        # Filter points by distance from the camera (0,0,0)
        distances = np.linalg.norm(points, axis=1)
        points = points[distances <= 2.0]
        
        self.points_raw1 = points
        self.received_pcd1 = True

    def cb_get_pointcloud2(self, data):
        np_data = ros_numpy.numpify(data)
        points = np.zeros((np_data.shape[0] * np_data.shape[1], 3))
        points[:, 0] = np_data['x'].flatten()
        points[:, 1] = np_data['y'].flatten()
        points[:, 2] = np_data['z'].flatten()
        
        # Filter points by distance from the camera (0,0,0)
        distances = np.linalg.norm(points, axis=1)
        points = points[distances <= 2.0]
        
        self.points_raw2 = points
        self.received_pcd2 = True

    def process_pointclouds(self):
        while not self.received_pcd1 or not self.received_pcd2:
            rospy.sleep(0.1)
        
        # Preprocess point clouds
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(self.points_raw1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(self.points_raw2)

        # Downsample point clouds
        pcd1 = pcd1.voxel_down_sample(0.025)
        pcd2 = pcd2.voxel_down_sample(0.025)

        # Convert to numpy arrays
        src_pcd = np.array(pcd1.points).astype(np.float32)
        tgt_pcd = np.array(pcd2.points).astype(np.float32)

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)

        inputs = {
            'src_pcd': torch.tensor(src_pcd).to(self.config.device),
            'tgt_pcd': torch.tensor(tgt_pcd).to(self.config.device),
            'src_feats': torch.tensor(src_feats).to(self.config.device),
            'tgt_feats': torch.tensor(tgt_feats).to(self.config.device),
            'features': torch.cat((torch.tensor(src_feats), torch.tensor(tgt_feats))).to(self.config.device)
        }

        # Perform pose estimation
        with torch.no_grad():
            feats, scores_overlap, scores_saliency = self.config.model(inputs)
            src_pcd_torch = inputs['src_pcd']
            tgt_pcd_torch = inputs['tgt_pcd']
            src_feats_torch = feats[:src_pcd_torch.size(0)].detach().cpu()
            tgt_feats_torch = feats[src_pcd_torch.size(0):].detach().cpu()

            tsfm = ransac_pose_estimation(src_pcd_torch, tgt_pcd_torch, src_feats_torch, tgt_feats_torch, mutual=False)
            print(tsfm)

            # Transform the source point cloud
            src_pcd_transformed = to_o3d_pcd(src_pcd)
            src_pcd_transformed.transform(tsfm)

            # Combine the transformed source point cloud with the target point cloud
            tgt_pcd_o3d = to_o3d_pcd(tgt_pcd)
            combined_pcd = tgt_pcd_o3d + src_pcd_transformed

            # Convert combined point cloud to numpy array
            combined_points = np.asarray(combined_pcd.points)

            # Publish the combined point cloud
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "base_link"
            self.publish_pointcloud(combined_points, header)

    def publish_pointcloud(self, points, header):
        pc2 = ros_numpy.point_cloud2.array_to_pointcloud2(points, header=header)
        self.pub.publish(pc2)



def main(config):
    matcher = PointCloudMatcher(config)
    matcher.process_pointclouds()

if __name__ == '__main__':
    # Load configs
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

    # Create dataset and dataloader
    info_train = load_obj(config.train_info)
    train_set = IndoorDataset(info_train, config, data_augmentation=True)
    _, neighborhood_limits = get_dataloader(dataset=train_set,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=config.num_workers)

    # Load pretrained weights
    assert config.pretrain is not None
    state = torch.load(config.pretrain, map_location=torch.device('cuda'))
    config.model.load_state_dict(state['state_dict'])

    # Process point clouds
    main(config)
    rospy.spin()











