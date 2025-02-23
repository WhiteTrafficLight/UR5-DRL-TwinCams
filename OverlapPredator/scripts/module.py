import torch
import numpy as np
import os, sys, time, json, copy
from easydict import EasyDict as edict
from sklearn.neighbors import NearestNeighbors


from models.architectures import KPFCNN
from lib.utils import load_obj, setup_seed, load_config
from lib.benchmark_utils import ransac_pose_estimation

setup_seed(0)

class PointCloudMatcher:
    def __init__(self, config_path, pretrain_path, device='cpu'):
        self.config = self.load_config(config_path)
        self.config = edict(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.config.device = self.device
        
        self.model = self.initialize_model(pretrain_path)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    def initialize_model(self, pretrain_path):
        self.config.architecture = ['simple', 'resnetb']
        for i in range(self.config.num_layers - 1):
            self.config.architecture.extend(['resnetb_strided', 'resnetb', 'resnetb'])
        for i in range(self.config.num_layers - 2):
            self.config.architecture.extend(['nearest_upsample', 'unary'])
        self.config.architecture.extend(['nearest_upsample', 'last_unary'])
        
        model = KPFCNN(self.config).to(self.device)
        state = torch.load(pretrain_path, map_location=self.device)
        model.load_state_dict(state['state_dict'])
        model.eval()
        return model

    def process_point_clouds(self, src_pcd, tgt_pcd):
        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)

        src_pcd = torch.tensor(src_pcd, device=self.device)
        tgt_pcd = torch.tensor(tgt_pcd, device=self.device)
        src_feats = torch.tensor(src_feats, device=self.device)
        tgt_feats = torch.tensor(tgt_feats, device=self.device)

        return src_pcd, tgt_pcd, src_feats, tgt_feats

    def estimate_transform(self, src_pcd, tgt_pcd):
        src_pcd, tgt_pcd, src_feats, tgt_feats = self.process_point_clouds(src_pcd, tgt_pcd)

        with torch.no_grad():
            feats = torch.cat((src_feats, tgt_feats), dim=0)
            pcd = torch.cat((src_pcd, tgt_pcd), dim=0)
            len_src = src_pcd.size(0)

            src_feats, tgt_feats = feats[:len_src], feats[len_src:]

            # Forward pass through the model
            inputs = {'points': [pcd], 'stack_lengths': [[len_src, tgt_pcd.size(0)]]}
            feats, scores_overlap, scores_saliency = self.model(inputs)
            
            src_scores = scores_overlap[:len_src] * scores_saliency[:len_src]
            tgt_scores = scores_overlap[len_src:] * scores_saliency[len_src:]

            if src_pcd.size(0) > self.config.n_points:
                idx = np.arange(src_pcd.size(0))
                probs = (src_scores / src_scores.sum()).cpu().numpy().flatten()
                idx = np.random.choice(idx, size=self.config.n_points, replace=False, p=probs)
                src_pcd, src_feats = src_pcd[idx], src_feats[idx]
            if tgt_pcd.size(0) > self.config.n_points:
                idx = np.arange(tgt_pcd.size(0))
                probs = (tgt_scores / tgt_scores.sum()).cpu().numpy().flatten()
                idx = np.random.choice(idx, size=self.config.n_points, replace=False, p=probs)
                tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

            src_feats = src_feats.detach().cpu().numpy()
            tgt_feats = tgt_feats.detach().cpu().numpy()

            # Run RANSAC to find the best transformation
            tsfm = ransac_pose_estimation(src_pcd.cpu().numpy(), tgt_pcd.cpu().numpy(), src_feats, tgt_feats, mutual=False)

            return tsfm

# Usage Example
if __name__ == '__main__':
    config_path = 'path/to/config.json'
    pretrain_path = 'path/to/pretrained.pth'
    matcher = PointCloudMatcher(config_path, pretrain_path, device='cuda')

    # Example point clouds (replace with actual point cloud data)
    src_pcd = np.random.rand(1000, 3)  # Replace with actual source point cloud
    tgt_pcd = np.random.rand(1000, 3)  # Replace with actual target point cloud

    transform_matrix = matcher.estimate_transform(src_pcd, tgt_pcd)
    print("Transformation Matrix:\n", transform_matrix)

