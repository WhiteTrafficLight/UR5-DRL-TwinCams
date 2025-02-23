"""
Scripts for pairwise registration demo

Author: Shengyu Huang
Last modified: 22.02.2021
"""
import os, torch, time, shutil, json,glob,sys,copy, argparse
import numpy as np
import igraph
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch import optim, nn
import open3d as o3d

from sklearn.neighbors import NearestNeighbors


cwd = os.getcwd()
sys.path.append(cwd)
print(cwd)

from datasets.kitti import KITTIDataset
from datasets.dataloader import get_dataloader, get_datasets
from models.architectures import KPFCNN
from lib.utils import load_obj, setup_seed,natural_key, load_config
from lib.benchmark_utils import ransac_pose_estimation, to_o3d_pcd, get_blue, get_yellow, to_tensor
from lib.trainer import Trainer
from lib.loss import MetricLoss
import shutil
setup_seed(0)


class KittiDemo(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """
    def __init__(self,config, src_path, tgt_path):
        super(KittiDemo,self).__init__()
        self.config = config
        self.src_path = src_path
        self.tgt_path = tgt_path

    def __len__(self):
        return 1

    def __getitem__(self,item): 
        # get pointcloud
        #src_pcd = torch.load(self.src_path).astype(np.float32)
        #tgt_pcd = torch.load(self.tgt_path).astype(np.float32)   
        
        
        src_pcd = o3d.io.read_point_cloud(self.src_path)
        tgt_pcd = o3d.io.read_point_cloud(self.tgt_path)
        src_pcd = src_pcd.voxel_down_sample(0.025)
        tgt_pcd = tgt_pcd.voxel_down_sample(0.025)
        src_pcd = np.array(src_pcd.points).astype(np.float32)
        tgt_pcd = np.array(tgt_pcd.points).astype(np.float32)
        print("src_pcd:",np.shape(src_pcd))
        print("tgt_pcd:",np.shape(tgt_pcd))



        src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
        tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)

        # fake the ground truth information
        rot = np.eye(3).astype(np.float32)
        trans = np.ones((3,1)).astype(np.float32)
        correspondences = torch.ones(1,2).long()

        return src_pcd,tgt_pcd,src_feats,tgt_feats,rot,trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)
    
def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans


def transform(pts, trans):
    if len(pts.shape) == 3:
        trans_pts = torch.einsum('bnm,bmk->bnk', trans[:, :3, :3],
                                 pts.permute(0, 2, 1)) + trans[:, :3, 3:4]
        return trans_pts.permute(0, 2, 1)
    else:
        trans_pts = torch.einsum('nm,mk->nk', trans[:3, :3],
                                 pts.T) + trans[:3, 3:4]
        return trans_pts.T


def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    """
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

    # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (
            torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (
            torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)  # 升维度，然后变为对角阵
    H = Am.permute(0, 2, 1) @ Weight @ Bm  # permute : tensor中的每一块做转置

    # find rotation
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    return integrate_trans(R, t)


def post_refinement(initial_trans, src_kpts, tgt_kpts, iters, weights=None):
    inlier_threshold = 0.1
    pre_inlier_count = 0
    for i in range(iters):
        pred_tgt = transform(src_kpts, initial_trans)
        L2_dis = torch.norm(pred_tgt - tgt_kpts, dim=-1)
        pred_inlier = (L2_dis < inlier_threshold)[0]
        inlier_count = torch.sum(pred_inlier)
        if inlier_count <= pre_inlier_count:
            break
        pre_inlier_count = inlier_count
        initial_trans = rigid_transform_3d(
            A=src_kpts[:, pred_inlier, :],
            B=tgt_kpts[:, pred_inlier, :],
            weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier]
        )
    return initial_trans


def estimate_normal(pcd, radius=0.06, max_nn=30):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))


def transformation_error(pred_trans, gt_trans):
    pred_R = pred_trans[:3, :3]
    gt_R = gt_trans[:3, :3]
    pred_t = pred_trans[:3, 3:4]
    gt_t = gt_trans[:3, 3:4]
    tr = torch.trace(pred_R.T @ gt_R)
    RE = torch.acos(torch.clamp((tr - 1) / 2.0, min=-1, max=1)) * 180 / np.pi
    TE = torch.norm(pred_t - gt_t) * 100
    return RE, TE


def visualization(src_pcd, tgt_pcd, pred_trans):
    if not src_pcd.has_normals():
        estimate_normal(src_pcd)
        estimate_normal(tgt_pcd)
    src_pcd.paint_uniform_color([1, 0.706, 0])
    tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([src_pcd, tgt_pcd])

    src_pcd.transform(pred_trans)
    o3d.visualization.draw_geometries([src_pcd, tgt_pcd])


def transformation_Matrix_MAC(corr_data,GTmat):
    #corr_path = folder + '/corr_data.txt'
    #GTmat_path = folder + '/GTmat.txt'
    #src_pcd_path = folder + '/source.ply'
    #tgt_pcd_path = folder + '/target.ply'
    
    #corr_data = np.loadtxt(corr_path, dtype=np.float32)
    #GTmat = np.loadtxt(GTmat_path, dtype=np.float32)
    #src_pcd = o3d.io.read_point_cloud(src_pcd_path)
    #tgt_pcd = o3d.io.read_point_cloud(tgt_pcd_path)
    
    #src_pts = torch.from_numpy(corr_data[:, 0:3]).cuda()
    #tgt_pts = torch.from_numpy(corr_data[:, 3:6]).cuda()
    #GTmat = torch.from_numpy(GTmat).cuda()
    
    src_pts = torch.from_numpy(corr_data[:, 0:3])
    tgt_pts = torch.from_numpy(corr_data[:, 3:6])
    #GTmat = torch.from_numpy(GTmat)
    if torch.cuda.is_available():
        src_pts = src_pts.cuda()
        tgt_pts = tgt_pts.cuda()
        #GTmat = GTmat.cuda()
    
    t1 = time.perf_counter()
    src_dist = ((src_pts[:, None, :] - src_pts[None, :, :]) ** 2).sum(-1) ** 0.5
    tgt_dist = ((tgt_pts[:, None, :] - tgt_pts[None, :, :]) ** 2).sum(-1) ** 0.5
    cross_dis = torch.abs(src_dist - tgt_dist)
    FCG = torch.clamp(1 - cross_dis ** 2 / 0.1 ** 2, min=0)
    FCG = FCG - torch.diag_embed(torch.diag(FCG))
    FCG[FCG < 0.99] = 0
    SCG = torch.matmul(FCG, FCG) * FCG
    t2 = time.perf_counter()
    print(f'Graph construction: %.2fms' % ((t2 - t1) * 1000))

    SCG = SCG.cpu().numpy()
    t1 = time.perf_counter()
    graph = igraph.Graph.Adjacency((SCG > 0).tolist())
    graph.es['weight'] = SCG[SCG.nonzero()]
    graph.vs['label'] = range(0, corr_data.shape[0])
    graph.to_undirected()
    macs = graph.maximal_cliques(min=3)
    t2 = time.perf_counter()
    print(f'Search maximal cliques: %.2fms' % ((t2 - t1) * 1000))
    print(f'Total: %d' % len(macs))
    clique_weight = np.zeros(len(macs), dtype=float)
    for ind in range(len(macs)):
        mac = list(macs[ind])
        if len(mac) >= 3:
            for i in range(len(mac)):
                for j in range(i + 1, len(mac)):
                    clique_weight[ind] = clique_weight[ind] + SCG[mac[i], mac[j]]

    clique_ind_of_node = np.ones(corr_data.shape[0], dtype=int) * -1
    max_clique_weight = np.zeros(corr_data.shape[0], dtype=float)
    max_size = 3
    for ind in range(len(macs)):
        mac = list(macs[ind])
        weight = clique_weight[ind]
        if weight > 0:
            for i in range(len(mac)):
                if weight > max_clique_weight[mac[i]]:
                    max_clique_weight[mac[i]] = weight
                    clique_ind_of_node[mac[i]] = ind
                    max_size = len(mac) > max_size and len(mac) or max_size

    filtered_clique_ind = list(set(clique_ind_of_node))
    filtered_clique_ind.remove(-1)
    print(f'After filtered: %d' % len(filtered_clique_ind))
    
    group = []
    for s in range(3, max_size + 1):
        group.append([])
    for ind in filtered_clique_ind:
        mac = list(macs[ind])
        group[len(mac) - 3].append(ind)

    tensor_list_A = []
    tensor_list_B = []
    for i in range(len(group)):
        batch_A = src_pts[list(macs[group[i][0]])][None]
        batch_B = tgt_pts[list(macs[group[i][0]])][None]
        if len(group) == 1:
            continue
        for j in range(1, len(group[i])):
            mac = list(macs[group[i][j]])
            src_corr = src_pts[mac][None]
            tgt_corr = tgt_pts[mac][None]
            batch_A = torch.cat((batch_A, src_corr), 0)
            batch_B = torch.cat((batch_B, tgt_corr), 0)
        tensor_list_A.append(batch_A)
        tensor_list_B.append(batch_B)

    inlier_threshold = 0.1
    max_score = 0
    final_trans = torch.eye(4)
    for i in range(len(tensor_list_A)):
        trans = rigid_transform_3d(tensor_list_A[i], tensor_list_B[i], None, 0)
        pred_tgt = transform(src_pts[None], trans)  # [bs,  num_corr, 3]
        L2_dis = torch.norm(pred_tgt - tgt_pts[None], dim=-1)  # [bs, num_corr]
        MAE_score = torch.div(torch.sub(inlier_threshold, L2_dis), inlier_threshold)
        MAE_score = torch.sum(MAE_score * (L2_dis < inlier_threshold), dim=-1)
        max_batch_score_ind = MAE_score.argmax(dim=-1)
        max_batch_score = MAE_score[max_batch_score_ind]
        if max_batch_score > max_score:
            max_score = max_batch_score
            final_trans = trans[max_batch_score_ind]
    
    # RE TE
    """
    re, te = transformation_error(final_trans, GTmat)
    final_trans1 = post_refinement(initial_trans=final_trans[None], src_kpts=src_pts[None], tgt_kpts=tgt_pts[None], iters=20)
    re1, te1 = transformation_error(final_trans1[0], GTmat)
    if re1 <= re and te1 <= te:
        final_trans = final_trans1[0]
        re, te = re1, te1
        print('est_trans updated.')

    print(f'RE: %.2f, TE: %.2f' % (re, te))
    """
    final_trans = final_trans.cpu().numpy()
    
    return final_trans 
       

def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (1,1,1)'''
    color = np.array(color)
    white = np.array([1, 1, 1])
    vector = white-color
    return color + vector * percent

def calculate_relative_transformation(src_path, tgt_path):
    try:
        # Load the matrices, skipping the first row
        src_matrix = np.loadtxt(src_path, skiprows=1)
        tgt_matrix = np.loadtxt(tgt_path, skiprows=1)

        # Check if matrices are 4x4
        if src_matrix.shape != (4, 4) or tgt_matrix.shape != (4, 4):
            print("Error: Matrices are not of shape 4x4")
            return None

        # Calculate the relative transformation
        relative_transform = np.dot(np.linalg.inv(tgt_matrix), src_matrix)
        return relative_transform

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def draw_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, src_saliency, tgt_saliency, tsfm, tsfm_gt):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. overlap colors
    rot, trans = to_tensor(tsfm[:3,:3]), to_tensor(tsfm[:3,3][:,None])
    src_overlap = src_overlap[:,None].repeat(1,3).numpy()
    tgt_overlap = tgt_overlap[:,None].repeat(1,3).numpy()
    src_overlap_color = lighter(get_yellow(), 1 - src_overlap)
    tgt_overlap_color = lighter(get_blue(), 1 - tgt_overlap)
    src_pcd_overlap = copy.deepcopy(src_pcd_before)
    src_pcd_overlap.transform(tsfm)
    tgt_pcd_overlap = copy.deepcopy(tgt_pcd_before)
    src_pcd_overlap.colors = o3d.utility.Vector3dVector(src_overlap_color)
    tgt_pcd_overlap.colors = o3d.utility.Vector3dVector(tgt_overlap_color)

    ########################################
    # 3. draw registrations
    src_pcd_after = copy.deepcopy(src_pcd_before)
    src_pcd_after.transform(tsfm)
    
    src_pcd_after_gt = copy.deepcopy(src_pcd_before)
    src_pcd_after_gt.transform(tsfm_gt)
    tgt_pcd_before_gt = copy.deepcopy(tgt_pcd_before)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Input', width=960, height=540, left=0, top=0)
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='Inferred overlap region', width=960, height=540, left=0, top=600)
    vis2.add_geometry(src_pcd_overlap)
    vis2.add_geometry(tgt_pcd_overlap)

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name ='Our registration', width=960, height=540, left=960, top=0)
    vis3.add_geometry(src_pcd_after)
    vis3.add_geometry(tgt_pcd_before)
    
    vis4 = o3d.visualization.Visualizer()
    vis4.create_window(window_name ='ground truth', width=960, height=540, left=960, top=600)
    vis4.add_geometry(src_pcd_after_gt)
    vis4.add_geometry(tgt_pcd_before_gt)
    
    while True:
        vis1.update_geometry(src_pcd_before)
        vis1.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_pcd_overlap)
        vis2.update_geometry(tgt_pcd_overlap)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        vis3.update_geometry(src_pcd_after)
        vis3.update_geometry(tgt_pcd_before)
        if not vis3.poll_events():
            break
        vis3.update_renderer()
        
        vis4.update_geometry(src_pcd_after_gt)
        vis4.update_geometry(tgt_pcd_before)
        if not vis4.poll_events():
            break
        vis4.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()    


def main(config, demo_loader):
    config.model.eval()
    c_loader_iter = demo_loader.__iter__()
    with torch.no_grad():
        inputs = next(c_loader_iter)
        ##################################
        # load inputs to device.
        for k, v in inputs.items():  
            if type(v) == list:
                inputs[k] = [item.to(config.device) for item in v]
            else:
                inputs[k] = v.to(config.device)
        print("뭐가 문젠겨")
        ###############################################
        # forward pass
        feats, scores_overlap, scores_saliency = config.model(inputs)  #[N1, C1], [N2, C2]
        pcd = inputs['points'][0]
        len_src = inputs['stack_lengths'][0][0]
        c_rot, c_trans = inputs['rot'], inputs['trans']
        correspondence = inputs['correspondences']
        src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
        src_raw = copy.deepcopy(src_pcd)
        tgt_raw = copy.deepcopy(tgt_pcd)
        src_feats, tgt_feats = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
        src_overlap, src_saliency = scores_overlap[:len_src].detach().cpu(), scores_saliency[:len_src].detach().cpu()
        tgt_overlap, tgt_saliency = scores_overlap[len_src:].detach().cpu(), scores_saliency[len_src:].detach().cpu()

        ########################################
        # do probabilistic sampling guided by the score
        src_scores = src_overlap * src_saliency
        tgt_scores = tgt_overlap * tgt_saliency
        
        
        if(src_pcd.size(0) > config.n_points):
            idx = np.arange(src_pcd.size(0))
            probs = (src_scores / src_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size= config.n_points, replace=False, p=probs)
            src_pcd, src_feats = src_pcd[idx], src_feats[idx]
        if(tgt_pcd.size(0) > config.n_points):
            idx = np.arange(tgt_pcd.size(0))
            probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size= config.n_points, replace=False, p=probs)
            tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]
        
      
        ########################################
        # extract corr txt to experiment mac
           
        # Assuming src_feats and tgt_feats correspond to the features of the selected points in src_pcd and tgt_pcd
        if isinstance(src_feats, torch.Tensor):
            src_feats = src_feats.detach().cpu().numpy()
        if isinstance(tgt_feats, torch.Tensor):
            tgt_feats = tgt_feats.detach().cpu().numpy()

        # Use NearestNeighbors to find the closest point in tgt_pcd for each point in src_pcd based on the features.
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(tgt_feats)
        _, indices = nbrs.kneighbors(src_feats)

        # Now indices contains the index in tgt_pcd that corresponds to each point in src_pcd
        # Let's extract these correspondences
        matched_src_pcd = src_pcd
        matched_tgt_pcd = tgt_pcd[indices.reshape(-1)]

        # Concatenate the source and target points to form correspondences
        corr_data = np.hstack((matched_src_pcd, matched_tgt_pcd))
        tsfm_gt = calculate_relative_transformation(config.src_pcd_trs,config.tgt_pcd_trs)
        tsfm_gt = torch.from_numpy(tsfm_gt).float().clone()
        tsfm = transformation_Matrix_MAC(corr_data,tsfm_gt)
        draw_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, src_saliency, tgt_saliency, tsfm,tsfm_gt)


if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help= 'Path to the config file.')
    args = parser.parse_args()
    
    config = load_config(args.config)
    config = edict(config)
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')
    
    # model initialization
    config.architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers-1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
        config.architecture.append('resnetb')
    for i in range(config.num_layers-2):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
    config.architecture.append('nearest_upsample')
    config.architecture.append('last_unary')
    config.model = KPFCNN(config).to(config.device)
    
    # create dataset and dataloader
    #info_train = load_obj(config.train_info)
    #train_set = KITTIDataset(config,data_augmentation=True)
    train_set, val_set, benchmark_set = get_datasets(config)
    demo_set = KittiDemo(config, config.src_pcd, config.tgt_pcd)

    #_, neighborhood_limits = get_dataloader(dataset=train_set,
    #                                    batch_size=config.batch_size,
    #                                    shuffle=True,
    #                                    num_workers=config.num_workers,
    #                                    )
    demo_loader, _ = get_dataloader(dataset=demo_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=1,
                                        neighborhood_limits=100)

    # load pretrained weights
    assert config.pretrain != None
    #state = torch.load(config.pretrain)
    state = torch.load(config.pretrain, map_location=torch.device('cpu'))

    config.model.load_state_dict(state['state_dict'])

    # do pose estimation
    main(config, demo_loader)
