# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import os
import mmcv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from mmdet3d.datasets.ray_miou_metric.ray_metrics import main_rayiou
from mmdet3d.datasets.ray_miou_metric.ego_pose_dataset import EgoPoseDataset

occ3d_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

openocc_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

def main(args):
    if args.version == 'v1.0-trainval':
        info_file = 'bevdetv2-nuscenes_infos_val.pkl'
    elif args.version == 'v1.0-mini':
        info_file = 'bevdetv2-nuscenes-mini_infos_val.pkl'
    data_infos = mmcv.load(os.path.join(args.data_root, info_file))['infos']

    lidar_origins = []
    occ_gts = []
    occ_preds = []

    for idx, batch in enumerate(DataLoader(EgoPoseDataset(data_infos), num_workers=8)):
        output_origin = batch[1]
        info = data_infos[idx]
        occ_path = os.path.join(args.gt_root, info['scene_name'], info['token'], 'labels.npz')
        occ_gt = np.load(occ_path, allow_pickle=True)['semantics']
        occ_gt = np.reshape(occ_gt, [200, 200, 16]).astype(np.uint8)

        occ_path = os.path.join(args.pred_dir, info['token'] + '.npz')
        occ_pred = np.load(occ_path, allow_pickle=True)['arr_0']
        occ_pred = np.reshape(occ_pred, [200, 200, 16]).astype(np.uint8)
        
        lidar_origins.append(output_origin)
        occ_gts.append(occ_gt)
        occ_preds.append(occ_pred)
    
    if args.data_type == 'occ3d':
        occ_class_names = occ3d_class_names
    elif args.data_type == 'openocc_v2':
        occ_class_names = openocc_class_names
    else:
        raise ValueError
    
    print(main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default='data/')
    parser.add_argument("--gt-root", type=str, default='data/gts')
    parser.add_argument("--version", type=str, default='v1.0-trainval')
    parser.add_argument("--pred-dir", type=str)
    parser.add_argument("--data-type", type=str, choices=['occ3d', 'openocc_v2'], default='occ3d')
    args = parser.parse_args()

    torch.random.manual_seed(0)
    np.random.seed(0)

    main(args)