# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import cv2
import numpy as np
from matplotlib import cm
import argparse
from nuscenes.nuscenes import NuScenes
import os
import torch.nn.functional as F
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default="./data")
parser.add_argument('--version', default='v1.0-trainval')
parser.add_argument('--scene-prefix', nargs='+', default=['scene'])
parser.add_argument('--model', default='metric3d_vit_large')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--target-dir', default='metric_3d_nusc')

args = parser.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Metadata
cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
input_size = (616, 1064) # for vit model
# input_size = (512, 992) # for vit model with nuscenes
h, w = (900, 1600)
min_depth, max_depth = 0, 50
scale = min(input_size[0] / h, input_size[1] / w)
h_resize, w_resize = int(h*scale), int(w*scale)
padding = [123.675, 116.28, 103.53]
pad_h = input_size[0] - h_resize
pad_w = input_size[1] - w_resize
pad_h_half = pad_h // 2
pad_w_half = pad_w // 2
pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
target_path = os.path.join('data', args.target_dir)

depth_cm = cm.get_cmap('plasma')

mean = torch.tensor([123.675, 116.28, 103.53]).float()[None, :, None, None]
std = torch.tensor([58.395, 57.12, 57.375]).float()[None, :, None, None]

# Prepare Dataset
nusc = NuScenes(version=args.version, dataroot=os.path.join(args.root, 'nuscenes'), verbose=True)
scenes = [s for s in nusc.scene if any([s['name'].startswith(prefix) for prefix in args.scene_prefix])]

# Prepare Model
model = torch.hub.load('yvanyin/metric3d', args.model, pretrain=True)
model.cuda().eval()

for scene in tqdm(scenes):
    sample_token = scene['first_sample_token']
    scene_path = os.path.join(target_path, scene['name'])
    os.makedirs(scene_path, exist_ok=True)

    while sample_token != '':
        sample = nusc.get('sample', sample_token)

        # Skip if file exists
        if not args.overwrite and os.path.exists(os.path.join(scene_path, sample_token + '.npy')):
            sample_token = sample['next']
            continue

        intrinsics = np.array([np.array(nusc.get('calibrated_sensor', nusc.get('sample_data', 
                        sample['data'][cam])['calibrated_sensor_token'])['camera_intrinsic'])[[0,1,0,1],[0,1,2,2]] * scale for cam in cams])
        image_paths = [os.path.join(nusc.dataroot, nusc.get('sample_data', sample['data'][cam])['filename']) for cam in cams]
        images = [cv2.imread(image_path) for image_path in image_paths]
        images = [cv2.resize(image, (w_resize, h_resize), interpolation=cv2.INTER_LINEAR) for image in images]
        images = [cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding) for image in images]
        images = torch.stack([torch.from_numpy(image.transpose((2, 0, 1))).float() for image in images])
        images = ((images - mean) / std).cuda()
        with torch.no_grad():
            pred_depth, confidence, output_dict = model.inference({'input': images})

        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[:, pad_info[0] : pred_depth.shape[1] - pad_info[1], pad_info[2] : pred_depth.shape[2] - pad_info[3]]

        # upsample to original size
        pred_depth = F.interpolate(pred_depth[:, None, :, :], (h, w), mode='bilinear').squeeze()

        canonical_to_real_scale = torch.tensor(intrinsics[:, 0] / 1000.0, device=device) # 1000.0 is the focal length of canonical camera
        pred_depth = pred_depth * canonical_to_real_scale[:, None, None] # now the depth is metric
        pred_depth = torch.clamp(pred_depth, 0, 300).float()

        # Store the depth image
        np.save(os.path.join(scene_path, sample_token + '.npy'), pred_depth.cpu().numpy().astype(np.float16))

        # get next sample
        sample_token = sample['next']
