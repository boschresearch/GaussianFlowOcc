# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import os

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from pyquaternion import Quaternion
from matplotlib import cm

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES

color_map = np.array(
        [[0, 0, 0, 255],
        [255, 120, 50, 255],  # 1 barrier orangey
        [255, 192, 203, 255],  # 2 bicycle pink
        [255, 255, 0, 255],  # 3 bus yellow
        [0, 150, 245, 255],  # 4 car blue
        [0, 255, 255, 255],  # 5 construction_vehicle cyan
        [200, 180, 0, 255],  # 6 motorcycle dark orange
        [255, 0, 0, 255],  # 7 pedestrian red
        [255, 240, 150, 255],  # 8 traffic_cone light yellow
        [135, 60, 0, 255],  # 9 trailer brown
        [160, 32, 240, 255],  # 10 truck purple
        [255, 0, 255, 255],  # 11 driveable_surface dark pink
        [139, 137, 137, 255],  # 12 other_flat dark red
        [75, 0, 75, 255],  # 13sidewalk dark purple
        [150, 240, 80, 255],  # 14 terrain light green
        [230, 230, 250, 255],  # 15 manmade white
        [0, 175, 0, 255],  # 16 vegetation green
        [0, 255, 127, 255],  # ego car dark cyan
    ], dtype=np.uint8)
depth_color_map = cm.get_cmap('magma')

def calculate_transformations(results, T, zero_index, scale_factor=None, crop_top=None):

    # T = len(self.render_frame_ids)
    N = len(results['cam_names'])

    # ego2global (TODO: Is this really ego_lidar? Or is it ego of front cam?)
    lidarego2global = np.eye(4, dtype=np.float32)[None, ...].repeat(T, 0)
    lidarego2global[:, :3, :3] = np.array([Quaternion(results['render_frames'][i]['ego2global_rotation']).rotation_matrix for i in range(T)])
    lidarego2global[:, :3, 3] = np.array([results['render_frames'][i]['ego2global_translation'] for i in range(T)])
    lidarego2global = torch.from_numpy(lidarego2global)

    # cam2camego
    cam2camego = np.eye(4, dtype=np.float32)[None, None, ...].repeat(T, 0).repeat(N, 1)
    cam2camego[..., :3, :3] = np.array([[Quaternion(results['render_frames'][i]['cams'][cam_name]['sensor2ego_rotation']).rotation_matrix for cam_name in results['cam_names']] for i in range(T)])
    cam2camego[..., :3, 3] = np.array([[results['render_frames'][i]['cams'][cam_name]['sensor2ego_translation'] for cam_name in results['cam_names']] for i in range(T)])
    cam2camego = torch.from_numpy(cam2camego)

    # camego2global
    camego2global = np.eye(4, dtype=np.float32)[None, None, ...].repeat(T, 0).repeat(N, 1)
    camego2global[..., :3, :3] = np.array([[Quaternion(results['render_frames'][i]['cams'][cam_name]['ego2global_rotation']).rotation_matrix for cam_name in results['cam_names']] for i in range(T)])
    camego2global[..., :3, 3] = np.array([[results['render_frames'][i]['cams'][cam_name]['ego2global_translation'] for cam_name in results['cam_names']] for i in range(T)])
    camego2global = torch.from_numpy(camego2global)

    # intrins
    intrins = torch.from_numpy(np.array([[results['render_frames'][i]['cams'][cam_name]['cam_intrinsic'] for cam_name in results['cam_names']] for i in range(T)]))
    if scale_factor is not None:
        intrins[..., 0, 0] *= scale_factor
        intrins[..., 1, 1] *= scale_factor
        intrins[..., 0, 2] *= scale_factor
        intrins[..., 1, 2] *= scale_factor
    if crop_top is not None:
        intrins[..., 1, 2] -= crop_top

    # sensor2ego -> ego2global -> global2egocam -> egocam2cam -> cam2img
    # lidar2global = torch.matmul(lidarego2global, lidar2lidarego)
    cam2global = torch.matmul(camego2global, cam2camego)
    # lidar2cam = torch.matmul(torch.inverse(cam2global), lidar2global[:, None, ...]) 
    # cam2img = torch.eye(4, dtype=torch.float32)[None, None, ...].repeat(T, N, 1, 1)
    # cam2img[..., :3, :3] = intrins
    # lidar2img = torch.matmul(cam2img, lidar2cam)

    # Also store/return transformations relevant for the renderer
    # I need: Cam2ego_l_t = cam2global * global2ego_l_t
    ego_l_t2global = lidarego2global[zero_index]
    cam2ego_lidar_t = torch.matmul(torch.inverse(ego_l_t2global[None, None, ...]), cam2global)
    # lidar2ego_l_t = torch.matmul(torch.inverse(ego_l_t2global[None, ...]), lidar2global)

    return cam2ego_lidar_t, intrins
    # return lidar2img, cam2ego_lidar_t, lidar2ego_l_t

def sample_render_frame_rgbs(results, coors, cam_idxs, rounded=False):
    rgbs = coors.new_zeros((coors.shape[0], 3)) 
    for i, frame in enumerate(results['render_frames']):
        img_paths = [frame['cams'][cam]['data_path'] for cam in results['cam_names']]
        imgs = torch.stack([torch.tensor(np.array(Image.open(path), dtype=np.float32)) / 255. for path in img_paths])
        temp_mask = cam_idxs[:, 0] == i

        if rounded:
            # pixel rgb
            coors_rounded = coors[temp_mask].round().long()
            sample_idxs = torch.cat((cam_idxs[temp_mask, 1][:, None], coors_rounded), dim=-1)
            rgbs[temp_mask] = imgs[tuple(sample_idxs.T)]

        else:
            for c, cam in enumerate(results['cam_names']):
                selected_cam_idxs = (cam_idxs[..., 1] == c) * temp_mask
                selected_coors = coors[selected_cam_idxs]
                selected_coors = (selected_coors / torch.tensor([[900, 1600]])) * 2 - 1
                rgbs[selected_cam_idxs] = F.grid_sample(
                    imgs[c][None, ...].permute(0, 3, 1, 2),
                    selected_coors[None, None, :, [1, 0]],
                    align_corners=False
                ).squeeze().permute(1,0)

    return rgbs

@PIPELINES.register_module()
class GaussianFlowOcc_GenerateLabels_LiDAR():
    def __init__(self, gt_root="data/rays", downscale_factor=.44, crop_top=140, mask_dynamic=True, temporal_frame_ids=[0],
                 original_resolution=(900, 1600), delete_duplicates=True, load_rgb=False, max_pixels=600000):
        self.gt_root = gt_root
        self.temporal_frame_ids = temporal_frame_ids
        self.zero_index = [i for i, t in enumerate(temporal_frame_ids) if t==0][0]
        self.downscale_factor = downscale_factor
        self.crop_top = crop_top
        self.mask_dynamic = mask_dynamic
        self.dynamic_classes = torch.tensor([2, 3, 4, 5, 6, 7, 9, 10])
        self.original_h, self.original_w = original_resolution
        self.delete_duplicates = delete_duplicates
        self.load_rgb = load_rgb
        self.neighbors = torch.cartesian_prod(torch.tensor([-1,0,1]), torch.tensor([-1,0,1])).unsqueeze(0)
        self.max_pixels = max_pixels

    def __call__(self, results):
        cam2ego_l_t, intrins = calculate_transformations(results, len(self.temporal_frame_ids), self.zero_index, scale_factor=self.downscale_factor, crop_top=self.crop_top)
        ego_l_t2cam = torch.inverse(cam2ego_l_t)
        height, width = int(self.original_h * self.downscale_factor - self.crop_top), int(self.original_w * self.downscale_factor)
        all_pixels = []
        for i, frame in enumerate(results['render_frames']):
            rays = np.load(os.path.join(self.gt_root, results['scene_name'], f'{frame["token"]}.npz'))
            coors, depths, labels, cam_idxs = torch.tensor(rays['pixels']), torch.tensor(rays['depths']), torch.tensor(rays['labels']), torch.tensor(rays['cam_indices'])

            # Downscale the coors
            coors = (coors * self.downscale_factor - torch.tensor([self.crop_top, 0.]))#.floor().long()
            coor_mask = (coors>0).all(-1)
            coors, depths, labels, cam_idxs = coors[coor_mask], depths[coor_mask], labels[coor_mask], cam_idxs[coor_mask]
            
            # Mask dynamic pixels at temporal cams
            if self.mask_dynamic and i != self.zero_index:
                dynamic_mask = ((labels.unsqueeze(1) - self.dynamic_classes.unsqueeze(0)) == 0).any(dim=-1)
                coors, depths, labels, cam_idxs = coors[~dynamic_mask], depths[~dynamic_mask], labels[~dynamic_mask], cam_idxs[~dynamic_mask]

            # load images
            if self.load_rgb:
                frame_imgs = [np.array(Image.open(frame['cams'][cam]['data_path'])) for cam in results['cam_names']]
                # resize & crop
                frame_imgs = [cv2.resize(img, None, None, self.downscale_factor, self.downscale_factor)[self.crop_top:] for img in frame_imgs]
                # normalize rgb
                frame_imgs = torch.stack([mmlabNormalize(img) for img in frame_imgs])
                all_sampled = []
                # sample rgb values for each lidar ray
                for cam_nr in range(len(results['cam_names'])):
                    cam_mask = cam_idxs == cam_nr
                    grid = (coors[cam_mask].unsqueeze(1) + self.neighbors)[..., [1, 0]]
                    grid = ((grid / torch.tensor([width, height])) * 2 - 1).unsqueeze(0)
                    sampled = F.grid_sample(frame_imgs[cam_nr][None, ...], grid, align_corners=False, padding_mode='reflection').squeeze(0).permute(1,2,0).flatten(1,2)
                    all_sampled.append(sampled)
                frame_imgs = torch.cat(all_sampled, dim=0)
            else:
                frame_imgs = torch.empty(0)

            temporal_index = torch.full((len(coors), 1), i)
            pixels_t = torch.cat((temporal_index, cam_idxs[:, None], coors, depths[:, None], labels[:, None], frame_imgs), dim=-1)
            all_pixels.append(pixels_t)

        all_pixels = torch.cat(all_pixels)
        
        # sample random amount of pixels if too many
        if len(all_pixels) > self.max_pixels:
            all_pixels = all_pixels[torch.randperm(len(all_pixels))[:self.max_pixels]]

        # Put everything in a single dict
        results['gs_gts_pixel'] = all_pixels

        # # Test visualization
        # test_map1 = color_map[all_label_maps[3,1]][..., :-1]
        # test_map2 = color_map[all_label_maps[1,1]][..., :-1]
        # test_img = cv2.resize(cv2.imread(results['curr']['cams']['CAM_FRONT']['data_path']), None, None, fx=0.44, fy=0.44)[self.crop_top:]
        # depth_test1 = (depth_color_map(all_depth_maps[1,2] / 45.)[..., :3] * 255).astype(np.uint8)

        # Provide the intrinsics and extrinsics for rendering the gaussians
        results['gs_intrins'] = intrins.float() # [T, C, 3, 3]
        results['gs_extrins'] = ego_l_t2cam # [T, C, 4, 4]

        return results
    
@PIPELINES.register_module()
class GaussianFlowOcc_GenerateLabels_LiDARHorizon():
    def __init__(self, gt_root="data/rays", downscale_factor=.44, crop_top=140, mask_dynamic=True, temporal_frame_ids=[0],
                 original_resolution=(900, 1600), delete_duplicates=True, load_rgb=False, normalize_rgb=False, max_pixels=600000, num_frames=3, select_frames=None):
        self.gt_root = gt_root
        self.temporal_frame_ids = temporal_frame_ids
        self.zero_index = [i for i, t in enumerate(temporal_frame_ids) if t==0][0]
        self.downscale_factor = downscale_factor
        self.crop_top = crop_top
        self.mask_dynamic = mask_dynamic
        self.dynamic_classes = torch.tensor([2, 3, 4, 5, 6, 7, 9, 10])
        self.original_h, self.original_w = original_resolution
        self.delete_duplicates = delete_duplicates
        self.load_rgb = load_rgb
        self.neighbors = torch.cartesian_prod(torch.tensor([-1,0,1]), torch.tensor([-1,0,1])).unsqueeze(0)
        self.max_pixels = max_pixels
        self.num_frames = num_frames
        self.normalize_rgb = normalize_rgb
        self.select_frames = select_frames

    def __call__(self, results):
        cam2ego_l_t, intrins = calculate_transformations(results, len(self.temporal_frame_ids), self.zero_index, scale_factor=self.downscale_factor, crop_top=self.crop_top)
        ego_l_t2cam = torch.inverse(cam2ego_l_t)
        height, width = int(self.original_h * self.downscale_factor - self.crop_top), int(self.original_w * self.downscale_factor)
        all_pixels = []

        # Alwas take current time step
        valid_frame_indices = [i for i in range(len(self.temporal_frame_ids)) if i!=self.zero_index and results['render_frames'][i]['valid']]
        if self.select_frames is None:
            selected_frames = np.random.choice(valid_frame_indices, min(self.num_frames, len(valid_frame_indices)), replace=False)
            # Pad with random other frames if there are not enough valid frames available
            if len(selected_frames) < self.num_frames:
                diff = self.num_frames - len(selected_frames)
                selected_frames = np.concatenate((selected_frames, np.random.choice(valid_frame_indices, diff, replace=False)))
            selected_frames = np.concatenate((selected_frames, [self.zero_index])).astype(np.int)
        else:
            selected_frames = np.array([self.temporal_frame_ids.index(i) for i in self.select_frames])

        render_frames = [results['render_frames'][i] for i in selected_frames]
        intrins, ego_l_t2cam = intrins[selected_frames], ego_l_t2cam[selected_frames]
        for i, frame in enumerate(render_frames):
            rays = np.load(os.path.join(self.gt_root, results['scene_name'], f'{frame["token"]}.npz'))
            coors, depths, labels, cam_idxs = torch.tensor(rays['pixels']), torch.tensor(rays['depths']), torch.tensor(rays['labels']), torch.tensor(rays['cam_indices'])

            # Downscale the coors
            coors = (coors * self.downscale_factor - torch.tensor([self.crop_top, 0.]))#.floor().long()
            coor_mask = (coors>0).all(-1)
            coors, depths, labels, cam_idxs = coors[coor_mask], depths[coor_mask], labels[coor_mask], cam_idxs[coor_mask]
            
            # Mask dynamic pixels at temporal cams
            if self.mask_dynamic and selected_frames[i] != self.zero_index:
                dynamic_mask = ((labels.unsqueeze(1) - self.dynamic_classes.unsqueeze(0)) == 0).any(dim=-1)
                coors, depths, labels, cam_idxs = coors[~dynamic_mask], depths[~dynamic_mask], labels[~dynamic_mask], cam_idxs[~dynamic_mask]

            # load images
            if self.load_rgb:
                frame_imgs = [np.array(Image.open(frame['cams'][cam]['data_path'])) for cam in results['cam_names']]
                # resize & crop
                frame_imgs = [cv2.resize(img, None, None, self.downscale_factor, self.downscale_factor)[self.crop_top:] for img in frame_imgs]
                # normalize rgb
                if self.normalize_rgb:
                    frame_imgs = torch.stack([mmlabNormalize(img) for img in frame_imgs])
                else:
                    frame_imgs = torch.stack([torch.tensor(img) / 255. for img in frame_imgs]).permute(0, 3, 1, 2)
                all_sampled = []
                # sample rgb values for each lidar ray
                for cam_nr in range(len(results['cam_names'])):
                    cam_mask = cam_idxs == cam_nr
                    grid = (coors[cam_mask].unsqueeze(1) + self.neighbors)[..., [1, 0]]
                    grid = ((grid / torch.tensor([width, height])) * 2 - 1).unsqueeze(0)
                    sampled = F.grid_sample(frame_imgs[cam_nr][None, ...], grid, align_corners=False, padding_mode='reflection').squeeze(0).permute(1,2,0).flatten(1,2)
                    all_sampled.append(sampled)
                frame_imgs = torch.cat(all_sampled, dim=0)
            else:
                frame_imgs = torch.empty(0)

            # Filter out depth > 40m
            depth_mask = depths <= 40
            coors, depths, labels, cam_idxs = coors[depth_mask], depths[depth_mask], labels[depth_mask], cam_idxs[depth_mask]

            temporal_index = torch.full((len(coors), 1), i)
            pixels_t = torch.cat((temporal_index, cam_idxs[:, None], coors, depths[:, None], labels[:, None], frame_imgs), dim=-1)
            all_pixels.append(pixels_t)

        all_pixels = torch.cat(all_pixels)
        
        # sample random amount of pixels if too many
        if len(all_pixels) > self.max_pixels:
            all_pixels = all_pixels[torch.randperm(len(all_pixels))[:self.max_pixels]]

        # Put everything in a single dict
        results['gs_gts_pixel'] = all_pixels

        # Provide the intrinsics and extrinsics for rendering the gaussians
        results['gs_intrins'] = intrins.float() # [T, C, 3, 3]
        results['gs_extrins'] = ego_l_t2cam # [T, C, 4, 4]
        results['selected_frames'] = selected_frames

        return results
    
@PIPELINES.register_module()
class GaussianFlowOcc_GeneratePseudoLabels():
    def __init__(self, grounded_sam_root='data/grounded_sam_nusc', depth_root='data/metric_3d_nusc', temporal_frame_ids=[0], downscale_factor=.44, crop_top=140, mask_dynamic=True,
                 original_resolution=(900, 1600), delete_duplicates=True, load_rgb=False, max_depth=40):
        self.grounded_sam_root = grounded_sam_root
        self.depth_root = depth_root
        self.load_rgb = load_rgb
        self.temporal_frame_ids = temporal_frame_ids
        self.zero_index = [i for i, t in enumerate(temporal_frame_ids) if t==0][0]
        self.downscale_factor = downscale_factor # How much to downscale render resolution
        self.save_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
        self.crop_top = crop_top
        self.mask_dynamic = mask_dynamic
        self.dynamic_classes = torch.tensor([2, 3, 4, 5, 6, 7, 9, 10])
        self.original_h, self.original_w = original_resolution
        self.delete_duplicates = delete_duplicates
        self.max_depth = max_depth

    def __call__(self, results):
        cam2ego_l_t, intrins = calculate_transformations(results, len(self.temporal_frame_ids), self.zero_index, scale_factor=self.downscale_factor, crop_top=self.crop_top)
        ego_l_t2cam = torch.inverse(cam2ego_l_t)
        all_masks = []
        all_depths = []
        all_rgbs = []
        for i, frame in enumerate(results['render_frames']):
            # Semantic labels
            if self.grounded_sam_root is not None:
                masks_path = os.path.join(self.grounded_sam_root, results['scene_name'], f"{frame['token']}.npy")
                masks_f = torch.tensor(np.load(masks_path, allow_pickle=True))
                if self.downscale_factor != 1:
                    masks_f = F.interpolate(masks_f[:, None, ...].float(), scale_factor=self.downscale_factor, mode='nearest').squeeze(1).long()
                if self.crop_top != 0:
                    masks_f = masks_f[:, self.crop_top:, :]
                all_masks.append(masks_f)
            # Depth labels
            if self.depth_root is not None:
                depth_path = os.path.join(self.depth_root, results['scene_name'], f"{frame['token']}.npy")
                depth_f = torch.tensor(np.load(depth_path, allow_pickle=True))
                if self.downscale_factor != 1:
                    depth_f = F.interpolate(depth_f[:, None, ...].float(), scale_factor=self.downscale_factor, mode='bilinear').squeeze(1)
                if self.crop_top != 0:
                    depth_f = depth_f[:, self.crop_top:, :]
                all_depths.append(depth_f)
            # RGB labels
            if self.load_rgb:
                frame_imgs = [np.array(Image.open(frame['cams'][cam]['data_path'])) for cam in results['cam_names']] # load images
                frame_imgs = [cv2.resize(img, None, None, self.downscale_factor, self.downscale_factor)[self.crop_top:] for img in frame_imgs] # resize & crop
                frame_imgs = torch.stack([mmlabNormalize(img) for img in frame_imgs]) # normalize rgb
                all_rgbs.append(frame_imgs)

            # TODO: Dynmamic masking if specified

        # Combine labels and swap dims so that the cameras are in the correct order
        semantics = torch.stack(all_masks)[:, [self.save_order.index(cam) for cam in results['cam_names']], ...] if self.grounded_sam_root is not None else None # [T, nC, H, W]
        depths = torch.stack(all_depths) if self.depth_root is not None else None # [T, nC, H, W]
        rgbs = torch.stack(all_rgbs).permute(0, 1, 3, 4, 2) if self.load_rgb else None # [T, nC, H, W, 3]

        # Filter by max depth
        if depths is not None:
            depth_mask = depths > self.max_depth
            if semantics is not None:
                semantics[depth_mask] = 0
            if rgbs is not None:
                rgbs[depth_mask] = 0.
            depths[depth_mask] = 0.

        results['gs_gts'] = {}
        if semantics is not None:
            results['gs_gts']['semantic'] = semantics
        if depths is not None:
            results['gs_gts']['depth'] = depths
        if rgbs is not None:
            results['gs_gts']['rgb'] = rgbs

        # Provide the intrinsics and extrinsics for rendering the gaussians
        results['gs_intrins'] = intrins.float() # [T, C, 3, 3]
        results['gs_extrins'] = ego_l_t2cam # [T, C, 4, 4]

        return results

@PIPELINES.register_module()
class GaussianFlowOcc_GeneratePseudoLabelsHorizon():
    def __init__(self, grounded_sam_root='data/grounded_sam_nusc', depth_root='data/metric_3d_nusc', temporal_frame_ids=[0], downscale_factor=.44, crop_top=140, mask_dynamic=True,
                 original_resolution=(900, 1600), delete_duplicates=True, load_rgb=False, normalize_rgb=False, max_depth=40, num_frames=3, select_frames=None):
        self.grounded_sam_root = grounded_sam_root
        self.depth_root = depth_root
        self.load_rgb = load_rgb
        self.temporal_frame_ids = temporal_frame_ids
        self.zero_index = [i for i, t in enumerate(temporal_frame_ids) if t==0][0]
        self.downscale_factor = downscale_factor # How much to downscale render resolution
        self.save_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
        self.crop_top = crop_top
        self.mask_dynamic = mask_dynamic
        self.dynamic_classes = torch.tensor([2, 3, 4, 5, 6, 7, 9, 10])
        self.original_h, self.original_w = original_resolution
        self.delete_duplicates = delete_duplicates
        self.max_depth = max_depth
        self.num_frames = num_frames
        self.normalize_rgb = normalize_rgb
        self.select_frames = select_frames

    def __call__(self, results):
        cam2ego_l_t, intrins = calculate_transformations(results, len(self.temporal_frame_ids), self.zero_index, scale_factor=self.downscale_factor, crop_top=self.crop_top)
        ego_l_t2cam = torch.inverse(cam2ego_l_t)
        all_masks = []
        all_depths = []
        all_rgbs = []

        # Alwas take current time step
        valid_frame_indices = [i for i in range(len(self.temporal_frame_ids)) if i!=self.zero_index and results['render_frames'][i]['valid']]
        if self.select_frames is None:
            selected_frames = np.random.choice(valid_frame_indices, min(self.num_frames, len(valid_frame_indices)), replace=False)
            selected_frames = np.concatenate((selected_frames, [self.zero_index])).astype(np.int)
        else:
            selected_frames = np.array([self.temporal_frame_ids.index(i) for i in self.select_frames])
        render_frames = [results['render_frames'][i] for i in selected_frames]
        intrins, ego_l_t2cam = intrins[selected_frames], ego_l_t2cam[selected_frames]

        for i, frame in enumerate(render_frames):
            # Semantic labels
            if self.grounded_sam_root is not None:
                masks_path = os.path.join(self.grounded_sam_root, results['scene_name'], f"{frame['token']}.npy")
                masks_f = torch.tensor(np.load(masks_path, allow_pickle=True))
                if self.downscale_factor != 1:
                    masks_f = F.interpolate(masks_f[:, None, ...].float(), scale_factor=self.downscale_factor, mode='nearest').squeeze(1).long()
                if self.crop_top != 0:
                    masks_f = masks_f[:, self.crop_top:, :]
                all_masks.append(masks_f)
            # Depth labels
            if self.depth_root is not None:
                depth_path = os.path.join(self.depth_root, results['scene_name'], f"{frame['token']}.npy")
                depth_f = torch.tensor(np.load(depth_path, allow_pickle=True))
                if self.downscale_factor != 1:
                    depth_f = F.interpolate(depth_f[:, None, ...].float(), scale_factor=self.downscale_factor, mode='bilinear').squeeze(1)
                if self.crop_top != 0:
                    depth_f = depth_f[:, self.crop_top:, :]
                all_depths.append(depth_f)
            # RGB labels
            if self.load_rgb:
                frame_imgs = [np.array(Image.open(frame['cams'][cam]['data_path'])) for cam in results['cam_names']] # load images
                frame_imgs = [cv2.resize(img, None, None, self.downscale_factor, self.downscale_factor)[self.crop_top:] for img in frame_imgs] # resize & crop
                if self.normalize_rgb:
                    frame_imgs = torch.stack([mmlabNormalize(img) for img in frame_imgs])  # normalize rgb
                else:
                    frame_imgs = torch.stack([torch.tensor(img) / 255. for img in frame_imgs]).permute(0, 3, 1, 2)
                all_rgbs.append(frame_imgs)

            # TODO: Dynmamic masking if specified

        # Combine labels and swap dims so that the cameras are in the correct order
        semantics = torch.stack(all_masks)[:, [self.save_order.index(cam) for cam in results['cam_names']], ...] if self.grounded_sam_root is not None else None # [T, nC, H, W]
        depths = torch.stack(all_depths) if self.depth_root is not None else None # [T, nC, H, W]
        rgbs = torch.stack(all_rgbs).permute(0, 1, 3, 4, 2) if self.load_rgb else None # [T, nC, H, W, 3]

        # Filter by max depth
        if depths is not None:
            depth_mask = depths > self.max_depth
            if semantics is not None:
                semantics[depth_mask] = 0
            if rgbs is not None:
                rgbs[depth_mask] = 0.
            depths[depth_mask] = 0.

        results['gs_gts'] = {}
        if semantics is not None:
            results['gs_gts']['semantic'] = semantics
        if depths is not None:
            results['gs_gts']['depth'] = depths
        if rgbs is not None:
            results['gs_gts']['rgb'] = rgbs

        # Provide the intrinsics and extrinsics for rendering the gaussians
        results['gs_intrins'] = intrins.float() # [T, C, 3, 3]
        results['gs_extrins'] = ego_l_t2cam # [T, C, 4, 4]
        results['selected_frames'] = selected_frames

        return results
        
@PIPELINES.register_module()
class GaussianFlowOcc_GeneratePseudoLabelsPixel():
    def __init__(self, grounded_sam_root='data/grounded_sam_nusc', depth_root='data/metric_3d_nusc', temporal_frame_ids=[0], downscale_factor=.44, crop_top=140, mask_dynamic=True,
                 original_resolution=(900, 1600), delete_duplicates=True, load_rgb=False, max_depth=40, max_pixels=40000):
        self.grounded_sam_root = grounded_sam_root
        self.depth_root = depth_root
        self.load_rgb = load_rgb
        self.temporal_frame_ids = temporal_frame_ids
        self.zero_index = [i for i, t in enumerate(temporal_frame_ids) if t==0][0]
        self.downscale_factor = downscale_factor # How much to downscale render resolution
        self.save_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
        self.crop_top = crop_top
        self.mask_dynamic = mask_dynamic
        self.dynamic_classes = torch.tensor([2, 3, 4, 5, 6, 7, 9, 10])
        self.original_h, self.original_w = original_resolution
        self.delete_duplicates = delete_duplicates
        self.max_depth = max_depth
        self.max_pixels = max_pixels

    def __call__(self, results):
        cam2ego_l_t, intrins = calculate_transformations(results, len(self.temporal_frame_ids), self.zero_index, scale_factor=self.downscale_factor, crop_top=self.crop_top)
        ego_l_t2cam = torch.inverse(cam2ego_l_t)
        all_masks = []
        all_depths = []
        all_rgbs = []
        for i, frame in enumerate(results['render_frames']):
            # Semantic labels
            if self.grounded_sam_root is not None:
                masks_path = os.path.join(self.grounded_sam_root, results['scene_name'], f"{frame['token']}.npy")
                masks_f = torch.tensor(np.load(masks_path, allow_pickle=True))
                if self.downscale_factor != 1:
                    masks_f = F.interpolate(masks_f[:, None, ...].float(), scale_factor=self.downscale_factor, mode='nearest').squeeze(1).long()
                if self.crop_top != 0:
                    masks_f = masks_f[:, self.crop_top:, :]
                all_masks.append(masks_f)
            # Depth labels
            if self.depth_root is not None:
                depth_path = os.path.join(self.depth_root, results['scene_name'], f"{frame['token']}.npy")
                depth_f = torch.tensor(np.load(depth_path, allow_pickle=True))
                if self.downscale_factor != 1:
                    depth_f = F.interpolate(depth_f[:, None, ...].float(), scale_factor=self.downscale_factor, mode='bilinear').squeeze(1)
                if self.crop_top != 0:
                    depth_f = depth_f[:, self.crop_top:, :]
                all_depths.append(depth_f)
            # RGB labels
            if self.load_rgb:
                frame_imgs = [np.array(Image.open(frame['cams'][cam]['data_path'])) for cam in results['cam_names']] # load images
                frame_imgs = [cv2.resize(img, None, None, self.downscale_factor, self.downscale_factor)[self.crop_top:] for img in frame_imgs] # resize & crop
                frame_imgs = torch.stack([mmlabNormalize(img) for img in frame_imgs]) # normalize rgb
                all_rgbs.append(frame_imgs)

            # TODO: Dynmamic masking if specified

        # Combine labels and swap dims so that the cameras are in the correct order
        semantics = torch.stack(all_masks)[:, [self.save_order.index(cam) for cam in results['cam_names']], ...] if self.grounded_sam_root is not None else None # [T, nC, H, W]
        depths = torch.stack(all_depths) if self.depth_root is not None else None # [T, nC, H, W]
        rgbs = torch.stack(all_rgbs).permute(0, 1, 3, 4, 2) if self.load_rgb else None # [T, nC, H, W, 3]

        # Filter by max depth
        if depths is not None:
            depth_mask = depths > self.max_depth
            if semantics is not None:
                semantics[depth_mask] = 0
            if rgbs is not None:
                rgbs[depth_mask] = 0.
            depths[depth_mask] = 0.

        # 1. Sample random pixel indices
        indices = semantics.nonzero()
        selected_indices = indices[torch.randperm(len(indices))[:self.max_pixels]]
        # 2. Retrieve the values from the indices
        selected_semantics = semantics[tuple(selected_indices.T)]
        selected_depths = depths[tuple(selected_indices.T)] if depths is not None else torch.empty(0)
        selected_rgbs = rgbs[tuple(selected_indices.T)] if rgbs is not None else torch.empty(0)
        # 3. Concatenate everything
        pixels = torch.cat((selected_indices, selected_depths[:, None], selected_semantics[:, None], selected_rgbs), dim=-1)

        results['gs_gts_pixel'] = pixels

        # Provide the intrinsics and extrinsics for rendering the gaussians
        results['gs_intrins'] = intrins.float() # [T, C, 3, 3]
        results['gs_extrins'] = ego_l_t2cam # [T, C, 4, 4]

        return results

@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results
    
@PIPELINES.register_module()
class LoadOccGTFromFileTemporal(object):
    def __init__(self, gt_root):
        self.gt_root = gt_root
    def __call__(self, results):
        # Load adjacent frame occupancy (from "selected_frames" in the pipeline)
        all_semantics = []
        all_mask_lidar = []
        all_mask_camera = []
        for frame in results['selected_frames']:
            token = results['render_frames'][frame]['token']
            occ_gt_path = os.path.join(self.gt_root, results['scene_name'], f'{token}', 'labels.npz')
            occ_labels = np.load(occ_gt_path)
            all_semantics.append(occ_labels['semantics'])
            all_mask_lidar.append(occ_labels['mask_lidar'])
            all_mask_camera.append(occ_labels['mask_camera'])
            
        # Concatenate all labels
        all_semantics = np.stack(all_semantics, axis=0)
        all_mask_lidar = np.stack(all_mask_lidar, axis=0)
        all_mask_camera = np.stack(all_mask_camera, axis=0)
        
        results['voxel_semantics'] = all_semantics
        results['mask_lidar'] = all_mask_lidar
        results['mask_camera'] = all_mask_camera

        return results

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int, optional): Number of sweeps. Defaults to 10.
        load_dim (int, optional): Dimension number of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimension to use.
            Defaults to [0, 1, 2, 4].
        time_dim (int, optional): Which dimension to represent the timestamps
            of each points. Defaults to 4.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool, optional): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool, optional): Whether to remove close points.
            Defaults to False.
        test_mode (bool, optional): If `test_mode=True`, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 time_dim=4,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.time_dim = time_dim
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, self.time_dim] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, self.time_dim] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadAdjacentPointsFromFile(object):
    """ Can also load adjacent points clouds"""
    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None
        points_class = get_points_type(self.coord_type)

        for i in range(len(results['render_frames'])):
            # Load point clouds for temporal ray generation
            pts_filename_adj = results['render_frames'][i]['lidar_path']
            points_adj = self._load_points(pts_filename_adj).reshape(-1, self.load_dim)[:, self.use_dim]
            results['render_frames'][i]['points'] =  points_class(points_adj, points_dim=points_adj.shape[-1], attribute_dims=attribute_dims)

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def __call__(self, results):
        assert 'points' in results
        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype=np.int64,
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str


@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, c2e, intrins = results['img_inputs'][:5]
        post_rots, post_trans, bda = results['img_inputs'][5:]
        depth_map_list = []
        # lidar2imgs = []
        # points_imgs = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            depth_map_list.append(depth_map)

        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results

def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    # denorm = denormalize(img)
    return img

def denormalize(img):
    mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float64).reshape(1, -1)
    std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float64).reshape(1, -1)
    img = img.permute(0,1,3,4,2) * std
    img = img + mean
    img = img.to(torch.uint8).contiguous()
    return img

@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor_transforms(self, cam_info, cam_name):
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        sensor2egos = []
        ego2globals = []
        cam2ego_lidars = []
        intrins = []
        post_rots = []
        post_trans = []
        ego_l2globals = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        # EDIT: Compute global2ego_lidar for later use
        ego_lidar2global = torch.eye(4)
        ego_lidar2global[:3, :3] = torch.tensor(Quaternion(results['curr']['ego2global_rotation']).rotation_matrix)
        ego_lidar2global[:3, 3] = torch.tensor(results['curr']['ego2global_translation'])
        global2ego_lidar = torch.inverse(ego_lidar2global)
        ego_l2globals.append(ego_lidar2global)

        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)
            
            # EDIT: Also compute cam2ego_lidar
            cam2global = ego2global.matmul(sensor2ego)
            cam2ego_lidar = global2ego_lidar.matmul(cam2global)

            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            cam2ego_lidars.append(cam2ego_lidar)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])
                # EDIT: Also compute cam2ego_lidar
                ego_lidar2global = torch.eye(4)
                ego_lidar2global[:3, :3] = torch.tensor(Quaternion(adj_info['ego2global_rotation']).rotation_matrix)
                ego_lidar2global[:3, 3] = torch.tensor(adj_info['ego2global_translation'])
                ego_l2globals.append(ego_lidar2global)

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    global2ego_lidar = torch.inverse(ego_lidar2global)
                    cam2global = ego2global.matmul(sensor2ego)
                    cam2ego_lidar = global2ego_lidar.matmul(cam2global)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)
                    cam2ego_lidars.append(cam2ego_lidar)

        if self.sequential:
            N = len(results['cam_names'])
            T = len(results['adjacent']) + 1
            imgs = torch.stack(imgs).reshape(N, T, 3, *imgs[0].shape[1:]).permute(1, 0, 2, 3, 4) 
            sensor2egos = torch.stack(sensor2egos).reshape(T, N, 4, 4)
            ego2globals = torch.stack(ego2globals).reshape(T, N, 4, 4)
            cam2ego_lidars = torch.stack(cam2ego_lidars).reshape(T, N, 4, 4)
            intrins = torch.stack(intrins).reshape(T, N, 3, 3)
            post_rots = torch.stack(post_rots).reshape(T, N, 3, 3)
            post_trans = torch.stack(post_trans).reshape(T, N, 3)
            ego_l2globals = torch.stack(ego_l2globals)
        else:
            imgs = torch.stack(imgs)
            sensor2egos = torch.stack(sensor2egos)
            ego2globals = torch.stack(ego2globals)
            cam2ego_lidars = torch.stack(cam2ego_lidars)
            intrins = torch.stack(intrins)
            post_rots = torch.stack(post_rots)
            post_trans = torch.stack(post_trans)
            ego_l2globals = torch.stack(ego_l2globals).squeeze(0)
        results['canvas'] = canvas
        return (imgs, sensor2egos, ego2globals, ego_l2globals, cam2ego_lidars, intrins, post_rots, post_trans)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        if results['ann_infos'] is not None:
            gt_boxes, gt_labels = results['ann_infos']
            gt_boxes, gt_labels = torch.Tensor(np.array(gt_boxes)), torch.tensor(np.array(gt_labels))
        else:
            gt_boxes, gt_labels = torch.empty(0, 9), torch.empty(0)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        # imgs, rots, trans, intrins = results['img_inputs'][:4]
        # post_rots, post_trans = results['img_inputs'][4:]
        imgs, rots, trans, e2g, c2e, intrins, post_rots, post_trans = results['img_inputs']
        results['img_inputs'] = (imgs, rots, trans, e2g, c2e, intrins, post_rots,
                                 post_trans, bda_rot)
        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy
        if 'voxel_semantics' in results:
            if flip_dx:
                results['voxel_semantics'] = results['voxel_semantics'][::-1,...].copy()
                if 'mask_lidar' in results.keys():
                    results['mask_lidar'] = results['mask_lidar'][::-1,...].copy()
                if 'mask_camera' in results.keys():
                    results['mask_camera'] = results['mask_camera'][::-1,...].copy()
            if flip_dy:
                results['voxel_semantics'] = results['voxel_semantics'][:,::-1,...].copy()
                if 'mask_lidar' in results.keys():
                    results['mask_lidar'] = results['mask_lidar'][:,::-1,...].copy()
                if 'mask_camera' in results.keys():
                    results['mask_camera'] = results['mask_camera'][:,::-1,...].copy()
        return results
