# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_loss, build_head
from mmdet3d.models.gaussianflowocc_modules.utils import move_gaussians_temporal_module
from .mvx_two_stage import MVXTwoStageDetector
from mmcv.cnn.bricks.transformer import build_feedforward_network
from gsplat import quat_scale_to_covar_preci

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Imported from pytorch3d https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_raw_multiply"""
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

@DETECTORS.register_module()
class GaussianFlowOcc(MVXTwoStageDetector):

    def __init__(self,
                 num_classes=16,
                 with_others=False,
                 in_channels=128,
                 rasterizer=None,
                 temporal_module=None,
                 gaussian_decoder=None,
                 voxel_grid_cfg=None,
                 eval_threshold_range=[.1],
                 gaussian_init_scale=2,
                 num_gaussians=10000,
                 max_neighborhood=4,
                 use_opacity=True,
                 use_scale=True,
                 use_rotation=True,
                 scale_act=True,
                 render_semantic=True,
                 render_depth=True,
                 render_rgb=False,
                 sh_degree=0,
                 initial_mean=True,
                 scale_range=None, # [0.05, .32],
                 temporal_frame_ids=[0],
                 scale_multiplier=1.,
                 move_dynamic_gaussians=False,
                 use_movement_reg=False,
                 num_head_layers=1,
                 use_mask=True,
                 loss_occ_density=None,
                 loss_occ_semantics=None,
                 temporal_loss_3d=False,
                 **kwargs):
        super(GaussianFlowOcc, self).__init__(**kwargs)
        self.pts_bbox_head = None
        self.with_others = with_others
        self.num_classes = num_classes + 1 if with_others else num_classes
        self.eval_threshold_range = eval_threshold_range
        self.voxel_grid_cfg = voxel_grid_cfg
        self.max_neighborhood = max_neighborhood # in each direction in 3D
        self.voxel_centers = None
        self.prev_feat = None
        self.dynamic_classes = torch.tensor([2, 3, 4, 5, 6, 7, 9, 10])
        self.temporal_frame_ids = torch.tensor(temporal_frame_ids)
        self.zero_index = [i for i, t in enumerate(temporal_frame_ids) if t==0][0]

        self.gaussian_decoder = build_feedforward_network(gaussian_decoder) if gaussian_decoder is not None else None
        self.rasterizer = build_head(rasterizer) if rasterizer is not None else None
        self.move_dynamic_gaussians = move_dynamic_gaussians
        self.next_t_index = [i for i, t in enumerate(temporal_frame_ids) if t==1][0] if 1 in temporal_frame_ids else None

        self.scale_multiplier = scale_multiplier

        # Initial gaussian properties
        if num_gaussians is None:
            sparse_grid_cfg = self.voxel_grid_cfg.deepcopy()
            sparse_grid_cfg['x'][2] *= gaussian_init_scale
            sparse_grid_cfg['y'][2] *= gaussian_init_scale
            sparse_grid_cfg['z'][2] *= gaussian_init_scale
            self.initial_means = nn.Parameter(self.create_voxel_centers(sparse_grid_cfg).flatten(0, -2), requires_grad=initial_mean)
        else:
            self.initial_means = nn.Parameter(self.create_voxel_centers_random(self.voxel_grid_cfg, num_gaussians), requires_grad=initial_mean)
        # Depth
        self.render_depth = render_depth

        # Semantic Head
        self.render_semantic = render_semantic
        semantic_layers = [nn.Linear(in_channels, in_channels*2), nn.LeakyReLU(inplace=True)]
        for i in range(num_head_layers):
            semantic_layers.append(nn.Linear(in_channels*2, in_channels*2))
            semantic_layers.append(nn.LeakyReLU(inplace=True))
        semantic_layers.append(nn.Linear(in_channels*2, self.num_classes))
        self.semantic_head = nn.Sequential(*semantic_layers)

        # RGB Head
        self.render_rgb = render_rgb
        self.sh_degree = sh_degree
        rgb_layers = [nn.Linear(in_channels, in_channels*2), nn.LeakyReLU(inplace=True)]
        for i in range(num_head_layers):
            rgb_layers.append(nn.Linear(in_channels*2, in_channels*2))
            rgb_layers.append(nn.LeakyReLU(inplace=True))
        rgb_layers.append(nn.Linear(in_channels*2, 3*((sh_degree+1)**2)))
        self.rgb_head = nn.Sequential(*rgb_layers) if self.render_rgb else None

        # Opacity Head
        opacity_layers = [nn.Linear(in_channels, in_channels*2), nn.LeakyReLU(inplace=True)]
        for i in range(num_head_layers):
            opacity_layers.append(nn.Linear(in_channels*2, in_channels*2))
            opacity_layers.append(nn.LeakyReLU(inplace=True))
        opacity_layers.append(nn.Linear(in_channels*2, 1))
        opacity_layers.append(nn.Sigmoid())
        self.opacity_head = nn.Sequential(*opacity_layers) if use_opacity else None

        # Scale Head
        scale_layers = [nn.Linear(in_channels, in_channels*2), nn.LeakyReLU(inplace=True)]
        for i in range(num_head_layers):
            scale_layers.append(nn.Linear(in_channels*2, in_channels*2))
            scale_layers.append(nn.LeakyReLU(inplace=True))
        scale_layers.append(nn.Linear(in_channels*2, 3))
        if scale_act:
            scale_layers.append(nn.Sigmoid())
        self.scale_head = nn.Sequential(*scale_layers) if use_scale else None
        self.scale_range = scale_range

        # Rotation Head
        rotation_layers = [nn.Linear(in_channels, in_channels*2), nn.LeakyReLU(inplace=True)]
        for i in range(num_head_layers):
            rotation_layers.append(nn.Linear(in_channels*2, in_channels*2))
            # rotation_layers.append(nn.LayerNorm(in_channels))
            rotation_layers.append(nn.LeakyReLU(inplace=True))
        rotation_layers.append(nn.Linear(in_channels*2, 4))
        self.rotation_head = nn.Sequential(*rotation_layers) if use_rotation else None

        # Learnable query vector
        self.gaussian_queries = nn.Parameter(torch.empty(self.initial_means.shape[0], in_channels))
        nn.init.normal_(self.gaussian_queries)

        # Temporal Module
        self.temporal_module = build_feedforward_network(temporal_module) if temporal_module is not None else None
        self.movement_regularizer = build_loss(dict(type='MovementRegularizer')) if temporal_module is not None and use_movement_reg else None

        # 3D loss
        self.use_mask = use_mask
        self.temporal_loss_3d = temporal_loss_3d
        self.loss_occ_density = build_loss(loss_occ_density) if loss_occ_density is not None else None
        self.loss_occ_semantics = build_loss(loss_occ_semantics) if loss_occ_semantics is not None else None

    def create_voxel_centers(self, grid_cfg):
        Z = int((grid_cfg['z'][1] - grid_cfg['z'][0]) / grid_cfg['z'][2])
        H = int((grid_cfg['x'][1] - grid_cfg['x'][0]) / grid_cfg['x'][2])
        W = int((grid_cfg['y'][1] - grid_cfg['y'][0]) / grid_cfg['y'][2])
        self.Z, self.H, self.W = Z, H, W

        xs = torch.linspace(0.5 * grid_cfg['x'][2] + grid_cfg['x'][0], grid_cfg['x'][1] - 0.5 * grid_cfg['x'][2], W).view(W, 1, 1).expand(W, H, Z)
        ys = torch.linspace(0.5 * grid_cfg['y'][2] + grid_cfg['y'][0], grid_cfg['y'][1] - 0.5 * grid_cfg['y'][2], H).view(1, H, 1).expand(W, H, Z)
        zs = torch.linspace(0.5 * grid_cfg['z'][2] + grid_cfg['z'][0], grid_cfg['z'][1] - 0.5 * grid_cfg['z'][2], Z).view(1, 1, Z).expand(W, H, Z)
        
        ref_3d = torch.stack((xs, ys, zs), -1)
        return ref_3d
    
    def create_voxel_centers_random(self, grid_cfg, N):

        x = torch.rand(N).uniform_(grid_cfg['x'][0], grid_cfg['x'][1])
        y = torch.rand(N).uniform_(grid_cfg['y'][0], grid_cfg['y'][1])
        z = torch.rand(N).uniform_(grid_cfg['z'][0], grid_cfg['z'][1])
        
        points = torch.vstack((x, y, z)).T
        return points
    
    def init_weights(self):
        super().init_weights()
        # initialize heads
        for m in self.semantic_head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=.01)
                nn.init.constant_(m.bias, 0)
        if self.render_rgb:
            for m in self.rgb_head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=.01)
                    nn.init.constant_(m.bias, 0)
        if self.opacity_head is not None:
            for m in self.opacity_head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=.01)
                    nn.init.constant_(m.bias, 0)
        if self.scale_head is not None:
            for m in self.scale_head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=.01)
                    nn.init.constant_(m.bias, 0)
        if self.rotation_head is not None:
            for m in self.rotation_head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=.01)
                    nn.init.constant_(m.bias, 0)

    def gaussians_to_occupancy(self, means, quats, scale, opacity, feature,):
        # Voxelize Gaussians 
        if self.voxel_centers is None:
            self.voxel_centers = self.create_voxel_centers(self.voxel_grid_cfg).to(means.device)
            self.min_positions = torch.tensor([self.voxel_grid_cfg['x'][0], self.voxel_grid_cfg['y'][0], self.voxel_grid_cfg['z'][0]], device=means.device)
        _, inv_covariance = quat_scale_to_covar_preci(quats, scale * self.scale_multiplier)

        mean_vox_index = torch.floor((means - self.min_positions) / self.voxel_grid_cfg['x'][2])
        neighborhood = torch.arange(-self.max_neighborhood, self.max_neighborhood+1, device=means.device)
        neighborhood = torch.stack(torch.meshgrid(neighborhood, neighborhood, neighborhood, indexing='ij'), dim=-1).reshape(-1, 3)
        neighborhood_index = (mean_vox_index[:, None, :] + neighborhood[None, ...]).long()
        neighborhood_index_mask = ((neighborhood_index >= 0) & (neighborhood_index < torch.tensor([self.W, self.H, self.Z], device=means.device))).all(dim=-1)
        neighborhood_index_flat = neighborhood_index[neighborhood_index_mask]
        neighborhood_coords = self.voxel_centers[tuple(neighborhood_index_flat.T)]
        diff = neighborhood_coords - means[:, None, :].repeat(1, neighborhood.shape[0], 1)[neighborhood_index_mask]
        cov_batched = inv_covariance[:, None, ...].repeat(1, neighborhood.shape[0], 1, 1)[neighborhood_index_mask]
        contribs = torch.exp(-0.5 * ((diff[..., None, :] @ cov_batched) @ diff[..., :, None])).squeeze()

        # distribute opacity and features to each voxel
        opacities = opacity[:, None, 0].repeat(1, neighborhood.shape[0])[neighborhood_index_mask] * contribs
        features = feature[:, None, :].repeat(1, neighborhood.shape[0], 1)[neighborhood_index_mask] * contribs[..., None]
        voxel_occupancy = torch.zeros((self.W, self.H, self.Z), device=means.device)
        voxel_semantics = torch.zeros((self.W, self.H, self.Z, self.num_classes), device=means.device)
        indices_unique, labels = neighborhood_index_flat.unique(dim=0, return_inverse=True)
        grouped_opacities = torch.zeros((indices_unique.size(0),), device=labels.device).scatter_add(0, labels, opacities)
        grouped_semantics = torch.zeros((indices_unique.size(0), self.num_classes), device=labels.device).scatter_add(0, labels[:, None].expand_as(features), features)
        voxel_occupancy[tuple(indices_unique.T)] = grouped_opacities.clamp(0, 1)
        voxel_semantics[tuple(indices_unique.T)] = grouped_semantics
        return voxel_occupancy, voxel_semantics

    def ego_motion_compensation(self, ego2global_next):
        prev_means, prev_feat = self.prev_feat[..., :3], self.prev_feat[..., 3:]
        # Apply gaussian flow first
        if self.move_dynamic_gaussians and self.temporal_module is not None:
            offsets = self.temporal_module(prev_feat, [torch.tensor([self.next_t_index]).numpy() for i in range(ego2global_next.shape[0])])
            prev_means = move_gaussians_temporal_module(prev_means, self.semantic_head(prev_feat), offsets, self.dynamic_classes)[:, 0]

        # Then transform to next frame
        cur2next = torch.inverse(ego2global_next) @ self.prev_ego2global
        prev_means = (cur2next[:, None, ...] @ torch.cat((prev_means, prev_means.new_ones(*prev_means.shape[:2], 1)), dim=-1)[..., None])[..., :3, 0]
        self.prev_feat = torch.cat([prev_means, prev_feat], dim=-1)

    def forward_test(self, img_metas, img_inputs=None, **kwargs):
        num_augs = len(img_metas)
        if num_augs == 1:
            return self.simple_test(img_metas[0], img_inputs=img_inputs[0], **kwargs)

    def simple_test(self,
                    img_metas,
                    img_inputs=None,
                    render_preds=False,
                    clip_low_density_regions=True,
                    gs_intrins=None,
                    gs_extrins=None,
                    return_means=False,
                    **kwargs):        
        # Reset stored self.prev_feat when the sample is the first sample of a scene
        if not img_metas[0]['has_prev_sample']:
            prev_means, prev_feature = self.forward_gaussian(img_inputs, img_metas, None)
            self.prev_feat = torch.cat([prev_means, prev_feature], dim=-1)
            self.prev_ego2global = img_inputs[3]
        # Apply ego-motion compensation
        self.ego_motion_compensation(img_inputs[3])
        out_dict = {}

        if self.gaussian_decoder.store_intermediate:
            # TODO: Needs major rework
            gaussians_per_block = self.forward_gaussian(img_inputs, img_metas, self.prev_feat)
            means = torch.stack(([g[0] for g in gaussians_per_block]))
            quats = torch.stack(([g[1] for g in gaussians_per_block]))
            scale = torch.stack(([g[2] for g in gaussians_per_block]))
            opacity = torch.stack(([g[3] for g in gaussians_per_block]))
            feature = torch.stack(([g[4] for g in gaussians_per_block]))
            # velocity = torch.stack(([g[5] for g in gaussians_per_block]))
            self.prev_feat = torch.cat([means[0], quats[0], scale[0], opacity[0], feature[0]], dim=-1)
            # self.prev_feat = torch.cat([means[0], quats[0], scale[0], opacity[0], velocity[0], feature[0]], dim=-1) if velocity[0].abs().sum()>1e-2 else None
            sem_feature = self.semantic_head(feature)
            occupancy = [self.gaussians_to_occupancy(means[i][0], quats[i][0], scale[i][0], opacity[i][0], sem_feature[i][0]) for i in range(len(means))]
            density = torch.stack(([o[0] for o in occupancy])) 
            semantics = torch.stack(([o[1] for o in occupancy]))
            if clip_low_density_regions:
                density[density<1e-3] = 0
                density[..., 11:] = 0
            
            out_dict['previous_density'] = density[:-1]
            out_dict['previous_occ'] = occupancy[:-1]
            out_dict['previous_means'] = means[:-1]
            
        else:
            means, feature = self.forward_gaussian(img_inputs, img_metas, self.prev_feat)
            self.prev_feat = torch.cat([means, feature], dim=-1)
            self.prev_ego2global = img_inputs[3]
            sem_feature = self.semantic_head(feature)
            opacity = self.opacity_head(feature) if self.opacity_head is not None else torch.ones_like(means[..., :1])
            scale = self.scale_head(feature) if self.scale_head is not None else torch.full_like(means, .3)
            if self.scale_range is not None:
                scale = self.scale_range[0] + scale * (self.scale_range[1] - self.scale_range[0])
            quats = F.normalize(self.rotation_head(feature), dim=-1) if self.rotation_head is not None else torch.tensor([1., 0., 0., 0.], device=means.device).repeat(means.shape[0], means.shape[1], 1)
            density, semantics = self.gaussians_to_occupancy(means[0], quats[0], scale[0], opacity[0], sem_feature[0])

        # clip low density regions & remove roof
        if clip_low_density_regions:
            density[density<1e-3] = 0
            density[..., 11:] = 0

        if render_preds:
            if type(gs_intrins) == list:
                gs_intrins = gs_intrins[0]
                gs_extrins = gs_extrins[0]
            rendered_outs_no_temporal = self.rasterizer(means, quats, scale, opacity, sem_feature, gs_intrins, gs_extrins, mode='RGB+D')
            out_dict['rendered_semantics'] = (rendered_outs_no_temporal[..., :-1].argmax(dim=-1) + 
                                              int(not self.with_others)).squeeze(0).to(torch.uint8).cpu().numpy()
            out_dict['rendered_depths'] = rendered_outs_no_temporal[..., -1].squeeze(0).cpu().numpy()
            
        # combine density and semantics
        semantics = semantics.argmax(dim=-1) + int(not self.with_others)
        free_space = torch.stack([density < tr for tr in self.eval_threshold_range])

        out_dict['occupancy'] = semantics.to(torch.uint8).cpu().numpy()
        out_dict['free_space'] = free_space.cpu().numpy()

        if return_means:
            out_dict['means'] = means.squeeze(0).cpu().numpy()
            out_dict['opacity'] = opacity.squeeze(0).cpu().numpy()
            out_dict['feature'] = sem_feature.squeeze(0).cpu().numpy()
            out_dict['label'] = (sem_feature.argmax(dim=-1) + int(not self.with_others)).squeeze(0).cpu().numpy()
            out_dict['scale'] = scale.squeeze(0).cpu().numpy()
            out_dict['quats'] = quats.squeeze(0).cpu().numpy()

        return [out_dict]

    def prepare_prev_feat(self, img_inputs, img_metas):
        with torch.no_grad():
            prev_feat = None
            T = img_inputs[0].shape[1]
            for t in reversed(range(1, T)): # dont compute for current
                img_inputs_cur = [i[:, t, ...].contiguous() for i in img_inputs[:-1]] + [img_inputs[-1]]
                means, feature = self.forward_gaussian(img_inputs_cur, img_metas, prev_feat)
                # Ego motion compensation
                ego2global_c = img_inputs_cur[3]
                ego2global_next = img_inputs[3][:, t-1]

                # Apply gaussian flow first
                if self.move_dynamic_gaussians and self.temporal_module is not None:
                    offsets = self.temporal_module(feature, [torch.tensor([self.next_t_index]).numpy() for i in range(len(img_metas))])
                    means = move_gaussians_temporal_module(means, self.semantic_head(feature), offsets, self.dynamic_classes)[:, 0]

                # Then transform to next frame
                cur2next = torch.inverse(ego2global_next) @ ego2global_c
                means = (cur2next[:, None, ...] @ torch.cat((means, means.new_ones(*means.shape[:2], 1)), dim=-1)[..., None])[..., :3, 0]
                prev_feat = torch.cat([means, feature], dim=-1)

        return prev_feat

    def forward_gaussian(self, img_inputs, img_metas, prev_feat=None):            
        # Gaussian Init
        input_shape = torch.tensor(img_inputs[0].shape[-2:])
        B = img_inputs[1].shape[0]
        means = self.initial_means[None, ...].repeat(B, 1, 1)
        feature = self.gaussian_queries[None, ...].repeat(B, 1, 1)

        # Extract img feats
        img_feats = self.extract_img_feat(img_inputs[0], img_metas) # [B*N, C, H_f, W_f]

        # Gaussian Transformer
        means, feature = self.gaussian_decoder(means, feature, prev_feat, img_feats, img_inputs[4:8], input_shape)       
        return  means, feature

    def forward_train(self,
                      img_metas=None,
                      img_inputs=None,
                      gs_gts=None,
                      gs_gts_pixel=None,
                      gs_intrins=None,
                      gs_extrins=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      **kwargs):
        # img inputs with temporal:
        # img: [B, T, N, C, H, W], sensor2ego: [B, T, N, 4, 4], ego2global: [B, T, N, 4, 4], ego_l2global: [B, T, 4, 4],
        # cam2ego_l: [B, T, N, 4, 4], intrins: [B, T, N, 3, 3], # post_rot: [B, T, N, 3, 3], post_trans: [B, T, N, 3], bda: [B, 3, 3]
        assert not ((gs_gts is not None) and (gs_gts_pixel is not None)), "Only one of gs_gts or gs_gts_pixel should be provided"
        losses = dict()
        # reset self.prev_feat after eval
        if self.prev_feat is not None:
            self.prev_feat = None

        # Buildup of previous features if using temporal self-attn
        if img_inputs[0].ndim == 6:
            img_inputs_cur = [i[:, 0, ...].contiguous() for i in img_inputs[:-1]] + [img_inputs[-1]]
            prev_feat = self.prepare_prev_feat(img_inputs, img_metas)
        else:
            img_inputs_cur = img_inputs
            prev_feat = None

        means, feature = self.forward_gaussian(img_inputs_cur, img_metas, prev_feat)
        # Estimate gaussian properties
        sem_pred = self.semantic_head(feature)
        opacity = self.opacity_head(feature) if self.opacity_head is not None else torch.ones_like(means[..., :1])
        scale = self.scale_head(feature) if self.scale_head is not None else torch.full_like(means, .3)
        if self.scale_range is not None:
            scale = self.scale_range[0] + scale * (self.scale_range[1] - self.scale_range[0])
        quats = F.normalize(self.rotation_head(feature), dim=-1) if self.rotation_head is not None else torch.tensor([1., 0., 0., 0.], device=means.device).repeat(means.shape[0], means.shape[1], 1)

        # Using temporal module
        if self.temporal_module is not None:
            offsets = self.temporal_module(feature, [i['selected_frames'][:-1] for i in img_metas])
            if self.movement_regularizer is not None:
                losses['loss_movement'] = self.movement_regularizer(offsets, sem_pred)
            means = move_gaussians_temporal_module(means, sem_pred, offsets, self.dynamic_classes)

        # Gaussian Splatting
        rendered_outs = {}
        if self.render_semantic and self.rasterizer is not None:
            rendered_outs_sem = self.rasterizer(means, quats, scale, opacity, sem_pred, gs_intrins, gs_extrins, mode='RGB+D')
            rendered_outs['semantic'] = rendered_outs_sem[..., :-1]
            if self.render_depth:
                rendered_outs['depth'] = rendered_outs_sem[..., -1]
        if self.render_rgb and self.rasterizer is not None:
            rgb_pred = self.rgb_head(feature)
            if self.sh_degree > 0:
                B, N, C = rgb_pred.shape
                rendered_outs['rgb'] = self.rasterizer(means, quats, scale, opacity, rgb_pred.view(B, N, C//3, 3), gs_intrins,
                                                    gs_extrins, mode='RGB', sh_degree=self.sh_degree)
            else:
                rendered_outs['rgb'] = self.rasterizer(means, quats, scale, opacity, rgb_pred, gs_intrins, gs_extrins, mode='RGB')

        # Compute losses
        if gs_gts is not None:
            losses_gs = self.rasterizer.calculate_losses(rendered_outs, gs_gts)
            losses.update(losses_gs)
        elif gs_gts_pixel is not None:
            t, ncams = gs_extrins.shape[1:3]
            rendered_outs = {k: v.unflatten(1, (t, ncams)) for k, v in rendered_outs.items()}
            losses_gs = self.rasterizer.calculate_losses_pixel(rendered_outs, gs_gts_pixel)
            losses.update(losses_gs)

        # Compute 3D losses
        if voxel_semantics is not None:
            # Voxelize Gaussians
            all_semantics = []
            all_densities = []

            if self.temporal_loss_3d:
                assert means.ndim == 4, "Means should be 4D tensor"
                for b in range(means.shape[0]):
                    means_b = means[b]
                    quats_b = quats[b]
                    scale_b = scale[b]
                    opacity_b = opacity[b]
                    sem_pred_b = sem_pred[b]
                    for t in range(means_b.shape[0]):
                        density, semantics = self.gaussians_to_occupancy(means_b[t], quats_b, scale_b, opacity_b, sem_pred_b)
                        all_densities.append(density)
                        all_semantics.append(semantics)
            else:
                if means.ndim > 3:
                    means = means[:, -1]
                for b in range(means.shape[0]):
                    density, semantics = self.gaussians_to_occupancy(means[b], quats[b], scale[b], opacity[b], sem_pred[b])
                    all_densities.append(density)
                    all_semantics.append(semantics)
            density = torch.stack(all_densities)
            semantics = torch.stack(all_semantics)

            voxel_semantics=voxel_semantics.long().reshape(-1)
            density = density.reshape(-1)
            semantics = semantics.reshape(-1, semantics.shape[-1])
            semantic_mask = voxel_semantics!=17
            if not self.with_others:
                semantic_mask = semantic_mask & (voxel_semantics != 0)
                voxel_semantics = voxel_semantics - 1
            if self.use_mask:
                mask_camera = mask_camera.reshape(-1).to(torch.bool)            
                combined_mask = (semantic_mask * mask_camera)
                loss_density = self.loss_occ_density(density[mask_camera], semantic_mask.float()[mask_camera])
                loss_semantic = self.loss_occ_semantics(semantics[combined_mask], voxel_semantics[combined_mask])
            else:
                loss_density = self.loss_occ_density(density, semantic_mask.float())
                loss_semantic = self.loss_occ_semantics(semantics[semantic_mask], voxel_semantics[semantic_mask])

            losses['loss_density_3d'] = loss_density
            losses['loss_semantic_3d'] = loss_semantic

        return losses