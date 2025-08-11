# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
from mmdet3d.models.builder import HEADS
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization
from mmdet3d.models.losses.ssim import SSIM, CustomSSIM

@HEADS.register_module()
class GaussianFlowOccRasterizer(nn.Module):
    def __init__(self, raster_shape=(256, 704), loss_weighting=None, depth_lw=.1, sem_lw=1, rgb_lw=1):
        super().__init__()
        nusc_class_frequencies = torch.tensor([347376232,   7805289,    126214,   4351628,  36046855,   1153917,
            411093,   2137060,    636812,   4397702,  14448748, 316958899,
            8559216,  70197461,  70289730, 178178063, 122581273])
        if loss_weighting == 'filtered':
            log_weights = torch.log(nusc_class_frequencies.sum() / nusc_class_frequencies)
            self.class_weight = (log_weights.numel() / log_weights.sum()) * log_weights # scale to 1 mean
        elif loss_weighting == 'unfiltered':
            self.class_weight = None # TODO: Implement unfiltered weighting
        else:
            self.class_weight = None

        self.height, self.width = raster_shape

        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.loss_fn_depth = nn.MSELoss()
        self.ssim = SSIM()
        self.custom_ssim = CustomSSIM()
        self.l1 = nn.L1Loss(reduction='none')
        self.sem_lw = sem_lw
        self.depth_lw = depth_lw
        self.rgb_lw = rgb_lw
        self.dynamic_classes = torch.tensor([2, 3, 4, 5, 6, 7, 9, 10])
        self.neighbors = torch.cartesian_prod(torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1])).unsqueeze(0)

    def calculate_semantic_loss(self, pred, target):
        pred = pred.flatten(0, -2)
        target = target.flatten().long()
        mask = target != 0
        pixel_weights = target.new_ones(pred[mask].shape[0])
        if self.class_weight is not None:
            pixel_weights = pixel_weights * self.class_weight[target[mask]].to(pixel_weights.device)
        return self.sem_lw * (pixel_weights * self.loss_fn(pred[mask], target[mask]-1)).mean()
        
    def calculate_rgb_loss(self, pred, target):
        h, w, _ = target.shape[-3:]
        pred = pred.flatten(0, -4).permute(0, 3, 1, 2)
        target = target.permute(0,1,2,5,3,4).view(-1, 3, h, w)
        depth_mask = target.abs().sum(1) <= 0
        l1 = self.l1(pred, target).mean(1)
        ssim = self.ssim(pred, target).mean(1)
        l1[depth_mask] = 0
        ssim[depth_mask] = 0
        return self.rgb_lw * (.85 * ssim + .15 * l1).mean()
        
    def calculate_depth_loss(self, pred, target):
        target = target.flatten()
        mask_depth = target > 0
        return self.depth_lw * self.loss_fn_depth(pred.flatten()[mask_depth], target[mask_depth])

    def calculate_losses(self, render_outs, gs_gts):
        losses = {}
        if 'semantic' in render_outs.keys():
            losses['loss_gs_sem'] = self.calculate_semantic_loss(render_outs['semantic'], gs_gts['semantic'])
        if 'rgb' in render_outs.keys():
            losses['loss_gs_rgb'] = self.calculate_rgb_loss(render_outs['rgb'], gs_gts['rgb'])
        if 'depth' in render_outs.keys():
            losses['loss_gs_depth'] = self.calculate_depth_loss(render_outs['depth'], gs_gts['depth'])
        
        return losses
    
    def sample_indices(self, values, indices, coors):
        B, T, N = values.shape[:3]
        values = values.flatten(0, 2).permute(0, 3, 1, 2)

        # reshape coors to match indices
        flatted_indices = indices[:, 0] * (T * N) + indices[:, 1] * N + indices[:, 2]
        unique = flatted_indices.unique(return_counts=True)[1]
        max_len = unique.max()
        expanded_indices = torch.cat([torch.tensor(list(range(i)), device=indices.device) for i in unique])
        coors_reshape = coors.new_zeros((B*T*N, max_len, 2))
        for b_idx in range(B*T*N):
            cur_idx = flatted_indices == b_idx
            coors_reshape[b_idx, :cur_idx.sum()] = coors[cur_idx]

        # sample from values
        grid = (coors_reshape[..., [1, 0]] / torch.tensor([self.width, self.height], device=coors_reshape.device)).unsqueeze(1) * 2 - 1
        values_sampled = F.grid_sample(values, grid, align_corners=False).squeeze(2).permute(0, 2, 1)

        # reshape into original shape
        values_sampled = values_sampled[flatted_indices, expanded_indices]
        return values_sampled

    def calculate_semantic_loss_pixel(self, pred, indices, coors, labels_target):
        pred_sampled = self.sample_indices(pred, indices, coors)

        # compute loss
        pixel_weights = coors.new_ones(labels_target.shape[0])
        if self.class_weight is not None:
            pixel_weights = pixel_weights * self.class_weight[labels_target].to(pixel_weights.device)
        return self.sem_lw * (pixel_weights * self.loss_fn(pred_sampled, labels_target)).mean()
        
    def calculate_rgb_loss_pixel(self, pred, indices, coors, rgb_target):
        # Values for L1
        sampled_rgb_pred = self.sample_indices(pred, indices, coors)
        originial_rgb = rgb_target[...,3*4:3*5]
        # Values for SSIM
        indices_neigh = indices.unsqueeze(1).repeat(1, 9, 1)
        coors_neigh = coors.unsqueeze(1) + self.neighbors.to(coors.device)
        sampled_rgb_pred_neigh = self.sample_indices(pred, indices_neigh.flatten(0, 1), coors_neigh.flatten(0, 1)).view(-1, 9, 3)
        # Compute losses
        l1 = self.l1(sampled_rgb_pred, originial_rgb).mean(1)
        ssim = self.custom_ssim(sampled_rgb_pred_neigh, rgb_target.view(-1, 9, 3)).mean(1)
        return (.85 * ssim + .15 * l1).mean()
        
    def calculate_depth_loss_pixel(self, pred, indices, coors, depths_target):
        sampled_depth_pred = self.sample_indices(pred.unsqueeze(-1), indices, coors).squeeze(-1)
        mask_depth = depths_target > 0
        return self.depth_lw * self.loss_fn_depth(sampled_depth_pred[mask_depth], depths_target[mask_depth])
    
    def calculate_losses_pixel(self, render_outs, gs_gts):
        losses = {}
        batch_index = torch.cat([torch.full((gs_gts[i].shape[0],), i, device=gs_gts[i].device) for i in range(len(gs_gts))], dim=0)
        gs_gts = torch.cat((batch_index[..., None], torch.cat((gs_gts))), dim=1)
        indices = gs_gts[:, :3].long()
        coors = gs_gts[:, 3:5]
        depths = gs_gts[:, 5]
        labels = gs_gts[:, 6].long()
        rgb = gs_gts[:, 7:]

        if 'semantic' in render_outs.keys():
            losses['loss_gs_sem'] = self.calculate_semantic_loss_pixel(render_outs['semantic'], indices, coors, labels)
        if 'rgb' in render_outs.keys():
            losses['loss_gs_rgb'] = self.calculate_rgb_loss_pixel(render_outs['rgb'], indices, coors, rgb)
        if 'depth' in render_outs.keys():
            losses['loss_gs_depth'] = self.calculate_depth_loss_pixel(render_outs['depth'], indices, coors, depths)
        
        return losses

    def forward_flow(self, means, quats, scale, opacity, feature, gs_intrins, gs_extrins, mode, sh_degree):
        # Execute rasterization
        all_rendered_feats = []

        for batch_idx in range(means.shape[0]): # we now need to also loop over time
            rendered_features_t = []
            for t in range(means.shape[1]):
                rendered_features, _, _ = rasterization(
                        means[batch_idx, t], quats[batch_idx], scale[batch_idx], opacity[batch_idx, :, 0], feature[batch_idx],
                        gs_extrins[batch_idx, t], gs_intrins[batch_idx, t], self.width, self.height, render_mode=mode, sh_degree=sh_degree
                )
                rendered_features_t.append(rendered_features)
            all_rendered_feats.append(torch.cat(rendered_features_t))
        
        return torch.stack(all_rendered_feats)
    
    def forward_static(self, means, quats, scale, opacity, feature, gs_intrins, gs_extrins, mode, sh_degree):
        # Execute rasterization
        all_rendered_feats = []

        for batch_idx in range(means.shape[0]):
            rendered_features, _, _ = rasterization(
                    means[batch_idx], quats[batch_idx], scale[batch_idx], opacity[batch_idx, :, 0], feature[batch_idx],
                    gs_extrins[batch_idx].flatten(0, 1), gs_intrins[batch_idx].flatten(0, 1), self.width, self.height, render_mode=mode, sh_degree=sh_degree
            )
            all_rendered_feats.append(rendered_features)
        
        return torch.stack(all_rendered_feats)

    def forward(self, means, quats, scale, opacity, feature, gs_intrins, gs_extrins, mode='RGB', sh_degree=None):
        if means.ndim == 4:
            return self.forward_flow(means, quats, scale, opacity, feature, gs_intrins, gs_extrins, mode, sh_degree)
        else:
            return self.forward_static(means, quats, scale, opacity, feature, gs_intrins, gs_extrins, mode, sh_degree)
