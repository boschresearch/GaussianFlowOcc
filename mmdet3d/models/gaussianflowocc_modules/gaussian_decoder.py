# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK, ATTENTION
from mmcv.cnn.bricks.transformer import build_attention, build_feedforward_network, POSITIONAL_ENCODING, build_positional_encoding
from mmcv.runner.base_module import BaseModule
from mmcv.ops.multi_scale_deform_attn import (MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.init import normal_
import math

@POSITIONAL_ENCODING.register_module()
class PointPositionalEncoding(nn.Module):
    """Module for fusion of gaussian properties and features"""
    def __init__(self, point_dim=3, num_layers=1, out_channels=128, pc_range=[-40., -40., -1., 40., 40., 5.4], use_norm=True):
        super().__init__()
        self.out_channels = out_channels
        self.point_dim = point_dim
        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        layers = [nn.Linear(point_dim, out_channels), nn.LeakyReLU()]
        for i in range(num_layers):
            layers.append(nn.Linear(out_channels, out_channels))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(out_channels, out_channels))
        if use_norm: layers.append(nn.LayerNorm(out_channels))
        self.pos_enc = nn.Sequential(*layers)
                
        self.init_weights()
    
    def init_weights(self):
        for m in self.pos_enc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=.01)
                nn.init.constant_(m.bias, 0)    
    
    def forward(self, x):
        x = (x - self.pc_range[:3]) / (self.pc_range[3:] - self.pc_range[:3])
        return self.pos_enc(x[..., :self.point_dim])
    
@FEEDFORWARD_NETWORK.register_module()
class GaussianRectifier(nn.Module):
    def __init__(self, in_channels, rectified_params = ['mean', 'rotation', 'scale', 'opacity'], scale_range=[0.08, 0.32], scale_range_divider=3,
                 residual_properties=False, no_mean_combined=True, combined_properties=False, no_mean_residual=False, mean_range=None, use_offset=False, temporal_frame_ids=[0]):
        super().__init__()
        self.in_channels = in_channels
        self.no_mean_residual = no_mean_residual
        self.residual_properties = residual_properties
        self.combined_properties = combined_properties
        self.no_mean_combined = no_mean_combined
        assert not (residual_properties and combined_properties), "Cannot have both residual and combined properties."

        self.scale_min, self.scale_max = [0.05, 0.5]
        if residual_properties:
            if scale_range is not None:
                self.scale_range = [-(scale_range[1] - scale_range[0])/scale_range_divider, (scale_range[1] - scale_range[0])/scale_range_divider]
            else:
                self.scale_range = None
        else:
            self.scale_range = scale_range
        self.mean_range = mean_range
        self.rectified_params = rectified_params
        self.use_offset = use_offset
        self.temporal_frame_ids = temporal_frame_ids

        self.use_mean = 'mean' in rectified_params
        self.mean_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, 3)) if self.use_mean else None

        self.use_rot = 'rotation' in rectified_params
        self.quat_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, 4)) if self.use_rot else None
        
        self.use_scale = 'scale' in rectified_params
        if self.use_scale:
            scale_modules = [
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.LeakyReLU(),
                nn.Linear(in_channels, 3)]
            if self.scale_range is not None:
                scale_modules.append(nn.Sigmoid())
            self.scale_head = nn.Sequential(*scale_modules)
        else:
            self.scale_head = None
        
        self.use_opacity = 'opacity' in rectified_params
        if self.use_opacity:
            self.opacity_head = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.LeakyReLU(),
                nn.Linear(in_channels, 1))

        # self.init_weights() # seems to make it worse
                
    def init_weights(self):
        if self.use_mean:
            for m in self.mean_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
        if self.use_rot:
            for m in self.quat_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
        if self.use_scale:
            for m in self.scale_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
        if self.use_opacity:
            for m in self.opacity_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, means, quats, scale, opacity, feature):
        # Update Means
        if self.use_mean:
            mean_residual = self.mean_head(feature) if self.use_mean else None
            if self.mean_range is not None:
                mean_residual = self.mean_range[0] + (self.mean_range[1] - self.mean_range[0]) * torch.sigmoid(mean_residual)
            if self.residual_properties or (self.combined_properties and self.no_mean_combined):
                means = mean_residual if self.no_mean_residual else means + mean_residual
            elif self.combined_properties and not self.no_mean_combined:
                means = (means*.5 + mean_residual*.5)
        
        # Update rotation
        if self.use_rot:
            quats_residual = self.quat_head(feature) if self.use_rot else 0
            if self.residual_properties:
                quats = F.normalize(quats + quats_residual, dim=-1)
            elif self.combined_properties:
                quats = F.normalize((.5*quats + .5* quats_residual), dim=-1)
            else:
                quats = F.normalize(quats_residual, dim=-1)

        # Update scale
        if self.use_scale:
            scale_residual = self.scale_head(feature)
            if self.scale_range is not None:
                scale_residual = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * scale_residual
            if self.residual_properties:
                scale = torch.clamp(scale + scale_residual, self.scale_min, self.scale_max)
            elif self.combined_properties:
                scale = torch.clamp(.5*scale + .5*scale_residual, self.scale_min, self.scale_max)
            else:
                scale = scale_residual

        # Update opacity
        if self.use_opacity:
            opacity_residual = self.opacity_head(feature)
            if self.residual_properties:
                opacity = torch.clamp(opacity + opacity_residual, 0., 1.)
            elif self.combined_properties:
                opacity = torch.clamp(.5*opacity + .5*opacity_residual, 0., 1.)
            else:
                opacity = opacity_residual

        return means, quats, scale, opacity

@FEEDFORWARD_NETWORK.register_module()
class MeanRectifier(nn.Module):
    def __init__(self, in_channels, mean_range=None, num_layers=1):
        super().__init__()
        self.in_channels = in_channels
        self.mean_range = mean_range

        mean_layers = []
        for i in range(num_layers):
            mean_layers.append(nn.Linear(in_channels, in_channels))
            mean_layers.append(nn.LeakyReLU())
        mean_layers.append(nn.Linear(in_channels, 3))
        self.mean_head = nn.Sequential(*mean_layers)
        
    def forward(self, means, feature):
        mean_residual = self.mean_head(feature)
        if self.mean_range is not None:
            mean_residual = self.mean_range[0] + (self.mean_range[1] - self.mean_range[0]) * torch.sigmoid(mean_residual)
        means = means + mean_residual
        return means

@ATTENTION.register_module()
class InducedGaussianAttention(nn.Module):
    def __init__(self, in_channels=128, num_heads=8, dropout=0.1, n_inducing_points=500, pre_norm=False):
        super().__init__()
        self.pre_norm = pre_norm

        # Induced attention block
        self.inducing_points = nn.Parameter(torch.randn(n_inducing_points, in_channels))
        self.induced_attention = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout, batch_first=True)
        self.induced_ff = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, in_channels),
            nn.Dropout(dropout)
        )
        self.induced_ln1 = nn.LayerNorm(in_channels)
        self.induced_ln2 = nn.LayerNorm(in_channels)

        # Output attention block
        self.output_attention = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout, batch_first=True)
        self.output_ff = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, in_channels),
            nn.Dropout(dropout)
        )
        self.output_ln1 = nn.LayerNorm(in_channels)
        self.output_ln2 = nn.LayerNorm(in_channels)

    def forward(self, q, k):
        B = q.shape[0]
        inducing_points = self.inducing_points.unsqueeze(0).repeat(B, 1, 1)

        if self.pre_norm:
            q_pre = self.induced_ln1(inducing_points)
            z = self.induced_attention(q_pre, k, k)[0] + inducing_points
            z_pre = self.induced_ln2(z)
            induced = self.induced_ff(z_pre) + z

            out_pre = self.output_ln1(q)
            out_z = self.output_attention(out_pre, induced, induced)[0] + q
            out_z_pre = self.output_ln2(out_z)
            output = self.output_ff(out_z_pre) + out_z
        else:
            z = self.induced_ln1(self.induced_attention(inducing_points, k, k)[0] + inducing_points)
            induced = self.induced_ln2(self.induced_ff(z) + z)
            output = self.output_ln1(self.output_attention(q, induced, induced)[0] + q)
            output = self.output_ln2(self.output_ff(output) + output)

        return output
    
@ATTENTION.register_module()
class GaussianAttention(nn.Module):
    def __init__(self, in_channels=128, num_heads=8, dropout=0.1, pre_norm=False):
        super().__init__()
        self.attention = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, in_channels),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(in_channels)
        self.ln2 = nn.LayerNorm(in_channels)
        self.pre_norm = pre_norm
    
    def forward(self, q, k):
        if self.pre_norm:
            q_pre = self.ln1(q)
            z = self.attention(q_pre, k, k)[0] + q
            z_pre = self.ln2(z)
            output = self.ff(z_pre) + z
        else:
            z = self.ln1(self.attention(q, k, k)[0] + q)
            output = self.ln2(self.ff(z) + z)
        return output

@ATTENTION.register_module()
class GaussianImageCrossAttention(nn.Module): 
    def __init__(self, in_channels=128, num_heads=8, num_points=8, num_levels=3, dropout=0.1, 
                 use_deform_kernel=True, use_cam_embeds=True, use_level_embeds=True, num_cams=6, pre_norm=False):
        super().__init__()
        self.num_heads, self.num_levels, self.num_points = num_heads, num_levels, num_points
        self.in_channels = in_channels
        self.num_cams = num_cams
        self.im2col_step = 64

        self.sampling_offsets = nn.Linear(in_channels, self.num_heads * self.num_levels * self.num_points * 2)
        self.attention_weights = nn.Linear(in_channels, self.num_heads * self.num_levels * self.num_points)
        self.value_proj = nn.Linear(in_channels, in_channels)
        self.output_proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.in_channels)) if use_cam_embeds else None
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_levels, self.in_channels)) if use_level_embeds else None
        self.ln = nn.LayerNorm(in_channels, eps=1e-5)
        self.pre_norm = pre_norm
        self.use_deform_kernel = use_deform_kernel

        self.init_weights()

    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        kaiming_init(self.value_proj, mode='fan_in', distribution='uniform', bias=0.)
        kaiming_init(self.output_proj, mode='fan_in', distribution='uniform', bias=0.)
        if self.level_embeds is not None:
            normal_(self.level_embeds) 
        if self.cams_embeds is not None:
            normal_(self.cams_embeds) 
        self._is_init = True

    def project_reference_points(self, means, img_inputs, input_shape):
        H, W = input_shape
        cam2ego_lidar, intrins, post_rots, post_trans = img_inputs
        means_cam = (torch.inverse(cam2ego_lidar)[..., None, :, :] @ torch.cat((means, torch.ones((*means.shape[:2], 1), device=means.device)), dim=-1)[:, None, ..., None])[...,:3, :]
        means_img = (intrins[..., None, :, :] @ means_cam).squeeze(-1)
        means_img_normalized = torch.cat([means_img[..., :2] / (means_img[..., 2:] + 1e-4), means_img[..., 2:]], dim=-1) # add small eps to prevent division by zero
        means_img_normalized = (post_rots[..., None, :, :] @ means_img_normalized[..., None]).squeeze(-1) + post_trans[..., None, :]

        coor = means_img_normalized[..., :2]
        depth = means_img_normalized[..., 2]
        coor_normalized = coor / torch.tensor([W, H], device=means.device)
        mask = depth > 1e-2
        mask = mask & (coor_normalized[..., 0] > 0.) & (coor_normalized[..., 0]  < 1.) & (coor_normalized[..., 1] > 0.) & (coor_normalized[..., 1] < 1.)
        return coor_normalized, mask

    def forward(self, means, feature, img_feats, img_inputs, input_shape):
        B, N, _ = means.shape
        slots = torch.zeros_like(feature)
        identity = feature
        if self.pre_norm:
            feature = self.ln(feature)
        spatial_shapes = torch.tensor([(s.shape[-2], s.shape[-1]) for s in img_feats], device=means.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # Reference point projection
        ref_points, mask = self.project_reference_points(means, img_inputs, input_shape)

        # Do the rebatching operations
        max_len = 0
        indexes = []
        for j in range(B):
            indexes_batch = []
            for i, mask_per_img in enumerate(mask[j]):
                index_query_per_img = mask_per_img.nonzero()[:, 0]
                max_len = len(index_query_per_img) if len(index_query_per_img) > max_len else max_len
                indexes_batch.append(index_query_per_img)
            indexes.append(indexes_batch)

        sampling_offsets = self.sampling_offsets(feature).view(B, N, self.num_heads, self.num_levels, self.num_points, 2) # [bs, num_query, num_heads, num_levels, num_points, 2]
        queries_rebatch = feature.new_zeros(
            [B, self.num_cams, max_len, self.in_channels])
        reference_points_rebatch = ref_points.new_zeros(
            [B, self.num_cams, max_len, 2])
        sampling_offsets_rebatch = feature.new_zeros([B, self.num_cams, max_len, self.num_heads, self.num_levels, self.num_points, 2])

        for j in range(B):
            for i, reference_points_per_img in enumerate(ref_points[j]):   
                index_query_per_img = indexes[j][i]
                queries_rebatch[j, i, :len(index_query_per_img)] = feature[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[index_query_per_img]
                sampling_offsets_rebatch[j, i, :len(index_query_per_img)] = sampling_offsets[j, index_query_per_img]        

        # Then do the qkv projections and reshape into correct format (bs will be B*self.num_cams then!!)
        feat_flatten = []
        for lvl, feat in enumerate(img_feats):
            feat = feat.flatten(2).view(B, self.num_cams, self.in_channels, -1)
            if self.cams_embeds is not None:
                feat = feat + self.cams_embeds[None, ..., None].to(feat.dtype)
            if self.level_embeds is not None:
                feat = feat + self.level_embeds[None, lvl:lvl + 1, :, None].to(feat.dtype)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 3)
        value = feat_flatten.permute(0, 1, 3, 2).view(B*self.num_cams, -1, self.in_channels)
        value = self.value_proj(value).view(B*self.num_cams, -1, self.num_heads, self.in_channels//self.num_heads)
        
        attention_weights = self.attention_weights(queries_rebatch).view(B*self.num_cams, -1, self.num_heads, self.num_levels*self.num_points) # [bs, num_query, num_heads, num_points*1]
        attention_weights = attention_weights.softmax(-1).view(B*self.num_cams, -1, self.num_heads, self.num_levels, self.num_points)

        # Create sample positions
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        reference_points_rebatch = reference_points_rebatch.view(B*self.num_cams, -1, 2)
        reference_points_rebatch = reference_points_rebatch[:, :, None, None, None, :]
        sampling_offsets_rebatch = sampling_offsets_rebatch.view(B*self.num_cams, -1, self.num_heads, self.num_levels, self.num_points, 2)
        sampling_offsets_rebatch = sampling_offsets_rebatch / offset_normalizer[None, None, None, :, None, :]

        sampling_locations = reference_points_rebatch + sampling_offsets_rebatch

        # Apply deformable attention
        if torch.cuda.is_available() and value.is_cuda and self.use_deform_kernel:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step).view(B, self.num_cams, max_len, self.in_channels)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights).view(B, self.num_cams, max_len, self.in_channels)

        # Undo the rebatching
        for j in range(B):
            for i, index_query_per_img in enumerate(indexes[j]):
                slots[j, index_query_per_img] += output[j, i, :len(index_query_per_img)]

        count = mask.sum(1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]

        # output projection, dropout and skip connection
        slots = self.output_proj(slots)
        if self.pre_norm:
            return self.dropout(slots) + identity
        else:
            return  self.ln(self.dropout(slots) + identity)
               
class GaussianDecoderBlock(nn.Module):
    def __init__(self, channels_in,
                 temporal_att_cfg=None,
                 self_att_cfg=None,
                 cross_att_cfg=None,
                 hybrid_att_cfg=None,
                 ffn_cfg=None,
                 rect_cfg=dict(type='MeanRectifier', in_channels=128),
                 operation_order=None):
        super(GaussianDecoderBlock, self).__init__()
        self.channels_in = channels_in
        self.num_heads = channels_in // 8
        self.operation_order = operation_order

        # Gaussian-Image Cross-Attention
        self.cross_att = build_attention(cross_att_cfg) if cross_att_cfg is not None else None

        # Temporal Gaussian Self-Attention
        self.temporal_attn = build_attention(temporal_att_cfg) if temporal_att_cfg is not None else None

        # Gaussian Self-Attention
        self.self_attn = build_attention(self_att_cfg) if self_att_cfg is not None else None
        
        # Gaussian Hybrid-Attention
        self.hybrid_attn = build_attention(hybrid_att_cfg) if hybrid_att_cfg is not None else None

        # FFN
        self.ffn = build_feedforward_network(ffn_cfg) if ffn_cfg is not None else None

        # Refinement module
        self.gaussian_rectifier = build_feedforward_network(rect_cfg)

    def forward(self, means, feature, prev_feat, img_feats, img_inputs, input_shape):
        for operation in self.operation_order:
            if operation == 'temporal_att':
                feature = self.temporal_attn(feature, prev_feat)
            elif operation == 'self_att':
                feature = self.self_attn(feature, feature)
            elif operation == 'cross_att':
                feature = self.cross_att(means, feature, img_feats, img_inputs, input_shape)
            elif operation == 'ffn':
                feature = self.ffn(feature)
            elif operation == 'rect':
                means = self.gaussian_rectifier(means, feature)
            elif operation == 'hybrid_att':
                feature = self.hybrid_attn(feature, torch.cat((prev_feat, feature), dim=1))

        return means, feature

@FEEDFORWARD_NETWORK.register_module()
class GaussianDecoder(BaseModule):
    def __init__(self, in_channels, n_blocks_=4,
                 pos_enc_cfg=None,
                 temporal_att_cfg = None,
                 self_att_cfg = None,
                 cross_att_cfg = None,
                 hybrid_att_cfg=None,
                 ffn_cfg = None,
                 rect_cfg = dict(type='MeanRectifier', in_channels=128),
                 operation_order=None,
                 store_intermediate=False,
                 temporal_pos_enc=True):
        super(GaussianDecoder, self).__init__()
        self.feat_channels = in_channels
        self.store_intermediate = store_intermediate
        self.temporal_pos_enc = temporal_pos_enc
        assert operation_order is not None, 'Operation order must be defined'
        self.operation_order = operation_order
        self.blocks = nn.ModuleList(
            [GaussianDecoderBlock(in_channels, temporal_att_cfg=temporal_att_cfg,
                                  self_att_cfg=self_att_cfg, cross_att_cfg=cross_att_cfg, hybrid_att_cfg=hybrid_att_cfg,
                                   ffn_cfg=ffn_cfg, rect_cfg=rect_cfg, operation_order=operation_order) for _ in range(n_blocks_)]
        )
        self.pos_enc = build_positional_encoding(pos_enc_cfg) if pos_enc_cfg is not None else None
        self.use_temporal_att = temporal_att_cfg is not None or hybrid_att_cfg is not None

    def forward(self, means, feature, prev_feat, img_feats, img_inputs, input_shape):
        intermediate = []
        if self.use_temporal_att:
            if prev_feat is None:
                prev_means = means
                prev_features = feature
            else:
                prev_means, prev_features = prev_feat[..., :3], prev_feat[..., 3:]
            prev_feat = self.pos_enc(prev_means) + prev_features if self.temporal_pos_enc else prev_features

        # Blocks of Gaussian Decoder
        for block in self.blocks:
            if self.pos_enc is not None:
                feature = self.pos_enc(means) + feature
            means, feature = block(means, feature, prev_feat, img_feats, img_inputs, input_shape)
            if self.store_intermediate:
                intermediate.append([means, feature])

        return means, feature if not self.store_intermediate else intermediate