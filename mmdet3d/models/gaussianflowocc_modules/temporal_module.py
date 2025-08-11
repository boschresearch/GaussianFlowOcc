# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from mmcv.runner.base_module import BaseModule
import numpy as np

@FEEDFORWARD_NETWORK.register_module()
class TemporalFeedForwardNetwork(BaseModule):
    def __init__(self, num_timesteps, in_channels, num_layers):
        super(TemporalFeedForwardNetwork, self).__init__()
        self.in_channels = in_channels

        # Time embedding layer
        self.time_embed = nn.Embedding(num_timesteps, in_channels)

        # Gaussian feedforward network
        layers = nn.ModuleList([nn.Linear(in_channels*2, in_channels*3), nn.LeakyReLU()])
        for _ in range(num_layers):
            layers.append(nn.Linear(in_channels*3, in_channels*3))
            layers.append(nn.LeakyReLU())
        
        layers.append(nn.Linear(in_channels*3, 3))

        self.ffn = nn.Sequential(*layers)

    def forward(self, gaussians, target_timestep):
        # Get the time embedding
        B = gaussians.size(0)
        T = target_timestep[0].shape[0]
        time_indices = np.concatenate(target_timestep)
        time_embeds = self.time_embed.weight[time_indices].view(B, 1, T, self.in_channels).repeat(1, gaussians.size(1), 1, 1)

        # Concat the time embedding to the gaussians
        gaussian_features = torch.cat([gaussians[..., None, :].repeat(1, 1, T, 1), time_embeds], dim=-1)

        # Apply the feedforward network
        offsets = self.ffn(gaussian_features)

        return offsets