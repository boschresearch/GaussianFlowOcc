# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import torch

def move_gaussians_temporal_velocity(means, feature, velocity, dynamic_classes, temporal_frame_ids):
    """Function to move gaussian along time, with a single velocity per gaussian"""
    # Only move gaussians that are estimated as dynamic
    labels = torch.argmax(feature, dim=-1)+1
    dynamic_mask = ((labels.unsqueeze(-1) - dynamic_classes[None,None].to(labels.device)) == 0).any(dim=-1)

    #  Move means
    velocity = velocity.unsqueeze(-2) * temporal_frame_ids[None,None,:,None].to(velocity.device)
    velocity[~dynamic_mask] = 0 # set static velocities to 0
    means = means.unsqueeze(-2) + velocity
    return means.permute(0, 2, 1, 3) # B, T, N, 3

def move_gaussians_temporal_offset(means, feature, velocity, dynamic_classes, temporal_frame_ids):
    """Function to move gaussian along time, but with an explicit offset per target timestep"""
    labels = torch.argmax(feature, dim=-1)+1
    dynamic_mask = ((labels.unsqueeze(-1) - dynamic_classes[None,None].to(labels.device)) == 0).any(dim=-1)
    B, N, T3 = velocity.shape
    velocity = velocity.view(B, N, T3//3, 3)
    velocity[~dynamic_mask] = 0
    means = means.unsqueeze(-2).repeat(1, 1, len(temporal_frame_ids), 1)
    means[..., [i for i,c in enumerate(temporal_frame_ids) if c!=0], :] = means[..., [i for i,c in enumerate(temporal_frame_ids) if c!=0], :] + velocity
    return means.permute(0, 2, 1, 3) # B, T, N, 3

def move_gaussians_temporal_module(means, logits, offsets, dynamic_classes):
    """Function to move gaussian along time, but with an explicit offset per target timestep"""
    labels = torch.argmax(logits, dim=-1)+1
    dynamic_mask = ((labels.unsqueeze(-1) == dynamic_classes[None,None].to(labels.device))).any(dim=-1)
    B, N, T, _ = offsets.shape
    offsets[~dynamic_mask] = 0
    means_moved = means.clone().unsqueeze(-2).repeat(1, 1, T, 1)
    means_moved = means_moved + offsets
    means = torch.cat([means_moved, means.unsqueeze(-2)], dim=-2).permute(0, 2, 1, 3) # B, T, N, 3
    return means