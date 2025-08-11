# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import mmcv
import torch
import torch.nn.functional as F

def cal_depth_metric(depth_pred, depth_gt):
    """Inspired by https://github.com/huang-yh/SelfOcc/blob/1658c2afd464e30408ee61925da55d27538427e6/utils/metric_util.py#L265.
    Published with Apache-2.0 License.
    """
    depth_pred = torch.clamp(depth_pred, 1e-3, 80)
    total_num = len(depth_gt)

    thresh = torch.maximum((depth_gt / depth_pred), (depth_pred / depth_gt))
    a1 = (thresh < 1.25).to(torch.float).sum()
    a2 = (thresh < 1.25 ** 2).to(torch.float).sum()
    a3 = (thresh < 1.25 ** 3).to(torch.float).sum()

    se = ((depth_gt - depth_pred) ** 2).sum()

    se_log = ((torch.log(depth_gt) - torch.log(depth_pred)) ** 2).sum()

    abs_rel = (torch.abs(depth_gt - depth_pred) / depth_gt).sum()

    sq_rel = (((depth_gt - depth_pred) ** 2) / depth_gt).sum()

    return abs_rel, sq_rel, se, se_log, a1, a2, a3, total_num

def eval_depth(model, data_loader):
    model.eval()
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    abs_rels, sq_rels, ses, se_logs, a1s, a2s, a3s, total_nums = 0, 0, 0, 0, 0, 0, 0, 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, render_preds=True, **data)
        batch_size = len(result)
        assert batch_size == 1
        # Only extract the depth prediction (sample from lidar)
        pred_depth = result[0]['rendered_depths']
        h, w = pred_depth.shape[-2:]
        gs_gts = data['gs_gts_pixel'][0].data[0][0]
        cam_indices = gs_gts[:, 1].long()
        coors = gs_gts[:, 2:4]
        depths = gs_gts[:, 4]
        # labels = gs_gts[:, 5].long() # TODO: We can also compute the lidar ray IoU for labels!

        # Sample the depth prediction
        n_cams = data['gs_intrins'][0].data[0].shape[-3]
        all_depths_pred = torch.tensor([])
        all_depths_gt = torch.tensor([])
        for n in range(n_cams):
            mask = cam_indices == n
            coors_cam = coors[mask]
            grid = coors_cam[:, [1, 0]] / torch.tensor([w, h]) * 2 - 1
            gt_pred_cam = depths[mask]
            pred_depth_cam = pred_depth[n]
            pred_depth_sampled = F.grid_sample(
                torch.tensor(pred_depth_cam[None, None, :, :]),
                grid[None, None, ...],
                mode='bilinear',
                align_corners=False,
            ).squeeze()
            all_depths_pred = torch.cat((all_depths_pred, pred_depth_sampled))
            all_depths_gt = torch.cat((all_depths_gt, gt_pred_cam))
        
        # Compute the metrics but without averaging
        abs_rel, sq_rel, se, se_log, a1, a2, a3, total_num = cal_depth_metric(all_depths_pred, all_depths_gt)
        abs_rels += abs_rel
        sq_rels += sq_rel
        ses += se
        se_logs += se_log
        a1s += a1
        a2s += a2
        a3s += a3
        total_nums += total_num

        for _ in range(batch_size):
            prog_bar.update()

    # Compute the average metrics
    abs_rel = abs_rels / total_nums
    sq_rel = sq_rels / total_nums
    rmse = (ses / total_nums) ** .5
    rmse_log = (se_logs / total_nums) ** .5
    a1 = a1s / total_nums
    a2 = a2s / total_nums
    a3 = a3s / total_nums

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def eval_depth_average(model, data_loader):
    model.eval()
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    abs_rels, sq_rels, rmses, rmse_logs, a1s, a2s, a3s, total_count = 0, 0, 0, 0, 0, 0, 0, 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, render_preds=True, **data)
        batch_size = len(result)
        assert batch_size == 1
        # Only extract the depth prediction (sample from lidar)
        pred_depth = result[0]['rendered_depths']
        h, w = pred_depth.shape[-2:]
        gs_gts = data['gs_gts_pixel'][0].data[0][0]
        cam_indices = gs_gts[:, 1].long()
        coors = gs_gts[:, 2:4]
        depths = gs_gts[:, 4]
        # labels = gs_gts[:, 5].long() # TODO: We can also compute the lidar ray IoU for labels!

        # Sample the depth prediction
        n_cams = data['gs_intrins'][0].data[0].shape[-3]
        all_depths_pred = torch.tensor([])
        all_depths_gt = torch.tensor([])
        for n in range(n_cams):
            mask = cam_indices == n
            coors_cam = coors[mask]
            grid = coors_cam[:, [1, 0]] / torch.tensor([w, h]) * 2 - 1
            gt_pred_cam = depths[mask]
            pred_depth_cam = pred_depth[n]
            pred_depth_sampled = F.grid_sample(
                torch.tensor(pred_depth_cam[None, None, :, :]),
                grid[None, None, ...],
                mode='bilinear',
                align_corners=True,
            ).squeeze()
            all_depths_pred = torch.cat((all_depths_pred, pred_depth_sampled))
            all_depths_gt = torch.cat((all_depths_gt, gt_pred_cam))
        
        # Compute the metrics but without averaging
        abs_rel, sq_rel, se, se_log, a1, a2, a3, total_num = cal_depth_metric(all_depths_pred, all_depths_gt)
        abs_rels += (abs_rel / total_num)
        sq_rels += (sq_rel / total_num)
        rmses += ((se / total_num) ** .5)
        rmse_logs += ((se_log / total_num) ** .5)
        a1s += (a1 / total_num)
        a2s += (a2 / total_num)
        a3s += (a3 / total_num)
        total_count += 1

        for _ in range(batch_size):
            prog_bar.update()

    # Compute the average metrics
    abs_rel = abs_rels / total_count
    sq_rel = sq_rels / total_count
    rmse = rmses / total_count
    rmse_log = rmse_logs / total_count
    a1 = a1s / total_count
    a2 = a2s / total_count
    a3 = a3s / total_count

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3