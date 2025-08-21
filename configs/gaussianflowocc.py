# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

_base_ = ['./_base_/datasets/nus-3d.py', './_base_/default_runtime.py']

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

pc_range = [-40., -40., -1.0, 40., 40., 5.4]
eval_threshold_range=[.05]

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
gt_root = 'data/gts'
mask_gt_root = 'data/grounded_sam_nusc'
depth_gt_root = 'data/metric_3d_nusc'
file_client_args = dict(backend='disk')

# Other settings
batch_size = 4

# rasterizer settings
raster_downscale_factor = .44
raster_crop_top = 140
raster_shape = (int(data_config['src_size'][0] * raster_downscale_factor - raster_crop_top), 
                int(data_config['src_size'][1] * raster_downscale_factor))

hidden_dim = 256
multi_adj_frame_id_cfg = (1, 6, 1)

T = 6
temporal_frame_ids = list(range(-T,T+1,1))
num_frames=6

model = dict(
    type='GaussianFlowOcc',
    eval_threshold_range=eval_threshold_range,
    voxel_grid_cfg=grid_config,
    gaussian_init_scale=4,
    in_channels=hidden_dim,
    temporal_frame_ids=temporal_frame_ids,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=hidden_dim,
        start_level=0,
        num_outs=3),
    gaussian_decoder=dict(
        type='GaussianDecoder',
        in_channels=hidden_dim,
        n_blocks_=3,
        rect_cfg=dict(type='MeanRectifier', in_channels=hidden_dim),
        pos_enc_cfg=dict(type='PointPositionalEncoding', out_channels=hidden_dim),
        temporal_att_cfg = dict(type='InducedGaussianAttention', in_channels=hidden_dim),
        self_att_cfg = dict(type='InducedGaussianAttention', in_channels=hidden_dim),
        cross_att_cfg = dict(type='GaussianImageCrossAttention', in_channels=hidden_dim),
        operation_order=('temporal_att', 'self_att', 'cross_att', 'rect')
    ),
    rasterizer=dict(
        type='GaussianFlowOccRasterizer',
        raster_shape=raster_shape,
    ),
    temporal_module=dict(
        type='TemporalFeedForwardNetwork',
        num_timesteps=len(temporal_frame_ids),
        in_channels=hidden_dim,
        num_layers=3,
    )
)

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.,
    flip_dy_ratio=0.)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='GaussianFlowOcc_GeneratePseudoLabelsHorizon', downscale_factor=raster_downscale_factor, crop_top=raster_crop_top, num_frames=num_frames,
                            grounded_sam_root=mask_gt_root, depth_root=depth_gt_root, temporal_frame_ids=temporal_frame_ids),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gs_gts', 'gs_intrins', 'gs_extrins'])
]

val_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img_inputs'])
        ])
]

test_pipeline = val_pipeline

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    temporal_frame_ids=temporal_frame_ids,
    eval_threshold_range=eval_threshold_range,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file='data/bevdetv2-nuscenes_infos_val.pkl',
    gt_root=gt_root)

val_data_config = dict(
    pipeline=val_pipeline,
    ann_file='data/bevdetv2-nuscenes_infos_val.pkl',
    gt_root=gt_root)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file='data/bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        gt_root=gt_root,
        box_type_3d='LiDAR'),
    val=val_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='CustomCosineAnealing',
    start_at=9,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    min_lr_ratio=1e-2
)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
runner = dict(type='EpochBasedRunner', max_epochs=18)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=6
    ),
]
checkpoint_config=dict(interval=6)