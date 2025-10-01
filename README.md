<div align="center">

# [GaussianFlowOcc](): Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow

[Simon Boeder](https://scholar.google.de/citations?user=eEsebSgAAAAJ)<sup>1,2</sup>, [Fabian Gigengack](https://scholar.google.de/citations?user=_Zk9J1MAAAAJ)<sup>1</sup>, [Benjamin Risse](https://scholar.google.de/citations?user=rWx-1t0AAAAJ)<sup>2</sup><br>
<sup>1</sup>Bosch Research, <sup>2</sup>University of Münster

[**ICCV 2025**]()

[![arXiv](https://img.shields.io/badge/arXiv-2502.17288-red)](https://arxiv.org/abs/2502.17288)
</div>

![overview](assets/Overview.png)

Visualization videos can be found at `assets/`.

## Installation

### 1. Create virtual env
```shell script
conda create -n gaussianflowocc python=3.8 -y
conda activate gaussianflowocc
```

### 2. Install Repository
Please make sure to have CUDA 11.3 installed and in your PATH.

```shell script
# install pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# install openmim, used for installing mmcv
pip install -U openmim

# install mmcv
mim install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# install mmdet and ninja
pip install mmdet==2.25.1 ninja==1.11.1

# Install GaussianFlowOcc (as mmdet3d fork)
pip install -v -e .

# Install gsplat
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.2.0

# install GroundedSAM (for pseudo labels)
python -m pip install -e groundedsam/segment_anything
python -m pip install -e groundedsam/GroundingDINO
pip install diffusers transformers accelerate scipy safetensors
```

## Data Preparation
1. Please create a directory `./data` and `./ckpts` in the root directory of the repository.

2. Download nuScenes [https://www.nuscenes.org/download].

3. Download the Occ3D-nuScenes dataset from [https://github.com/Tsinghua-MARS-Lab/Occ3D]. The download link can be found in their README.md.

4. Generate the annotation files.  This will put the annotation files into the `./data` directory by default. The process can take up to ~1h.
```shell script
python tools/create_data_bevdet.py
```
5. Copy or softlink the files into the `./data` directory. The structure of the data directory should be as follows:

```shell script
gaussianflowocc
    ├──data
    │   ├── nuscenes
    │   │  ├── v1.0-trainval (Step 2, nuScenes+nuScenes-panoptic files)
    │   │  ├── sweeps (Step 2, nuScenes files)
    │   │  ├── samples (Step 2, nuScenes files)
    │   │  └── panoptic (Step 2, nuScenes-panoptic files)
    │   ├── gts (Step 3)
    │   ├── bevdetv2-nuscenes_infos_train.pkl (Step 4)
    │   ├── bevdetv2-nuscenes_infos_val.pkl (Step 4)
    │   ├── bevdetv2-nuscenes_infos_test.pkl (Step 4)
    │   ├── metric_3d_nusc (See next chapter)
    │   └── groundedsam (See next chapter)
    ├──ckpts
    └──...
```

## Generate Pseudo-Labels
Please create two directories `metric_3d_nusc` and `groundedsam` in a location with enough disk space and softlink them into `./data`, as the following scripts will write data to these locations (the `./data` directory should look like in the tree above).
### 1. Pseudo Depth
First, we generate the pseudo depth labels using [Metric3D](https://github.com/YvanYin/Metric3D) (~550 GB).
```shell script
python tools/generate_m3d_nusc.py
```

You can parallelize this by starting multiple runs and specify a scene range for each run.
Example:
```shell script
python tools/generate_m3d_nusc.py --scene-prefix scene-00 scene-01 scene-02
python tools/generate_m3d_nusc.py --scene-prefix scene-03 scene-04 scene-05
python tools/generate_m3d_nusc.py --scene-prefix scene-06 scene-07 scene-08
python tools/generate_m3d_nusc.py --scene-prefix scene-09 scene-10 scene-11
```

### 2. Pseudo Semantics
Next, we generate the pseudo semantic labels using [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) (~276 GB). 

Download the SAM checkpoint
```shell script
# Download SAM checkpoint into ./ckpts
cd ckpts
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 
cd ..
```

Run the generation script:
```shell script
# single GPU
python3 groundedsam/generate_grounded_sam.py --single-gpu
# multi GPU with 4 GPU's
python -m torch.distributed.launch --nproc_per_node 4 --master_port 29582 groundedsam/generate_grounded_sam.py
```

As for the pseudo depth labels, you can run multiple generation scripts simultaneously and restrict each run to a certain range of scenes by using the `--scene-prefixes` argument.  
If you would like to generate the masks also for the validation set, use the `--split val` argument.

## Train model
We provide configuration files for training our model with or without pseudo depth labels.
```shell
# In the root directory of the repository:
# single gpu
python tools/train.py configs/gaussianflowocc.py
# multiple gpu (replace "num_gpu" with the number of available GPUs) - 4 GPU's are reccomended.
./tools/dist_train.sh configs/gaussianflowocc.py num_gpu
```
In our experiments, we use 4 GPU's.
Due to some non-deterministic operations, the results may deviate slightly (up or down) from the results presented in the paper.

## Evaluate model
After training, you can evaluate the model on Occ3D-nuScenes.
```shell
# mIoU & IoU on Occ3D-nuScenes
python tools/test.py configs/gaussianflowocc.py work_dirs/gaussianflowocc/epoch_18_ema.pth --eval mIoU 
```

If you want to evaluate the RayIoU metric, please first run the standard evaluation with the extra flag `--save-occ-path` to store the predictions.
Afterwards, we can run the RayIoU eval script.
```shell
# Store predictions
python tools/test.py configs/gaussianflowocc.py work_dirs/gaussianflowocc/epoch_18_ema.pth --eval mIoU --save-occ-path ./occ/gaussianflowocc

# Run RayIoU eval
python tools/eval_ray_mIoU.py --pred-dir ./occ/gaussianflowocc
```

You can also increase the influence range of each Gaussian during voxelization to potentially increase the accuracy in favor of runtime performance using the `--nbh` parameter. By default, `--nbh` is set to 4.

```shell
# mIoU & IoU on Occ3D-nuScenes with max_neighborhood of 5
python tools/test.py configs/gaussianflowocc.py work_dirs/gaussianflowocc/epoch_18_ema.pth --eval mIoU --nbh 5
```

## Resume Runs
If the training is interrupted at any point and you want to resume from a checkpoint, you can simply use the `--resume-from` command as follows:
``` shell
./tools/dist_train.sh configs/gaussianflowocc.py num_gpu --resume-from /path/to/checkpoint/latest.pth
```
The checkpoints are usually saved under the `work_dirs` directory. By default, a checkpoint is created every 6 epochs.

## Copyright
Copyright (c) 2022 Robert Bosch GmbH
SPDX-License-Identifier: AGPL-3.0