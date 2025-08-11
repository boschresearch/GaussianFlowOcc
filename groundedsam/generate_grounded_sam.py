# Copyright (c) 2022 Robert Bosch GmbH

# SPDX-License-Identifier: AGPL-3.0
# This code is adapted from https://github.com/JunchengYan/GroundedSAM_OccNeRF under Apache-2.0 License

import os
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Grounding DINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import load_image, predict

# segment anything
from segment_anything import build_sam, SamPredictor 
import numpy as np

# diffusers
import torch
from huggingface_hub import hf_hub_download
import pickle
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

######### GS-Occ vocabulary
voc_classes = ["background", "car", "truck", "trailer", "bus", "construction_vehicle", "bicycle", "motorcycle", "pedestrian",
                "traffic_cone", "barrier", "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation"]
class_to_nusc_v1_map = torch.tensor([16, 9, 5, 3, 0, 4, 6, 7, 8, 2, 1, 10, 11, 12, 13, 14, 15])
nusc_v2_to_v1 = torch.tensor([0, 4, 10, 9, 3, 5, 2, 6, 7, 8, 1, 11, 12, 13, 14, 15, 16])

def class_mapping(voc):
    return [x for xs in [[voc_classes.index(c)]*len(t) for c, t in voc.items()] for x in xs]

def phrase_mapping(voc):
    return [c for c, t in voc.items() for _ in t]

def flattened(voc, for_separate=False):
    if for_separate:
        return [x.replace("-", " ") for v in voc.values() for x in v]
    else:
        return [x for v in voc.values() for x in v]

def combine_for_dino(voc):
    combined = " . ".join(flattened(voc))
    return combined

vocabulary = {
    'background': ['sky'],
    'car': ['car', 'vehicle', 'sedan', 'van', 'jeep'],
    'truck': ['lorry', 'truck', 'tractor'],
    'trailer': ['trailer'],
    'bus': ['bus'],
    'construction_vehicle': ['excavator', 'crane'],
    'bicycle': ['bicycle'],
    'motorcycle': ['motorcycle', 'scooter'],
    'pedestrian': ['person', 'pedestrian'],
    'traffic_cone': ['traffic-cone'],
    'barrier': ['barricade',  'barrier'],
    'driveable_surface': ['highway', 'street'],
    'other_flat': ['traffic-island', 'delimiter'],
    'sidewalk': ['sidewalk', 'walkway'],
    'terrain': ['grass', 'sand', 'gravel', 'terrain'],
    'manmade': ['building', 'wall', 'fence', 'pole', 'sign', 'light', 'bridge', 'billboard'],
    'vegetation': ['bush', 'plants', 'tree']
}
class_map = class_mapping(vocabulary)
phrase_map = phrase_mapping(vocabulary)

# color map in nusc_v1!
color_map = np.array(
        [[0, 0, 0, 255],
        [255, 120, 50, 255],  # barrier orangey
        [255, 192, 203, 255],  # bicycle pink
        [255, 255, 0, 255],  # bus yellow
        [0, 150, 245, 255],  # car blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle dark orange
        [255, 0, 0, 255],  # pedestrian red
        [255, 240, 150, 255],  # traffic_cone light yellow
        [135, 60, 0, 255],  # trailer brown
        [160, 32, 240, 255],  # truck purple
        [255, 0, 255, 255],  # driveable_surface dark pink
        [139, 137, 137, 255],  # other_flat dark red
        [75, 0, 75, 255],  # sidewalk dark purple
        [150, 240, 80, 255],  # terrain light green
        [230, 230, 250, 255],  # manmade white
        [0, 175, 0, 255],  # vegetation green
        [0, 255, 127, 255],  # ego car dark cyan
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ], dtype=np.uint8)
#########


BOX_TRESHOLD = 0.20
TEXT_TRESHOLD = 0.20

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

def show_mask(mask, image, phrases, logits, random_color=True, separate=False, visual=False):
    h, w = mask.shape[-2:]

    mult_mask = mask.squeeze(1) * logits[:, None, None].to(device)
    mult_mask = torch.cat((torch.zeros((1, h, w)).to(device), mult_mask), dim=0)
    idx_mask = mult_mask.argmax(dim=0)

    phrase_class = torch.tensor([0]+ [class_map[flattened(vocabulary, for_separate=separate).index(p)] for p in phrases], device=device)
    class_mask = phrase_class[idx_mask]
    v1_class_mask = nusc_v2_to_v1[class_mask]
    
    if visual:
        mask_image = color_map[v1_class_mask.cpu()][..., :3]
        image_overlay = (image * 0.5 + mask_image * 0.5).astype(np.uint8)
        return image_overlay, v1_class_mask
    else:
        return v1_class_mask
    
def generate_semantic_combined(groundingdino_model, sam_predictor, local_rank, image_filename=None, device='cpu', save_name=None, visual=False):
    image_source, image = load_image(image_filename)
    image = image.to(device)

    prompt = combine_for_dino(vocabulary)
    all_prompts = flattened(vocabulary)
    boxes, logits, phrases = predict(
        model=groundingdino_model.module, 
        image=image, 
        caption=prompt, 
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        only_max_prompt=True,
        device='cuda')
                
    valid_indices = [index for index in range(len(phrases)) if phrases[index] in all_prompts]
    
    boxes = boxes[valid_indices]
    logits = logits[valid_indices]
    phrases = [phrases[index] for index in valid_indices]
    
    sam_predictor.set_image(image_source)
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
    
    if transformed_boxes.shape[0] == 0 or transformed_boxes == None:
        print("No box detected!")
        return np.ones((H, W)) * (-1)
    logits_masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                return_logits=True,
                multimask_output = False,
            )
    masks = logits_masks > sam_predictor.model.mask_threshold
    
    if visual:
        frame_with_mask, mask_np = show_mask(masks, image_source, phrases, logits, visual=visual)
        Image.fromarray(frame_with_mask).save('./tmp/{}_mask.png'.format(save_name))
    else:
        mask_np = show_mask(masks, image_source, phrases, logits, visual=visual)

    return mask_np

def generate_semantic(groundingdino_model, sam_predictor, local_rank, image_filename=None, device='cpu', save_name=None, resize=[800], visual=False):
    image_source, image = load_image(image_filename, random_resize=resize)
    image = image.to(device)
    boxes_list, logits_list, phrases_list = [], [], []
    prompts = flattened(vocabulary, for_separate=True)
    for prompt in prompts:
        boxes, logits, phrases = predict(
            model=groundingdino_model.module, 
            image=image, 
            caption=prompt, 
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device='cuda')
        
        boxes_list.append(boxes)
        logits_list.append(logits)
        phrases_list += phrases

    boxes = torch.cat(boxes_list, dim=0)
    logits = torch.cat(logits_list, dim=0)
    phrases = phrases_list

    valid_indices = [index for index in range(len(phrases)) if phrases[index] in prompts]
    
    boxes = boxes[valid_indices]
    logits = logits[valid_indices]
    phrases = [phrases[index] for index in valid_indices]
        
    sam_predictor.set_image(image_source)
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
    
    if transformed_boxes.shape[0] == 0 or transformed_boxes == None:
        print("No box detected!")
        return np.ones((H, W)) * (-1)
    logits_masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                return_logits=True,
                multimask_output = False,
            )
    masks = logits_masks > sam_predictor.model.mask_threshold
    
    if visual:
        frame_with_mask, mask_np = show_mask(masks, image_source, phrases, logits, visual=visual, separate=True)
        Image.fromarray(frame_with_mask).save('./tmp/{}_mask.png'.format(save_name))
    else:
        mask_np = show_mask(masks, image_source, phrases, logits, visual=visual, separate=True)

    return mask_np

class NuscenesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--version", default='v1.0-trainval', type=str)
    parser.add_argument("--method", choices=['separate', 'combined'], default='separate')
    parser.add_argument("--split", choices=['train', 'val', 'test'], default='train')
    parser.add_argument("--visual", action='store_true')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--single-gpu", action='store_true')
    parser.add_argument("--max-size", type=int, default=800)
    parser.add_argument("--scene-prefixes", nargs='+', default=None)
    args = parser.parse_args()

    with torch.no_grad():
        torch.cuda.set_device(args.local_rank)
        if args.single_gpu:
            dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=1, rank=0) 
        else:
            dist.init_process_group(backend='nccl') 
        device = torch.device("cuda", args.local_rank)

        data_path = 'data/nuscenes'
        version = args.version
        split = args.split
        prefix = '-mini' if version == 'v1.0-mini' else ''
        
        with open(f'data/bevdetv2-nuscenes{prefix}_infos_{split}.pkl', 'rb') as f:
            nusc_data = pickle.load(f)['infos']
        
        nusc_dataset = NuscenesDataset(list(range(len(nusc_data))))
        train_sampler = DistributedSampler(nusc_dataset, shuffle=False)
        trainloader = DataLoader(nusc_dataset, batch_size=1, num_workers=4, sampler=train_sampler)

        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        
        groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
        groundingdino_model = groundingdino_model.to(device)
        groundingdino_model = DDP(groundingdino_model, device_ids=[args.local_rank], output_device=args.local_rank)
        
        sam_checkpoint = 'ckpts/sam_vit_h_4b8939.pth'
        sam = build_sam(checkpoint=sam_checkpoint)
        sam = sam.to(device)
                
        sam = DDP(sam, device_ids=[args.local_rank], output_device=args.local_rank)

        sam_predictor = SamPredictor(sam.module)
        
        save_path = 'data/grounded_sam_nusc'
        camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
                
        print('Starting to generate nuscenes semantic masks using groundingdino and sam')
        if args.scene_prefixes is not None:
            print(f'Generating for prefixes: {args.scene_prefixes}')

        for index_data in tqdm(trainloader):

            sample = nusc_data[index_data]
            token = sample['token']
            scene = sample['scene_name']
            all_masks = []

            if args.scene_prefixes is not None:
                if not any([scene.startswith(prefix) for prefix in args.scene_prefixes]):
                    continue

            if not os.path.exists(os.path.join(save_path, scene)):
                os.makedirs(os.path.join(save_path, scene), exist_ok=True) 

            if os.path.exists(os.path.join(save_path, scene, f'{token}.npy')):
                if not args.overwrite:
                    continue

            for cam in camera_names:
                camera_sample = sample['cams'][cam]

                # load image
                image_filename = camera_sample['data_path']
                if args.method == 'separate':
                    mask = generate_semantic(groundingdino_model, sam_predictor, args.local_rank, image_filename=image_filename, device=device, save_name=f'{cam}_{token}', visual=args.visual, resize=[args.max_size])
                elif args.method == 'combined':
                    mask = generate_semantic_combined(groundingdino_model, sam_predictor, args.local_rank, image_filename=image_filename, device=device, save_name=f'{cam}_{token}', visual=args.visual)

                mask = mask.numpy().astype(np.int8) # 900, 1600
                all_masks.append(mask)
            
            all_masks = np.stack(all_masks)
            np.save(os.path.join(save_path, scene, f'{token}.npy'), all_masks)