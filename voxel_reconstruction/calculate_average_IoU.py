import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data

from utils.binvox_visualization import get_volume_views

import utils.data_transforms
import utils.data_loaders 

from utils.binvox_rw import read_as_3d_array

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

from config import cfg

from dataset import ShapeNetDataset
from dataset_utils import image_to_numpy, imagenet_deprocess 

import argparse
import logging
from detectron2.utils.logger import setup_logger

import cv2


logger = logging.getLogger("demo")


class Quantitative_analysis_demo():
    def __init__(self, cfg, output_dir):
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.refiner = Refiner(cfg)
        self.merger = Merger(cfg)
#         self.thresh = cfg.VOXEL_THRESH
        self.th = cfg.TEST.VOXEL_THRESH
        
        checkpoint = torch.load(cfg.CHECKPOINT)
        encoder_state_dict = clean_state_dict(checkpoint['encoder_state_dict'])
        self.encoder.load_state_dict(encoder_state_dict)
        decoder_state_dict = clean_state_dict(checkpoint['decoder_state_dict'])
        self.decoder.load_state_dict(decoder_state_dict)
        if cfg.NETWORK.USE_REFINER:
            refiner_state_dict = clean_state_dict(checkpoint['refiner_state_dict'])
            self.refiner.load_state_dict(refiner_state_dict)
        if cfg.NETWORK.USE_MERGER:
            merger_state_dict = clean_state_dict(checkpoint['merger_state_dict'])
            self.merger.load_state_dict(merger_state_dict)
        
        self.output_dir = output_dir
        

    def calculate_iou(self,imgs,GT_voxels,sid, mid, iid, sampled_idx):
        dir1 = os.path.join(self.output_dir,str(sid),str(mid))
        if not os.path.exists(dir1):
            os.makedirs(dir1)
        
        image_features = self.encoder(imgs)
        raw_features, generated_volume = self.decoder(image_features)
        generated_volume = self.merger(raw_features, generated_volume)
        generated_volume = self.refiner(generated_volume)
        generated_volume = generated_volume.squeeze()
        
        sample_iou = []
        for th in self.th:
            _volume = torch.ge(generated_volume, th).float()
            intersection = torch.sum(_volume.mul(GT_voxels)).float()
            union = torch.sum(torch.ge(_volume.add(GT_voxels), 1)).float()
            sample_iou.append((intersection / union).item())
        return sample_iou

        
       
        
        

def clean_state_dict(state_dict):
    out = {}
    for k, v in state_dict.items():
        while k.startswith("module."):
            k = k[7:]
        out[k] = v
    return out

def get_parser():
    parser = argparse.ArgumentParser(description="Pix2Vox Demo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data_dir")
    parser.add_argument("--synset_id", default="04530566")
    parser.add_argument("--output_dir")
    parser.add_argument("--next_best_view", default=None)
    parser.add_argument("--voxel_dir")
    parser.add_argument("--sample_size")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = setup_logger(name="demo")
    logger.info("Arguments: " + str(args))
    
    cfg.CHECKPOINT = args.checkpoint
                     
    data_dir = args.data_dir
    synset_id = args.synset_id
    output_dir = args.output_dir
    
    dataset = ShapeNetDataset(cfg, data_dir)
    synset_ids = dataset.synset_ids
    first_idx = synset_ids.index(synset_id)
    model_num = int(synset_ids.count(synset_id)/24)
    
    sample_size = int(args.sample_size)
    idx_list = torch.randint(0, model_num, size=(sample_size,))
    idx_list += first_idx
    
    all_iou = []
    for idx in idx_list:
        item = dataset[idx] # img, verts, faces, points, normals, voxels, P, _imgs, render_RTs, RT, sid, mid, iid
        img = item[0].squeeze()
        imgs = item[7]
        if args.next_best_view:
            sampled_idx = int(args.next_best_view)
        else:
            sampled_idx = torch.randint(0, 23, size=(1,)).item()
        sampled_img = imgs[sampled_idx]
        imgs = torch.stack([img,sampled_img])
        imgs = imgs.unsqueeze(0) # (1,2,3,224,224)
        sid = item[-3]
        mid = item[-2]
        iid = 24
#         print(sid,mid)
        voxel_dir = args.voxel_dir
        voxel_path = os.path.join(voxel_dir,sid,mid,'model.binvox')
        with open(voxel_path, 'rb') as f:
            volume = utils.binvox_rw.read_as_3d_array(f)
            GT_voxels = volume.data.astype(np.float32)
        GT_voxels = torch.tensor(GT_voxels)
        demo = Quantitative_analysis_demo(cfg, output_dir=output_dir)
        calculated_iou = demo.calculate_iou(imgs, GT_voxels, sid, mid, iid, sampled_idx)
        all_iou.append(calculated_iou)
    all_iou = torch.tensor(all_iou)
    sum_iou = torch.sum(all_iou, 0)
    avg_iou = sum_iou/sample_size
    print(avg_iou)

    
    

