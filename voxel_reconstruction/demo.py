import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data

from utils.binvox_visualization import get_volume_views

import utils.data_transforms
import utils.data_loaders 

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


class Visualization_demo():
    def __init__(self, cfg, output_dir):
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.refiner = Refiner(cfg)
        self.merger = Merger(cfg)
        
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
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir
        

    def run_on_images(self,imgs, sid, mid, iid, sampled_idx):
        dir1 = os.path.join(output_dir,str(sid),str(mid))
        if not os.path.exists(dir1):
            os.makedirs(dir1)
           
        deprocess = imagenet_deprocess(rescale_image=False)
        image_features = self.encoder(imgs)
        raw_features, generated_volume = self.decoder(image_features)
        generated_volume = self.merger(raw_features, generated_volume)
        generated_volume = self.refiner(generated_volume)
        img = image_to_numpy(deprocess(imgs[0][0]))
        save_img = os.path.join(dir1, "%06d.png" % (iid))
        cv2.imwrite(save_img, img[:, :, ::-1])
        img1 = image_to_numpy(deprocess(imgs[0][1]))
        save_img1 = os.path.join(dir1, "%06d.png" % (sampled_idx))
        cv2.imwrite(save_img1, img1[:, :, ::-1])
        get_volume_views(generated_volume, dir1, iid)

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
    parser.add_argument("--index")
    parser.add_argument("--output_dir")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = setup_logger(name="demo")
    logger.info("Arguments: " + str(args))
    
    cfg.CHECKPOINT = args.checkpoint
    data_dir = args.data_dir
    idx = int(args.index)
    output_dir = args.output_dir
    
    dataset = ShapeNetDataset(cfg, data_dir)
    item = dataset[idx] # img, verts, faces, points, normals, voxels, P, id_str, imgs, sid, mid, iid, sampled_idx
    imgs = item[-5]
    imgs = imgs.unsqueeze(0) # (1,2,3,224,224)
#     print(imgs.shape)
    sid = item[-4]
    mid = item[-3]
    iid = item[-2]
    sampled_idx = item[-1]
    
    demo = Visualization_demo(cfg, output_dir=output_dir)
    demo.run_on_images(imgs, sid, mid, iid, sampled_idx)
    logger.info("Reconstruction saved in %s" % (args.output_dir))
    
    
