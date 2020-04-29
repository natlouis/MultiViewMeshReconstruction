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

from pytorch3d.ops import cubify
from pytorch3d.io import save_obj
# from shapenet.utils.coords import get_blender_intrinsic_matrix, voxel_to_world

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

        mesh = cubify(generated_volume, 0.3)
#         mesh = voxel_to_world(meshes)
        save_mesh = os.path.join(dir1, "%s_%s.obj" % (iid, sampled_idx))
        verts, faces = mesh.get_mesh_verts_faces(0)
        save_obj(save_mesh, verts, faces)
        
        generated_volume = generated_volume.squeeze()
        img = image_to_numpy(deprocess(imgs[0][0]))
        save_img = os.path.join(dir1, "%02d.png" % (iid))
#         cv2.imwrite(save_img, img[:, :, ::-1])
        cv2.imwrite(save_img, img)
        img1 = image_to_numpy(deprocess(imgs[0][1]))
        save_img1 = os.path.join(dir1, "%02d.png" % (sampled_idx))
        cv2.imwrite(save_img1, img1)
#         cv2.imwrite(save_img1, img1[:, :, ::-1])
        get_volume_views(generated_volume, dir1, iid, sampled_idx)
        
       
       
        
        

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
    parser.add_argument("--next_best_view", default=None)
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
    
    demo = Visualization_demo(cfg, output_dir=output_dir)
    demo.run_on_images(imgs, sid, mid, iid, sampled_idx)
    logger.info("Reconstruction saved in %s" % (args.output_dir))
    
    
