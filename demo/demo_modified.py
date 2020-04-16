#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import logging
# import multiprocessing as mp
import numpy as np
import os
import torch
from detectron2.utils.logger import setup_logger
from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes

from fvcore.common.file_io import PathManager

from shapenet.data.mesh_vox import MeshVoxDataset
from shapenet.modeling.mesh_arch import VoxMeshHead
from shapenet.utils import clean_state_dict 
import shapenet.utils.vis as vis_utils 
from shapenet.data.utils import image_to_numpy, imagenet_deprocess 
from shapenet.config import get_shapenet_cfg

import cv2

logger = logging.getLogger("demo")

class VisualizationDemo(object):
      def __init__(self, cfg, output_dir="./vis"):
        """
        Args:
            cfg (CfgNode):
        """
        self.predictor =  VoxMeshHead(cfg)

        #Load pretrained weights into model
        cp = torch.load(PathManager.get_local_path(cfg.MODEL.CHECKPOINT))
        state_dict = clean_state_dict(cp["best_states"]["model"])
        self.predictor.load_state_dict(state_dict)

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

      def run_on_image(self, image, id_str):
        deprocess = imagenet_deprocess(rescale_image=False)
        voxel_scores, meshes_pred = self.predictor(image)

        img = image_to_numpy(deprocess(image[0]))
        vis_utils.visualize_prediction(id_str, img, meshes_pred[-1][0], self.output_dir)

def setup_cfg(args):
    cfg = get_shapenet_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.CHECKPOINT = args.checkpoint 
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="ShapeNet Demo")
    parser.add_argument(
        "--config-file",
        default="configs/shapenet/voxmesh_R50.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--data_dir", help="A directory for input images")
    parser.add_argument("--index", help="The index of the input image")
    
    parser.add_argument("--output", help="A directory to save output visualizations")
    
    parser.add_argument("--checkpoint",help="A path to a checkpoint file")
    
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = setup_logger(name="demo")
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    output_dir = args.output
    demo = VisualizationDemo(
        cfg, output_dir=output_dir
    )

#     data_dir = './datasets/shapenet/ShapeNetV1processed'
    data_dir = args.data_dir
    dataset = MeshVoxDataset(data_dir)

    #Randomly select an image from the ShapeNet dataset if index args is empty
    if args.index is None:
        idx = np.random.randint(len(dataset))
    else:
        idx = int(args.index)

    item = dataset[idx] #(img, verts, faces, points, normals, voxels, P, id_str)
    img = item[0].unsqueeze(0)
    id_str = item[-1] 

    prediction = demo.run_on_image(img, id_str)
    logger.info("Predictions saved in %s" % (args.output))
