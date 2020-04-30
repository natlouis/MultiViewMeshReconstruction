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
from pytorch3d.structures import Meshes, Textures

from fvcore.common.file_io import PathManager

#from shapenet.data.mesh_vox import MeshVoxDataset
from shapenet.data.mesh_vox_multi import MeshVoxMultiDataset
from shapenet.modeling.mesh_arch import VoxMeshHead
from shapenet.utils import clean_state_dict 
import shapenet.utils.vis as vis_utils 
from shapenet.data.utils import image_to_numpy, imagenet_deprocess 
from shapenet.config import get_shapenet_cfg
from shapenet.data import build_data_loader, register_shapenet

from shapenet.utils.coords import project_verts 

from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    HardPhongShader,
    BlendParams,
    PointLights
)

from tools.loss_prediction_module import LossPredictionModule

import cv2
import matplotlib.pyplot as plt
import glob

#import wandb

logger = logging.getLogger("demo")

class VisualizationDemo(object):
      def __init__(self, cfg, checkpoint_lp_model, output_dir="./vis"):
        """
        Args:
            cfg (CfgNode):
        """
        self.predictor =  VoxMeshHead(cfg)

        self.device = torch.device('cuda')
        #Load pretrained weights into model
        cp = torch.load(PathManager.get_local_path(cfg.MODEL.CHECKPOINT))
        state_dict = clean_state_dict(cp["best_states"]["model"])
        self.predictor.load_state_dict(state_dict)

        self.loss_predictor = LossPredictionModule()
        #Path to trained prediction module
        state_dict = torch.load(checkpoint_lp_model, map_location='cuda:0')
        self.loss_predictor.load_state_dict(state_dict)

        self.predictor.to(self.device)
        self.loss_predictor.to(self.device)
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

      def run_on_image(self, imgs, id_strs, meshes_gt, render_RTs, RTs):
        deprocess = imagenet_deprocess(rescale_image=False)

        #Silhouette Renderer
        render_image_size = 256
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=render_image_size, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=50, 
        )   

        rot_y_90 = torch.tensor([[0, 0, 1, 0], 
                                    [0, 1, 0, 0], 
                                    [-1, 0, 0, 0], 
                                    [0, 0, 0, 1]]).to(RTs) 
        with torch.no_grad():
            voxel_scores, meshes_pred = self.predictor(imgs)

        B,_,H,W = imgs.shape
        probability_map = 0.01 * torch.ones((B, 24)).to(self.device)
        viewgrid = torch.zeros((B,24,render_image_size,render_image_size)).to(device) # batch size x 24 x H x W
        _meshes_pred = meshes_pred[-1]

        for b, (cur_gt_mesh, cur_pred_mesh) in enumerate(zip(meshes_gt, _meshes_pred)):
            sid = id_strs[b].split('-')[0]

            RT = RTs[b]

            #For some strange reason all classes (expect vehicle class) require a 90 degree rotation about the y-axis
            #for the GT mesh
            invRT = torch.inverse(RT.mm(rot_y_90))
            invRT_no_rot = torch.inverse(RT)

            cur_pred_mesh._verts_list[0] = project_verts(cur_pred_mesh._verts_list[0], invRT)
            #Invert without the rotation for the vehicle class
            if sid == '02958343':
                cur_gt_mesh._verts_list[0] = project_verts(
                    cur_gt_mesh._verts_list[0], invRT_no_rot)
            else:
                cur_gt_mesh._verts_list[0] = project_verts(
                    cur_gt_mesh._verts_list[0], invRT)

            '''
            plt.figure(figsize=(10, 10))

            fig = plt.figure(1)
            ax_pred = [fig.add_subplot(5,5,i+1) for i in range(24)]
            fig = plt.figure(2)
            ax_gt = [fig.add_subplot(5,5,i+1) for i in range(24)]
            '''

            for iid in range(len(render_RTs)):

                R = render_RTs[b][iid][:3,:3].unsqueeze(0)
                T = render_RTs[b][iid][:3,3].unsqueeze(0)
                cameras = OpenGLPerspectiveCameras(device=self.device, R=R, T=T)
                silhouette_renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(
                    cameras=cameras, 
                    raster_settings=raster_settings
                    ),  
                shader=SoftSilhouetteShader(blend_params=blend_params)
                )   

                ref_image        = (silhouette_renderer(meshes_world=cur_gt_mesh, R=R, T=T)>0).float()
                silhouette_image = (silhouette_renderer(meshes_world=cur_pred_mesh, R=R, T=T)>0).float()

                #Add image silhouette to viewgrid
                viewgrid[b,iid] = silhouette_image[...,-1]

                # MSE Loss between both silhouettes
                silh_loss = torch.sum((silhouette_image[0, :, :, 3] - ref_image[0, :, :, 3]) ** 2)
                probability_map[b, iid] = silh_loss.detach()

                '''
                ax_gt[iid].imshow(ref_image[0,:,:,3].cpu().numpy())
                ax_gt[iid].set_title(iid)
                ax_pred[iid].imshow(silhouette_image[0,:,:,3].cpu().numpy())
                ax_pred[iid].set_title(iid)
                '''

            '''
            img = image_to_numpy(deprocess(imgs[b]))
            ax_gt[iid].imshow(img)
            ax_pred[iid].imshow(img)
            fig = plt.figure(3)
            ax = fig.add_subplot(111)
            ax.imshow(img)

            #plt.show()
            '''

        probability_map = probability_map/(torch.max(probability_map, dim=1)[0].unsqueeze(1)) # Normalize
        probability_map = torch.nn.functional.softmax(probability_map, dim=1) # Softmax across images
        pred_prob_map = self.loss_predictor(viewgrid)

        gt_max = torch.argmax(probability_map, dim=1)
        pred_max = torch.argmax(pred_prob_map, dim=1)

        #print('--'*30)
        #print('Item: {}'.format(id_str))
        #print('Highest actual loss: {}'.format(gt_max))
        #print('Highest predicted loss: {}'.format(pred_max))

        #print('GT prob map: {}'.format(probability_map.squeeze()))
        #print('Pred prob map: {}'.format(pred_prob_map.squeeze()))

        correct = torch.sum(pred_max == gt_max).item()
        total   = len(pred_prob_map)

        return correct, total

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
    parser.add_argument("--saved_weights_dir", help="Path to saved weights to eval from")
    parser.add_argument("--split", default='val', help='train_eval or val split')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = setup_logger(name="demo")
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    data_dir, splits_file = register_shapenet(cfg.DATASETS.NAME)
    cfg.DATASETS.DATA_DIR = data_dir
    cfg.DATASETS.SPLITS_FILE = splits_file
    split = args.split

    data_loader = build_data_loader(cfg, 'MeshVoxMulti', split, multigpu=False, num_workers=8)
    print('Steps in dataloader: {}'.format(len(data_loader)))

    data_dir = args.data_dir
    saved_weights_dir = args.saved_weights_dir
    saved_weights = sorted(glob.glob(saved_weights_dir+'/*.pth'))

    #wandb.init(project='MeshRCNN', config=cfg, name='eval_'+split+'_'+saved_weights_dir)
    for checkpoint_lp_model in saved_weights:
        running_correct = 0
        running_total   = 0
        print('Checkpoint: {}'.format(checkpoint_lp_model))
        step = int(checkpoint_lp_model.split('_')[-1].split('.pth')[0])
        demo = VisualizationDemo(
            cfg, checkpoint_lp_model=checkpoint_lp_model
        )

        device = torch.device('cuda')
        for batch in data_loader:
            batch = data_loader.postprocess(batch, device)
            imgs, meshes_gt, points_gt, normals_gt,\
                voxels_gt, id_strs, _, render_RTs, RTs = batch

            correct, total = demo.run_on_image(imgs, id_strs, meshes_gt, render_RTs, RTs)

            running_correct += correct
            running_total   += total

        avg_acc = (running_correct*1.0/running_total)*100
        print('Accuracy : {:.3f}'.format(avg_acc))
        #wandb.log({'step':step, 'accuracy':avg_acc})
