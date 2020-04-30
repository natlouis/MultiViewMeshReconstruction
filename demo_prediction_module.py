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

from shapenet.data.mesh_vox import MeshVoxDataset
# from shapenet.data.mesh_vox_multi import MeshVoxMultiDataset
from shapenet.modeling.mesh_arch import VoxMeshHead
from shapenet.utils import clean_state_dict 
import shapenet.utils.vis as vis_utils 
from shapenet.data.utils import image_to_numpy, imagenet_deprocess 
from shapenet.config import get_shapenet_cfg

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

logger = logging.getLogger("demo")

class VisualizationDemo(object):
      def __init__(self, cfg, output_dir="./vis"):
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
#         state_dict = torch.load('prediction_module_1500.pth', map_location='cuda:0')
        state_dict = torch.load('prediction_module_11500.pth')
        self.loss_predictor.load_state_dict(state_dict)

        self.predictor.to(self.device)
        self.loss_predictor.to(self.device)
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
 
      def run_on_image(self, image, id_str, gt_verts, gt_faces):
        deprocess = imagenet_deprocess(rescale_image=False)

        with torch.no_grad():
            voxel_scores, meshes_pred = self.predictor(image.to(self.device))

        sid,mid,iid = id_str.split('-')
        iid = int(iid)

        #Transform vertex space
        metadata_path = os.path.join('./datasets/shapenet/ShapeNetV1processed', sid, mid, "metadata.pt")
        metadata = torch.load(metadata_path)
        K = metadata["intrinsic"]
        RTs = metadata["extrinsics"].to(self.device)
        rot_y_90 = torch.tensor([[0, 0, 1, 0], 
                                    [0, 1, 0, 0], 
                                    [-1, 0, 0, 0], 
                                    [0, 0, 0, 1]]).to(RTs) 

        mesh = meshes_pred[-1][0]
        #For some strange reason all classes (expect vehicle class) require a 90 degree rotation about the y-axis
        #for the GT mesh
        invRT = torch.inverse(RTs[iid].mm(rot_y_90))
        invRT_no_rot = torch.inverse(RTs[iid])
        mesh._verts_list[0] = project_verts(mesh._verts_list[0], invRT)

        #Get look at view extrinsics
        render_metadata_path = os.path.join('datasets/shapenet/ShapeNetRenderingExtrinsics', sid, mid, 'rendering_metadata.pt')
        render_metadata = torch.load(render_metadata_path)
        render_RTs = render_metadata['extrinsics'].to(self.device)

        verts, faces = mesh.get_mesh_verts_faces(0)
        verts_rgb = torch.ones_like(verts)[None]
        textures = Textures(verts_rgb=verts_rgb.to(self.device))
        mesh.textures = textures 

        plt.figure(figsize=(10, 10))

        #Silhouette Renderer
        render_image_size = 256
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=render_image_size, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=50, 
        )   

        gt_verts = gt_verts.to(self.device)
        gt_faces = gt_faces.to(self.device)
        verts_rgb = torch.ones_like(gt_verts)[None]
        textures = Textures(verts_rgb=verts_rgb)
        #Invert without the rotation for the vehicle class
        if sid == '02958343':
            gt_verts = project_verts(gt_verts, invRT_no_rot.to(self.device))
        else:
            gt_verts = project_verts(gt_verts, invRT.to(self.device))
        gt_mesh = Meshes(
            verts=[gt_verts],
            faces=[gt_faces],
            textures=textures
        )

        probability_map = 0.01 * torch.ones((1, 24))
        viewgrid = torch.zeros((1,24,render_image_size,render_image_size)).to(self.device)
        fig = plt.figure(1)
        ax_pred = [fig.add_subplot(5,5,i+1) for i in range(24)]
        #fig = plt.figure(2)
        #ax_gt = [fig.add_subplot(5,5,i+1) for i in range(24)]

        for i in range(len(render_RTs)):
            if i == iid: #Don't include current view
                continue 

            R = render_RTs[i][:3,:3].unsqueeze(0)
            T = render_RTs[i][:3,3].unsqueeze(0)
            cameras = OpenGLPerspectiveCameras(device=self.device, R=R, T=T)

            silhouette_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
                ),  
            shader=SoftSilhouetteShader(blend_params=blend_params)
            )   

            ref_image        = (silhouette_renderer(meshes_world=gt_mesh, R=R, T=T)>0).float()
            silhouette_image = (silhouette_renderer(meshes_world=mesh, R=R, T=T)>0).float()

            # MSE Loss between both silhouettes
            silh_loss = torch.sum((silhouette_image[0, :, :, 3] - ref_image[0, :, :, 3]) ** 2)
            probability_map[0, i] = silh_loss.detach()

            viewgrid[0,i] = silhouette_image[...,-1]

            #ax_gt[i].imshow(ref_image[0,:,:,3].cpu().numpy())
            #ax_gt[i].set_title(i)

            ax_pred[i].imshow(silhouette_image[0,:,:,3].cpu().numpy())
            ax_pred[i].set_title(i)

        img = image_to_numpy(deprocess(image[0]))
        #ax_gt[iid].imshow(img)
        ax_pred[iid].imshow(img)
        #fig = plt.figure(3)
        #ax = fig.add_subplot(111)
        #ax.imshow(img)

        pred_prob_map = self.loss_predictor(viewgrid)
        print('Highest actual loss: {}'.format(torch.argmax(probability_map)))
        print('Highest predicted loss: {}'.format(torch.argmax(pred_prob_map)))
        plt.show()
        #plt.savefig('./output_demo/figures/'+id_str+'.png')
        #vis_utils.visualize_prediction(id_str, img, mesh, self.output_dir)
        return torch.argmax(pred_prob_map).item()

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
    
    parser.add_argument("--output", help="A directory to save output visualizations")
    
    parser.add_argument("--checkpoint",help="A path to a checkpoint file")
    parser.add_argument("--synset_id", default="04530566")
    parser.add_argument("--sample_size")
    
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
    synset_id = args.synset_id
    
    dataset = MeshVoxDataset(data_dir, return_mesh=True)
    
    synset_ids = dataset.synset_ids
    first_idx = synset_ids.index(synset_id)
    model_num = int(synset_ids.count(synset_id)/24)
    
    sample_size = int(args.sample_size)
    torch.manual_seed(0)
    idx_list = torch.randint(0, model_num, size=(sample_size,))
    idx_list += first_idx
    
    
    prediction_idx = []
    for idx in idx_list:

        item = dataset[idx] #(img, verts, faces, points, normals, voxels, P, id_str)
        img = item[0].unsqueeze(0)
        verts = item[1]
        faces = item[2]
        P     = item[6]
        id_str = item[7] 
        #_imgs = item[8]
        #render_RTs = item[9]
        #RT    = item[10]

        prediction = demo.run_on_image(img, id_str, verts, faces)
        prediction_idx.append(prediction)
    print(prediction_idx)