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

        #Load pretrained weights into model
        cp = torch.load(PathManager.get_local_path(cfg.MODEL.CHECKPOINT))
        state_dict = clean_state_dict(cp["best_states"]["model"])
        self.predictor.load_state_dict(state_dict)

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

      def run_on_image(self, image, id_str, gt_verts, gt_faces):
        deprocess = imagenet_deprocess(rescale_image=False)

        with torch.no_grad():
            voxel_scores, meshes_pred = self.predictor(image)

        sid,mid,iid = id_str.split('-')
        iid = int(iid)

        #Transform vertex space
        metadata_path = os.path.join('./datasets/shapenet/ShapeNetV1processed', sid, mid, "metadata.pt")
        metadata = torch.load(metadata_path)
        K = metadata["intrinsic"]
        RTs = metadata["extrinsics"]
        rot_y_90 = torch.tensor([[0, 0, 1, 0], 
                                    [0, 1, 0, 0], 
                                    [-1, 0, 0, 0], 
                                    [0, 0, 0, 1]]).to(RTs) 

        mesh = meshes_pred[-1][0]
        #For some strange reason all classes (expect vehicle class) require a 90 degree rotation about the y-axis
        #for the GT mesh
        invRT = torch.inverse(RTs[iid].mm(rot_y_90))
        invRT_no_rot = torch.inverse(RTs[iid])
        mesh._verts_list[0] = project_verts(mesh._verts_list[0], invRT.cpu())

        #Get look at view extrinsics
        render_metadata_path = os.path.join('datasets/shapenet/ShapeNetRenderingExtrinsics', sid, mid, 'rendering_metadata.pt')
        render_metadata = torch.load(render_metadata_path)
        render_RTs = render_metadata['extrinsics']

        plt.figure(figsize=(10, 10))
        #R, T = look_at_view_transform(dist=0.742719815206*1.75, elev=27.0590432267, azim=-19.5372820907)
        R = render_RTs[iid][:3,:3].unsqueeze(0)
        T = render_RTs[iid][:3,3].unsqueeze(0)
        cameras = OpenGLPerspectiveCameras(R=R, T=T)

        #Phong Renderer
        lights = PointLights(location=[[0.0, 0.0, -3.0]])
        raster_settings = RasterizationSettings(
            image_size=137, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            bin_size=0
        )   
        phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
            ),  
        shader=HardPhongShader(lights=lights)
        )

        #Silhouette Renderer
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=137, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=250, 
        )   
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
            ),  
        shader=SoftSilhouetteShader(blend_params=blend_params)
        )   

        verts, faces = mesh.get_mesh_verts_faces(0)
        verts_rgb = torch.ones_like(verts)[None]
        textures = Textures(verts_rgb=verts_rgb)
        mesh.textures = textures 

        verts_rgb = torch.ones_like(gt_verts)[None]
        textures = Textures(verts_rgb=verts_rgb)
        #Invert without the rotation for the vehicle class
        if sid == '02958343':
            gt_verts = project_verts(gt_verts, invRT_no_rot.cpu())
        else:
            gt_verts = project_verts(gt_verts, invRT.cpu())
        gt_mesh = Meshes(
	    verts=[gt_verts],   
	    faces=[gt_faces], 
	    textures=textures
	) 

        img = image_to_numpy(deprocess(image[0]))
        mesh_image = phong_renderer(meshes_world=mesh, R=R, T=T)
        gt_silh_image = silhouette_renderer(meshes_world=gt_mesh, R=R, T=T)
        silhouette_image = silhouette_renderer(meshes_world=mesh, R=R, T=T)

        plt.subplot(2,2,1)
        plt.imshow(img)
        plt.title('input image')
        plt.subplot(2,2,2)
        plt.imshow(mesh_image[0, ..., :3].cpu().numpy())
        plt.title('rendered mesh')
        plt.subplot(2,2,3)
        plt.imshow(gt_silh_image[0, ..., 3].cpu().numpy())
        plt.title('silhouette of gt mesh')
        plt.subplot(2,2,4)
        plt.imshow(silhouette_image[0, ..., 3].cpu().numpy())
        plt.title('silhouette of rendered mesh')

        plt.show()
        #plt.savefig('./output_demo/figures/'+id_str+'.png')

        vis_utils.visualize_prediction(id_str, img, mesh, self.output_dir)

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
    dataset = MeshVoxDataset(data_dir, return_mesh=True)

    #Randomly select an image from the ShapeNet dataset if index args is empty
    if args.index is None:
        idx = np.random.randint(len(dataset))
    else:
        idx = int(args.index)

    item = dataset[idx] #(img, verts, faces, points, normals, voxels, P, id_str)
    img = item[0].unsqueeze(0)
    verts = item[1]
    faces = item[2]
    P     = item[6]
    id_str = item[7] 

    prediction = demo.run_on_image(img, id_str, verts, faces)
    logger.info("Predictions saved in %s" % (args.output))
