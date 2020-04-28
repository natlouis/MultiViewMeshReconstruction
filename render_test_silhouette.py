#!/usr/bin/env python
# coding: utf-8

#Render only silhouettes from each viewpoint

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.io import imread
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    SoftSilhouetteShader,
    HardPhongShader,
    BlendParams
)

from shapenet.utils.coords import SHAPENET_MAX_ZMAX, SHAPENET_MIN_ZMIN, project_verts, get_blender_intrinsic_matrix

# load a mesh
device = torch.device("cuda:0")
torch.cuda.set_device(device)
sid = '02958343'
mid = '8a49ece1b6d0b24dafe1d4530f4c6e24'
DATA_DIR = "./datasets/shapenet/ShapeNetCore.v1/"+sid+"/"+mid

metadata_path = os.path.join('./datasets/shapenet/ShapeNetV1processed', sid, mid, "metadata.pt")
metadata = torch.load(metadata_path)
K = metadata["intrinsic"]
RTs = metadata["extrinsics"].to(device)
#rotate 90 degrees about y-axis
rot_y_90 = torch.tensor([[0, 0, 1, 0], 
                            [0, 1, 0, 0], 
                            [-1, 0, 0, 0], 
                            [0, 0, 0, 1]]).to(RTs) 

#rotate -90 degrees about y-axis
rot_y_n90 = torch.tensor([[0, 0, -1, 0], 
                            [0, 1, 0, 0], 
                            [1, 0, 0, 0], 
                            [0, 0, 0, 1]]).to(RTs) 

#rotate -90 degrees about x-axis
rot_x_n90 = torch.tensor([[1, 0, 0, 0], 
                            [0, 0, 1, 0], 
                            [0, -1, 0, 0], 
                            [0, 0, 0, 1]]).to(RTs) 


invRT = torch.inverse(RTs[10].mm(rot_y_90))
invRT_no_rot = torch.inverse(RTs[0]) 
#Comment/uncomment GT .obj, Predicted Mesh, or GT Mesh to see plots for them

###########GT .obj
'''
obj_filename = os.path.join(DATA_DIR, "model.obj")
verts, faces_idx, _ = load_obj(obj_filename)

verts, faces_idx, _ = load_obj(obj_filename)
faces = faces_idx.verts_idx
'''

##########Predicted Mesh
#Generate this by running: python demo/demo_modified.py --config-file configs/shapenet/voxmesh_R50.yaml --data_dir datasets/shapenet/ShapeNetV1processed --output output_demo --checkpoint shapenet://voxmesh_R50.pth --index 0
'''
obj_filename = './output_demo/results_shapenet/02958343-8a49ece1b6d0b24dafe1d4530f4c6e24-10.obj'

verts, faces_idx, _ = load_obj(obj_filename)
faces = faces_idx.verts_idx
#verts = project_verts(verts, invRT.cpu())
'''

##########GT Mesh 
mesh_path = os.path.join('./datasets/shapenet/ShapeNetV1processed', sid, mid, "mesh.pt")
mesh_data = torch.load(mesh_path)
verts, faces = mesh_data["verts"], mesh_data["faces"]

verts_rgb = torch.ones_like(verts)[None] 
textures = Textures(verts_rgb=verts_rgb.to(device))
# print(verts_rgb.shape, verts.shape)
mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)

metadata_path = os.path.join('./datasets/shapenet/ShapeNetRendering', sid, mid, "rendering/rendering_metadata.txt")
metadata = []
with open(metadata_path, 'r') as f:
    for line in f:
        vals = [float(v) for v in line.strip().split(" ")]
        azimuth, elevation, yaw, dist_ratio, fov = vals 
        distance = 1.75 * dist_ratio
        metadata.append((azimuth, elevation, distance))

blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
raster_settings = RasterizationSettings(
    image_size=256, 
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
    faces_per_pixel=50, 
)

from matplotlib import gridspec

batch_size = 6
#fig = plt.figure(figsize=(8,8))
#ax = [fig.add_subplot(5,5,i+1) for i in range(24)]
nrow = 5
ncol = 5

fig = plt.figure(figsize=(ncol+1, nrow+1)) 

gs = gridspec.GridSpec(nrow, ncol,
         wspace=0.015, hspace=0.015, 
         top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
         left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 

#import pdb; pdb.set_trace()
for i, (azim, elev, dist) in enumerate(metadata):
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=-azim, device=device)
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )

    image = (renderer(meshes_world=mesh, R=R, T=T)>0).float()

    ax= plt.subplot(gs[i])
    ax.imshow(image.squeeze()[ ..., 3].cpu().numpy())
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')

#fig.subplots_adjust(wspace=0, hspace=0)
plt.show()
