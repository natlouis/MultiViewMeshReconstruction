#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
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
#     SoftSilhouetteShader,
    HardPhongShader
)
# import sys
# import os
# sys.path.append(os.path.abspath(''))
# from utils import image_grid


# In[2]:


# load a mesh
device = torch.device("cuda:0")
torch.cuda.set_device(device)
sid = '02958343'
mid = 'bf37249fc8e16fd8f9a88cc63b910f3'
DATA_DIR = "./datasets/shapenet/ShapeNetCore.v1/"+sid+"/"+mid
obj_filename = os.path.join(DATA_DIR, "model.obj")
verts, faces_idx, _ = load_obj(obj_filename)
faces = faces_idx.verts_idx
verts_rgb = torch.ones_like(verts)[None] 
textures = Textures(verts_rgb=verts_rgb.to(device))
# print(verts_rgb.shape, verts.shape)
mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)
from shapenet.utils.coords import compute_extrinsic_matrix
metadata_path = os.path.join('./datasets/shapenet/ShapeNetV1processed', sid, mid, "metadata.pt")
metadata = torch.load(metadata_path)
K = metadata["intrinsic"]
RT = metadata["extrinsics"][0].to(device) #Extrinsics for first image only
#rot = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).to(device)
#RT = RT.mm(rot.to(RT))

batch_size = 6
elev_all = torch.linspace(0, 180, batch_size)
azim_all = torch.linspace(-180, 180, batch_size)
images = []
for elev in elev_all:
    for azim in azim_all:
        #R = RT[:3,:3].unsqueeze(0)
        #T = -1*RT[:3,3].unsqueeze(0)

        R, T = look_at_view_transform(dist=-3.0, elev=elev, azim=azim, device=device)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        raster_settings = RasterizationSettings(
            image_size=512, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            bin_size=0
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
            ),
        shader=HardPhongShader(device=device, lights=lights)
        )
        image = renderer(meshes_world=mesh, R=R, T=T)
        plt.figure(figsize=(10, 10))
        plt.imshow(image[0, ..., :3].cpu().numpy())

        plt.show()


# In[9]:


batch_size = 6
elev_all = torch.linspace(0, 180, batch_size)
azim_all = torch.linspace(-180, 180, batch_size)
images = []
for elev in elev_all:
    for azim in azim_all:
        R, T = look_at_view_transform(dist=3, elev=elev, azim=azim,device=device)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        raster_settings = RasterizationSettings(
            image_size=512, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            bin_size=0
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
            ),
        shader=HardPhongShader(device=device, lights=lights)
        )
        image = renderer(meshes_world=mesh, R=R, T=T)
        images.append(image[0, ..., :3].cpu().numpy())

