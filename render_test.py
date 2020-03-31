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
DATA_DIR = "./ShapeNetCore.v1/03001627/1033ee86cc8bac4390962e4fb7072b86"
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
# verts, faces, aux = load_obj(obj_filename)
# faces_idx = faces.verts_idx.to(device)
# verts = verts.to(device)
# verts_uvs = aux.verts_uvs[None, ...].to(device)
# faces_uvs = faces.textures_idx[None, ...].to(device)
# verts_rgb = torch.ones_like(verts)[None].to(device)
# textures = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs,verts_rgb = verts_rgb)
# mesh = Meshes(
#     verts=[verts.to(device)],   
#     faces=[faces.to(device)], 
#     textures=textures
# )


# In[3]:


# batch_size = 18
# meshes = mesh.extend(batch_size)
# textures = meshes.textures
# print(textures._verts_rgb_padded.shape)


# In[4]:


# batch_size = 18
# meshes = mesh.extend(batch_size)
# elev = torch.linspace(0, 180, batch_size)
# azim = torch.linspace(-180, 180, batch_size)
# elev = 0
# azim = 180
# R, T = look_at_view_transform(dist=3, elev=elev, azim=azim,device=device)
# cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
# lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
# raster_settings = RasterizationSettings(
#     image_size=128, 
#     blur_radius=0.0, 
#     faces_per_pixel=1, 
#     bin_size=0
# )
# renderer = MeshRenderer(
#     rasterizer=MeshRasterizer(
#         cameras=cameras, 
#         raster_settings=raster_settings
#     ),
#     shader=HardPhongShader(device=device, lights=lights)
# )


# In[5]:


# image = renderer(meshes_world=mesh, R=R, T=T)


# In[6]:


# plt.figure(figsize=(10, 10))
# plt.imshow(image[0, ..., :3].cpu().numpy())


# In[7]:


# images = renderer(meshes_world=meshes, R=R, T=T)


# In[8]:


batch_size = 6
elev_all = torch.linspace(0, 180, batch_size)
azim_all = torch.linspace(-180, 180, batch_size)
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
        plt.figure(figsize=(10, 10))
        plt.imshow(image[0, ..., :3].cpu().numpy())


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

