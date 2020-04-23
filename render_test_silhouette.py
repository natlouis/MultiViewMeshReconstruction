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
    BlendParams
)

from shapenet.utils.coords import SHAPENET_MAX_ZMAX, SHAPENET_MIN_ZMIN, project_verts, get_blender_intrinsic_matrix

# load a mesh
device = torch.device("cuda:0")
torch.cuda.set_device(device)
sid = '02958343'
mid = '4856ef1e80d356d111f983eb293b51a'
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
obj_filename = './output_demo/results_shapenet/02958343-4856ef1e80d356d111f983eb293b51a-00.obj'

verts, faces_idx, _ = load_obj(obj_filename)
faces = faces_idx.verts_idx
invRT = torch.inverse(RTs[0].mm(rot_y_90))
#invRT = torch.inverse(RTs[0].mm(rot_x_n90)) 
#invRT = torch.inverse(RTs[0]) 
verts = project_verts(verts, invRT.cpu())

##########GT Mesh 
'''
mesh_path = os.path.join('./datasets/shapenet/ShapeNetV1processed', sid, mid, "mesh.pt")
mesh_data = torch.load(mesh_path)
verts, faces = mesh_data["verts"], mesh_data["faces"]
verts = project_verts(verts, RTs[0].cpu())
'''

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

batch_size = 6
plt.figure(figsize=(10, 10))
plt.title('R2N2 transformation settings')
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

    image = renderer(meshes_world=mesh, R=R, T=T)
    plt.subplot(5,5,i+1)
    plt.title(str(i).zfill(2)+'.png')
    plt.imshow(image.squeeze()[ ..., 3].cpu().numpy())

plt.show()

batch_size = 4
elev_all = torch.linspace(0, 180, batch_size)
azim_all = torch.linspace(-180, 180, batch_size)
i = 1
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
    bin_size=0
)
for elev in elev_all:
    for azim in azim_all:
        R, T = look_at_view_transform(dist=1.5, elev=elev, azim=azim,device=device)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=device, lights=lights)
        )
        image = renderer(meshes_world=mesh, R=R, T=T)

        print('azim: {}, elev: {}'.format(azim, elev))
        plt.subplot(5,5,i)
        plt.imshow(image[0, ..., :3].cpu().numpy())

        i += 1
plt.show()

