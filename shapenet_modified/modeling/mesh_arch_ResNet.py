# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from detectron2.utils.registry import Registry
from pytorch3d.ops import cubify
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from shapenet.modeling.backbone import build_backbone
from shapenet.modeling.heads import MeshRefinementHead
from shapenet.modeling.heads.voxel_head_ResNet import VoxelHead
from shapenet.utils.coords import get_blender_intrinsic_matrix, voxel_to_world
from shapenet.modeling.models.merger import Merger
from shapenet.utils.network_utils import init_weights

MESH_ARCH_REGISTRY = Registry("MESH_ARCH")


@MESH_ARCH_REGISTRY.register()
class VoxMeshHead(nn.Module):
    def __init__(self, cfg):
        super(VoxMeshHead, self).__init__()

        # fmt: off
        backbone                = cfg.MODEL.BACKBONE
        self.cubify_threshold   = cfg.MODEL.VOXEL_HEAD.CUBIFY_THRESH
        self.voxel_size         = cfg.MODEL.VOXEL_HEAD.VOXEL_SIZE
        # fmt: on

        self.register_buffer("K", get_blender_intrinsic_matrix())
        # backbone
        self.backbone, feat_dims, feat_dims1 = build_backbone(backbone)
        # voxel head
        cfg.MODEL.VOXEL_HEAD.COMPUTED_INPUT_CHANNELS = feat_dims[-1]
        self.voxel_head = VoxelHead(cfg)
        
        # merger
        self.merger = Merger(cfg)
        self.merger.apply(init_weights)
        # mesh head
        cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS = sum(feat_dims1)
        self.mesh_head = MeshRefinementHead(cfg)

    def _get_projection_matrix(self, N, device):
        return self.K[None].repeat(N, 1, 1).to(device).detach()

    def _dummy_mesh(self, N, device):
        verts_batch = torch.randn(N, 4, 3, device=device)
        faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
        faces = torch.tensor(faces, dtype=torch.int64)
        faces_batch = faces.view(1, 4, 3).expand(N, 4, 3).to(device)
        return Meshes(verts=verts_batch, faces=faces_batch)

    def cubify(self, voxel_scores):
        V = self.voxel_size
        N = voxel_scores.shape[0]
        voxel_probs = voxel_scores.sigmoid()
        active_voxels = voxel_probs > self.cubify_threshold
        voxels_per_mesh = (active_voxels.view(N, -1).sum(dim=1)).tolist()
        start = V // 4
        stop = start + V // 2
        for i in range(N):
            if voxels_per_mesh[i] == 0:
                print('ERROR is here')
                voxel_probs[i, start:stop, start:stop, start:stop] = 1
        meshes = cubify(voxel_probs, self.cubify_threshold)

        meshes = self._add_dummies(meshes)
        meshes = voxel_to_world(meshes)
        return meshes

    def _add_dummies(self, meshes):
        N = len(meshes)
        dummies = self._dummy_mesh(N, meshes.device)
        verts_list = meshes.verts_list()
        faces_list = meshes.faces_list()
        for i in range(N):
            if faces_list[i].shape[0] == 0:
                # print('Adding dummmy mesh at index ', i)
                vv, ff = dummies.get_mesh(i)
                verts_list[i] = vv
                faces_list[i] = ff
        return Meshes(verts=verts_list, faces=faces_list)

    def forward(self, imgs, voxel_only=False):
        # imgs [batch_size, n_views, img_c, img_h, img_w] (_,_,3,137,137)
        N = imgs.shape[0]
        device = imgs.device
        
        imgs = imgs.permute(1, 0, 2, 3, 4).contiguous()
        imgs = torch.split(imgs, 1, dim=0)
        feat1 = []
        feat2 = []
        feat3 = []
        raw_features = []
        gen_volumes = []
        for img in imgs:
            img_feats = self.backbone(img.squeeze(0))
            # img_feats: a list of 4 tensors
            feat1.append(img_feats[0])
            feat2.append(img_feats[1])
            feat3.append(img_feats[2])
            raw_feature, gen_volume = self.voxel_head(img_feats[-1])
            gen_volumes.append(torch.squeeze(gen_volume, dim=1))
            raw_features.append(raw_feature)
            
            
        feat1 = torch.stack(feat1).permute(1, 0, 2, 3, 4).contiguous() 
        feat2 = torch.stack(feat2).permute(1, 0, 2, 3, 4).contiguous() 
        feat3 = torch.stack(feat3).permute(1, 0, 2, 3, 4).contiguous() 
        
        feat1_mean = torch.mean(feat1, 1, True).squeeze(1)
        feat1_std = torch.std(feat1, 1, True).squeeze(1)
        feat1_max = torch.max(feat1, 1, True)[0].squeeze(1)
        
        feat2_mean = torch.mean(feat2, 1, True).squeeze(1)
        feat2_std = torch.std(feat2, 1, True).squeeze(1)
        feat2_max = torch.max(feat2, 1, True)[0].squeeze(1)
        
        feat3_mean = torch.mean(feat3, 1, True).squeeze(1)
        feat3_std = torch.std(feat3, 1, True).squeeze(1)
        feat3_max = torch.max(feat3, 1, True)[0].squeeze(1)
        
        static_feats = [feat1_mean, feat1_std, feat1_max, feat2_mean, feat2_std, feat2_max, feat3_mean, feat3_std, feat3_max] 
#         print(raw_features[0].shape)
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous() 
        gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
        voxel_scores = self.merger(raw_features, gen_volumes)

        P = self._get_projection_matrix(N, device)

        if voxel_only:
            dummy_meshes = self._dummy_mesh(N, device)
            dummy_refined = self.mesh_head(static_feats, dummy_meshes, P)
            return voxel_scores, dummy_refined

        cubified_meshes = self.cubify(voxel_scores)
        refined_meshes = self.mesh_head(static_feats, cubified_meshes, P)
        return voxel_scores, refined_meshes


@MESH_ARCH_REGISTRY.register()
class SphereInitHead(nn.Module):
    def __init__(self, cfg):
        super(SphereInitHead, self).__init__()

        # fmt: off
        backbone                = cfg.MODEL.BACKBONE
        self.ico_sphere_level   = cfg.MODEL.MESH_HEAD.ICO_SPHERE_LEVEL
        # fmt: on

        self.register_buffer("K", get_blender_intrinsic_matrix())
        # backbone
        self.backbone, feat_dims = build_backbone(backbone)
        # mesh head
        cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS = sum(feat_dims)
        self.mesh_head = MeshRefinementHead(cfg)

    def _get_projection_matrix(self, N, device):
        return self.K[None].repeat(N, 1, 1).to(device).detach()

    def forward(self, imgs):
        N = imgs.shape[0]
        device = imgs.device

        img_feats = self.backbone(imgs)
        P = self._get_projection_matrix(N, device)

        init_meshes = ico_sphere(self.ico_sphere_level, device).extend(N)
        refined_meshes = self.mesh_head(img_feats, init_meshes, P)
        return None, refined_meshes


@MESH_ARCH_REGISTRY.register()
class Pixel2MeshHead(nn.Module):
    def __init__(self, cfg):
        super(Pixel2MeshHead, self).__init__()

        # fmt: off
        backbone                = cfg.MODEL.BACKBONE
        self.ico_sphere_level   = cfg.MODEL.MESH_HEAD.ICO_SPHERE_LEVEL
        # fmt: on

        self.register_buffer("K", get_blender_intrinsic_matrix())
        # backbone
        self.backbone, feat_dims = build_backbone(backbone)
        # mesh head
        cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS = sum(feat_dims)
        self.mesh_head = MeshRefinementHead(cfg)

    def _get_projection_matrix(self, N, device):
        return self.K[None].repeat(N, 1, 1).to(device).detach()

    def forward(self, imgs):
        N = imgs.shape[0]
        device = imgs.device

        img_feats = self.backbone(imgs)
        P = self._get_projection_matrix(N, device)

        init_meshes = ico_sphere(self.ico_sphere_level, device).extend(N)
        refined_meshes = self.mesh_head(img_feats, init_meshes, P, subdivide=True)
        return None, refined_meshes


def build_model(cfg):
    name = cfg.MODEL.MESH_HEAD.NAME
    return MESH_ARCH_REGISTRY.get(name)(cfg)
