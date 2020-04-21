# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import Conv2d, ConvTranspose2d, get_norm
from torch import nn
from torch.nn import functional as F
##
from shapenet.modeling.models.encoder_modified import Encoder
from shapenet.modeling.models.decoder import Decoder
from shapenet.modeling.models.merger import Merger
##

class MultiViewVoxelHead(nn.Module):
    def __init__(self, cfg):
        super(VoxelHead, self).__init__()

        # fmt: off
#         self.voxel_size = cfg.MODEL.VOXEL_HEAD.VOXEL_SIZE
#         conv_dims       = cfg.MODEL.VOXEL_HEAD.CONV_DIM
#         num_conv        = cfg.MODEL.VOXEL_HEAD.NUM_CONV
#         input_channels  = cfg.MODEL.VOXEL_HEAD.COMPUTED_INPUT_CHANNELS
#         self.norm       = cfg.MODEL.VOXEL_HEAD.NORM
        # fmt: on
##
        # modify cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.merger = Merger(cfg)
        
    def forward(self, x):
        # x  torch.Size([batch_size, n_views, img_c, img_h, img_w])(_,_,3,224,224)
        feats, static_feats, feat_dims = self.encoder(x)
        # feats  torch.Size([batch_size, n_views, 256, 8, 8])
        # static_feats: a list
        raw_features, coarse_volumes = self.decoder(feats)
        # raw_features   torch.Size([batch_size, n_views, 9, 32, 32, 32])
        # coarse_volumes   torch.Size([batch_size, n_views, 32, 32, 32])
        coarse_volumes = self.merger(raw_features, coarse_volumes)
        # coarse_volumes  torch.Size([batch_size, 32, 32, 32])
        return coarse_volumes, static_feats, feat_dims
