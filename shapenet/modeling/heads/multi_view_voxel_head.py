# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import Conv2d, ConvTranspose2d, get_norm
from torch import nn
from torch.nn import functional as F
##
from shapenet.modeling.models.encoder import Encoder
from shapenet.modeling.models.decoder import Decoder
from shapenet.modeling.models.merger import Merger
##

class VoxelHead(nn.Module):
    def __init__(self, cfg):
        super(VoxelHead, self).__init__()

        # fmt: off
        self.voxel_size = cfg.MODEL.VOXEL_HEAD.VOXEL_SIZE
        conv_dims       = cfg.MODEL.VOXEL_HEAD.CONV_DIM
        num_conv        = cfg.MODEL.VOXEL_HEAD.NUM_CONV
        input_channels  = cfg.MODEL.VOXEL_HEAD.COMPUTED_INPUT_CHANNELS
        self.norm       = cfg.MODEL.VOXEL_HEAD.NORM
        # fmt: on
##
        # add cfg??
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.merger = Merger()
        def forward(self, x):
            # x  torch.Size([batch_size, n_views, img_c, img_h, img_w])(_,_,3,224,224)
            feats = self.encoder(x)
            # feats  torch.Size([batch_size, n_views, 256, 8, 8])
            raw_features, coarse_volumes = self.decoder(feats)
            # gen_volumes   torch.Size([batch_size, n_views, 32, 32, 32])
            # raw_features   torch.Size([batch_size, n_views, 9, 32, 32, 32])
            coarse_volumes = self.merger(raw_features, coarse_volumes)
            # coarse_volumes  torch.Size([batch_size, 32, 32, 32])
            return coarse_volumes
        
        
        
        
##

#         assert self.voxel_size % 2 == 0

#         self.conv_norm_relus = []
#         prev_dim = input_channels
#         for k in range(num_conv):
#             conv = Conv2d(
#                 prev_dim,
#                 conv_dims,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=not self.norm,
#                 norm=get_norm(self.norm, conv_dims),
#                 activation=F.relu,
#             )
#             self.add_module("voxel_fcn{}".format(k + 1), conv)
#             self.conv_norm_relus.append(conv)
#             prev_dim = conv_dims

#         self.deconv = ConvTranspose2d(
#             conv_dims if num_conv > 0 else input_channels,
#             conv_dims,
#             kernel_size=2,
#             stride=2,
#             padding=0,
#         )
#         self.predictor = Conv2d(conv_dims, self.voxel_size, kernel_size=1, stride=1, padding=0)

#         for layer in self.conv_norm_relus + [self.deconv]:
#             weight_init.c2_msra_fill(layer)
#         # use normal distribution initialization for voxel prediction layer
#         nn.init.normal_(self.predictor.weight, std=0.001)
#         if self.predictor.bias is not None:
#             nn.init.constant_(self.predictor.bias, 0)

#     def forward(self, x):
#         V = self.voxel_size
#         x = F.interpolate(x, size=V // 2, mode="bilinear", align_corners=False)
#         for layer in self.conv_norm_relus:
#             x = layer(x)
#         x = F.relu(self.deconv(x))
#         x = self.predictor(x)
#         return x
