# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import Conv2d, ConvTranspose2d, get_norm
from torch import nn
from torch.nn import functional as F
from shapenet.utils.network_utils import init_weights
# from shapenet.modeling.models.decoder import Decoder

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

        assert self.voxel_size % 2 == 0

        self.conv_norm_relus = []
        prev_dim = input_channels
        
        for k in range(num_conv):
            conv = Conv2d(
                prev_dim,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("voxel_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            prev_dim = conv_dims
       
        self.conv_norm_relus.append(
            Conv2d(
                prev_dim,
                conv_dims,
                kernel_size=2,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
        )
#         self.decoder = Decoder(cfg)
#         self.deconv = ConvTranspose2d(
#             conv_dims if num_conv > 0 else input_channels,
#             conv_dims,
#             kernel_size=2,
#             stride=2,
#             padding=0,
#         )

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(2048, 512, kernel_size=4, stride=2, bias=cfg.MODEL.VOXEL_HEAD.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=cfg.MODEL.VOXEL_HEAD.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=cfg.MODEL.VOXEL_HEAD.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=cfg.MODEL.VOXEL_HEAD.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.MODEL.VOXEL_HEAD.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )
        
#         self.predictor = Conv2d(conv_dims, self.voxel_size, kernel_size=1, stride=1, padding=0)
        # initialization
        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        self.layer1.apply(init_weights)
        self.layer2.apply(init_weights)
        self.layer3.apply(init_weights)
        self.layer4.apply(init_weights)
        self.layer5.apply(init_weights)
        
        # use normal distribution initialization for voxel prediction layer
#         nn.init.normal_(self.predictor.weight, std=0.001)
#         if self.predictor.bias is not None:
#             nn.init.constant_(self.predictor.bias, 0)

    def forward(self, feat):
        # feat (N, 512, 7, 7)
        for layer in self.conv_norm_relus:
            feat = layer(feat)
#         raw_feature, gen_volume = self.decoder(feat)
        # feat (N,256,8,8)
#         print(feat.shape)
        gen_volume = feat.view(-1,2048,2,2,2)
        gen_volume = self.layer1(gen_volume)
        gen_volume = self.layer2(gen_volume)
        gen_volume = self.layer3(gen_volume)
        gen_volume = self.layer4(gen_volume)
        raw_feature = gen_volume
        gen_volume = self.layer5(gen_volume)
        raw_feature = torch.cat((raw_feature, gen_volume), dim=1)
        return raw_feature, gen_volume

#         x = F.relu(self.deconv(x))
#         voxel_scores = self.predictor(x)
#         return x
