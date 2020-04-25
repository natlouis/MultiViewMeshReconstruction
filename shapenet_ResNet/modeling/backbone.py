# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.nn as nn

import torchvision


class ResNetBackbone(nn.Module):
    def __init__(self, net):
        super(ResNetBackbone, self).__init__()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stage1 = net.layer1
        self.stage2 = net.layer2
        self.stage3 = net.layer3
        self.stage4 = net.layer4

    def forward(self, imgs):
        # imgs (N, 3, 137, 137)
        feats = self.stem(imgs) 
        conv1 = self.stage1(feats) 
        conv2 = self.stage2(conv1)
        conv3 = self.stage3(conv2) 
        conv4 = self.stage4(conv3) #(N, 512, 7, 7)

        return [conv1, conv2, conv3, conv4]

_FEAT_DIMS = {  
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
    "resnet101": (256, 512, 1024, 2048),
    "resnet152": (256, 512, 1024, 2048),
}

_FEAT_DIMS1 = {
    "resnet18": (64, 64, 64, 128, 128, 128, 256, 256, 256),
    "resnet34": (64, 64, 64, 128, 128, 128, 256, 256, 256),
}

def build_backbone(name, pretrained=True):
#     resnets = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    resnets = ["resnet18", "resnet34"]
    if name in resnets and name in _FEAT_DIMS:
        cnn = getattr(torchvision.models, name)(pretrained=pretrained)
        backbone = ResNetBackbone(cnn)
        feat_dims = _FEAT_DIMS[name]
        feat_dims1 = _FEAT_DIMS1[name]
        return backbone, feat_dims, feat_dims1
    else:
        raise ValueError('Unrecognized backbone type "%s"' % name)
