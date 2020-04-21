# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
#         self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.stage1 = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:7]
        self.stage2 = torch.nn.Sequential(*list(vgg16_bn.features.children()))[7:14]
        self.stage3 = torch.nn.Sequential(*list(vgg16_bn.features.children()))[14:24]
        self.stage4 = torch.nn.Sequential(*list(vgg16_bn.features.children()))[24:27]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU()
        )

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False
       
    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        feats1 = []
        feats2 = []
        feats3 = []
        feats4 = []
        image_features = []
        for img in rendering_images:
            feat1 = self.stage1(img.squeeze(dim=0))#64
            feat2 = self.stage2(feat1)#128
            feat3 = self.stage2(feat2)#256
            feat4 = self.stage2(feat3)#512
            # feat4.size   torch.Size([batch_size, 512, 28, 28])
            
#             features = self.vgg(img.squeeze(dim=0))
#             print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            feat5 = self.layer1(feat4)
            # print(feat5.size())    # torch.Size([batch_size, 512, 26, 26])
            feat6 = self.layer2(feat5)
            # print(feat6.size())    # torch.Size([batch_size, 512, 24, 24])
            feat7 = self.layer3(feat6)
            # print(feat7.size())    # torch.Size([batch_size, 256, 8, 8])
            feats1.append(feat1)
            feats2.append(feat2)
            feats3.append(feat3)
            image_features.append(feat7)
        feats1 = torch.stack(feats1).permute(1, 0, 2, 3, 4).contiguous() 
        feats2 = torch.stack(feats2).permute(1, 0, 2, 3, 4).contiguous() 
        feats3 = torch.stack(feats3).permute(1, 0, 2, 3, 4).contiguous() 
        feats2 = torch.stack(feats2).permute(1, 0, 2, 3, 4).contiguous() 
        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 8, 8])
        feats1_mean = torch.mean(feats1, 1, True).squeeze()
        feats1_std = torch.std(feats1, 1, True).squeeze()
        feats1_max = torch.max(feats1, 1, True)[0].squeeze()
        
        feats2_mean = torch.mean(feats2, 1, True).squeeze()
        feats2_std = torch.std(feats2, 1, True).squeeze()
        feats2_max = torch.max(feats2, 1, True)[0].squeeze()
        
        feats3_mean = torch.mean(feats3, 1, True).squeeze()
        feats3_std = torch.std(feats3, 1, True).squeeze()
        feats3_max = torch.max(feats3, 1, True)[0].squeeze()
        
        static_features = [feats1_mean, feats1_std, feats1_max, feats2_mean, feats2_std, feats2_max, feats3_mean, feats3_std, feats3_max] 
        feat_dims = [64,64,64,128,128,128,256,256,256]
        return image_features, static_features,feat_dims
  
    
    
    
