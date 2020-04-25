import torch
from torch import nn
from torchvision import models

class LossPredictionModule(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained = True)
        resnet.conv1 = nn.Conv2d(24, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet.fc = nn.Linear(in_features=512, out_features=24, bias=True)
        self.resnet = resnet
        
    def forward(self, viewgrid): 
        return self.resnet(viewgrid)

def demo():
    a = torch.zeros(10,24,256,256)
    model = LossPredictionModule()
    b = model(a)
    print(b.shape)

if __name__ == "__main__":
    demo()