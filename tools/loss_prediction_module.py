import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class LossPredictionModule(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(24, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet.fc = nn.Linear(in_features=512, out_features=24, bias=True)
        self.resnet = resnet

    def forward(self, viewgrid):  # viewgrid is (24, H, W)
        scores = self.resnet(viewgrid)
        prob_map = nn.functional.softmax(scores, dim=1)  # (N,24)
        return prob_map

    def train_batch(self, pred_viewgrid, gt_prob_map, optimizer):
        pred_prob_map = self.forward(pred_viewgrid)
        optimizer.zero_grad()
        gt_prob_map.requires_grad = False
        loss = F.mse_loss(pred_prob_map, gt_prob_map, reduction="sum")
        print("loss prediction mse loss:", loss)
        loss.backward()
        optimizer.step()


def demo():
    a = torch.zeros(10, 24, 256, 256)
    model = LossPredictionModule()
    b = model(a)
    print(b.shape)


if __name__ == "__main__":
    demo()
