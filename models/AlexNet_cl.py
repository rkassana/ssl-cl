import torch
import torch.nn as nn


class AlexNet_encoder(nn.Module):
    def __init__(self):
        super(AlexNet_encoder, self).__init__()

        self.feature1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True))

        self.maxpool1 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        self.feature2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.maxpool2 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        self.feature3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x, return_indices=False):
        indexes = []
        x = self.feature1(x)
        x, idx = self.maxpool1(x)
        indexes.append(idx)
        x = self.feature2(x)
        x, idx = self.maxpool2(x)
        indexes.append(idx)

        x = self.feature3(x)
        x, idx = self.maxpool3(x)
        indexes.append(idx)
        x = x.view(x.size(0), 256 * 2 * 2)

        if return_indices:
            #indices that are needed for decoder to maxunpool
            return x, indexes
        else:
            return x

