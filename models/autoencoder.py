import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F


class AE(nn.Module):

    def __init__(self, encoder = None):
        super(AE, self).__init__()

        self.encoder = encoder

        self.unmaxpool1 = nn.MaxUnpool2d(kernel_size=2)

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

        )

        self.unmaxpool2 = nn.MaxUnpool2d(kernel_size=2)

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(192,64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.unmaxpool3 = nn.MaxUnpool2d(kernel_size=2)

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x, indices = self.encoder(x, True)

        x = x.view(x.size(0), 256, 2, 2)

        x = self.unmaxpool1(x,indices[-1])

        x = self.decoder1(x)

        x = self.unmaxpool2(x, indices[-2])

        x = self.decoder2(x)

        x = self.unmaxpool3(x, indices[-3])

        x = self.decoder3(x)

        return x