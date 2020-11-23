import torch
import torch.nn as nn


class Multi_head(nn.Module):
    def __init__(self, num_classes_per_task=2):
        super(Multi_head, self).__init__()

        self.projection = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        self.out1 = nn.Linear(4096, num_classes_per_task)
        self.out2 = nn.Linear(4096, num_classes_per_task)
        self.out3 = nn.Linear(4096, num_classes_per_task)
        self.out4 = nn.Linear(4096, num_classes_per_task)
        self.out5 = nn.Linear(4096, num_classes_per_task)

    def forward(self, x):

        x  =self.projection(x)
        x1 = self.out1(x)
        x2 = self.out2(x)
        x3 = self.out3(x)
        x4 = self.out4(x)
        x5 = self.out5(x)

        return x1, x2, x3, x4, x5

