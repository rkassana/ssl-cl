import torch
import torch.nn as nn


class Multi_head(nn.Module):
    def __init__(self, num_classes_per_task=2):
        super(Multi_head, self).__init__()

        self.projection1 = self.build_projection()

        self.projection2 = self.build_projection()

        self.projection3 = self.build_projection()

        self.projection4 = self.build_projection()

        self.projection5 = self.build_projection()

        self.out1 = nn.Linear(4096, num_classes_per_task)
        self.out2 = nn.Linear(4096, num_classes_per_task)
        self.out3 = nn.Linear(4096, num_classes_per_task)
        self.out4 = nn.Linear(4096, num_classes_per_task)
        self.out5 = nn.Linear(4096, num_classes_per_task)

    def build_projection(self):

        return nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, task_id=0):



        if task_id == 0:
            x = self.projection1(x)
            x1 = self.out1(x)
            return x1
        elif task_id == 1:
            x = self.projection2(x)
            x2 = self.out2(x)
            return x2
        elif task_id == 2:
            x = self.projection3(x)
            x3 = self.out3(x)
            return x3
        elif task_id == 3:
            x = self.projection4(x)
            x4 = self.out4(x)
            return x4
        elif task_id == 4:
            x = self.projection5(x)
            x5 = self.out5(x)
            return x5


