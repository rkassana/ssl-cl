import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ConvModel(nn.Module):
    def __init__(self, num_classes, in_chanel):
        super(ConvModel, self).__init__()
        self.num_classes = num_classes
        self.in_channel = in_chanel
        self.layer1 = self._make_layers(in_channels=in_chanel, plans=10, kernel_size=5, stride=1)
        self.layer2 = self._make_layers(in_channels=10, plans=20, kernel_size=5, stride=1)
        self.layer3 = self._make_layers(in_channels=20, plans=40, kernel_size=3, stride=1)
        self.layer4 = self._make_layers(in_channels=40, plans=64, kernel_size=5)
        if in_chanel == 1:
            self.fc1 = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(256, 680)),
                ('ELU', nn.ELU()),
                ('Dropout', nn.Dropout(p=0.5))]
            ))
        else:
            self.fc1 = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(1024, 680)),
                ('ELU', nn.ELU()),
                ('Dropout', nn.Dropout(p=0.5))]
            ))
        self.fc2 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(680, 280)),
            ('ELU', nn.ELU()),
            ('Dropout', nn.Dropout(p=0.5))]
        ))
        self.fc3 = nn.Linear(280, num_classes)
        self.last = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.last(out)
        return out

    def _make_layers(self, in_channels, plans, kernel_size=5, stride=None):
        layers = []
        layers += [('Conv', nn.Conv2d(in_channels, plans, kernel_size=kernel_size, stride=1)),
                   ('BatchNorm', nn.BatchNorm2d(plans)),
                   ('ElU', nn.ELU(inplace=True)),
                   ('MaxPool', nn.MaxPool2d(kernel_size=3, stride=stride))]

        return nn.Sequential(OrderedDict(layers))