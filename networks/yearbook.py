import torch
import torch.nn as nn
import torch.nn.functional as F


class YearbookNetwork(nn.Module):
    def __init__(self, args, num_input_channels, num_classes):
        super(YearbookNetwork, self).__init__()
        self.args = args
        self.net = nn.Sequential(self.conv_block(num_input_channels, 32), self.conv_block(32, 32),
                                 self.conv_block(32, 32), self.conv_block(32, 32))
        self.hid_dim = 32

        self.logits = nn.Linear(32, num_classes)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.net(x)

        x = torch.mean(x, dim=(2, 3))

        return self.logits(x)