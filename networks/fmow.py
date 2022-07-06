from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121

IMG_HEIGHT = 224
NUM_CLASSES = 62

class FMoWNetwork(nn.Module):
    def __init__(self, weights=None):
        super(FMoWNetwork, self).__init__()
        self.num_classes = NUM_CLASSES
        self.enc = densenet121(pretrained=True).features
        self.classifier = nn.Linear(1024, self.num_classes)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward(self, x):
        features = self.enc(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out