'''
Adapted from https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py.
'''

import numpy as np
import torch

def mixup_data(x, y, mix_alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if mix_alpha > 0:
        lam = np.random.beta(mix_alpha, mix_alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)