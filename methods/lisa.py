import ipdb
import torch
import numpy as np
import torch.nn.functional as F

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mix_up(x1, x2, y1, y2, mix_alpha):
    # y1, y2 should be one-hot labels, which means the shape of y1 and y2 should be [bsz, n_classes]
    # length = min(len(x1), len(x2))
    # x1 = x1[:length]
    # x2 = x2[:length]
    # y1 = y1[:length]
    # y2 = y2[:length]

    n_classes = y1.shape[1]
    bsz = len(x1)
    l = np.random.beta(mix_alpha, mix_alpha, [bsz, 1])
    if len(x1.shape) == 4:
        l_x = np.tile(l[..., None, None], (1, *x1.shape[1:]))
    else:
        l_x = np.tile(l, (1, *x1.shape[1:]))
    l_y = np.tile(l, [1, n_classes])

    # mixed_input = l * x + (1 - l) * x2
    mixed_x = torch.tensor(l_x, dtype=torch.float32).to(x1.device) * x1 + torch.tensor(1-l_x, dtype=torch.float32).to(x2.device) * x2
    mixed_y = torch.tensor(l_y, dtype=torch.float32).to(y1.device) * y1 + torch.tensor(1-l_y, dtype=torch.float32).to(y2.device) * y2

    return mixed_x, mixed_y


def cut_mix_up(x1, x2, y1, y2, mix_alpha):

    # input = torch.cat([x1,x2])
    # target = torch.cat([y1,y2])
    #
    # rand_index = torch.cat([torch.arange(len(y2)) + len(y1), torch.arange(len(y1))])

    lam = np.random.beta(mix_alpha, mix_alpha)
    # target_a = target
    # target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x1.shape, lam)
    x1[:, :, bbx1:bbx2, bby1:bby2] = x2[:, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x1.shape[-1] * x1.shape[-2]))

    return x1, lam*y1 + (1-lam)*y2

def lisa(features, labels, dataset, mix_alpha, num_classes, time_idx, cut_mix=False, embedding=None):
    '''
    Performs LISA mnist augmentation.
    :param data: train mnist, Torch Tensor of shape (batch_size, 3, 28, 28)
    :param labels: train labels, Torch Tensor of shape (batch_size)
    :param mini_batch_size: int size of minibatch B1
    :param num_classes: int number of classes
    :return:
        x_mix: Torch tensor of shape
        y_mix: Torch tensor of shape
    '''
    # Sample a minibatch, B1, of train_data
    X1 = features
    if str(dataset) != 'weather':
        Y1 = labels.unsqueeze(1)
    else:
        Y1 = labels
    X2 = torch.zeros(X1.shape).cuda()
    Y2 = torch.zeros(Y1.shape, dtype=torch.long).cuda()

    class_ids = torch.unique(Y1)
    # Intra-label LISA: same label, different domains
    for eachclass in class_ids:
        sel_idx = torch.nonzero(Y1 == eachclass)[:,0]
        if str(dataset) == 'drug':
            sel_x2, sel_y2 = dataset.get_lisa_new_sample(time_idx, eachclass.item(), sel_idx.shape[0])
        else:
            sel_x2, sel_y2 = dataset.get_lisa_new_sample(time_idx, eachclass.item(), sel_idx.shape[0])
        if sel_x2 is None or sel_y2 is None:
            return X1, Y1
        if embedding is not None:
            sel_x2 = embedding(sel_x2)
        X2[sel_idx] = sel_x2
        Y2[sel_idx] = sel_y2
    Y1 = torch.cat([F.one_hot(Y1[i], num_classes) for i in range(len(Y1))], dim=0)
    Y2 = torch.cat([F.one_hot(Y2[i], num_classes) for i in range(len(Y2))], dim=0)
    if cut_mix:
        x_mix, y_mix = cut_mix_up(X1, X2, Y1, Y2, mix_alpha)
    else:
        x_mix, y_mix = mix_up(X1, X2, Y1, Y2, mix_alpha)
    return x_mix, y_mix

# def lisa(features, labels, dataset, mix_alpha, num_classes, time_idx, cut_mix=False, embedding=None):
#     '''
#     Performs LISA mnist augmentation.
#     :param data: train mnist, Torch Tensor of shape (batch_size, 3, 28, 28)
#     :param labels: train labels, Torch Tensor of shape (batch_size)
#     :param mini_batch_size: int size of minibatch B1
#     :param num_classes: int number of classes
#     :return:
#         x_mix: Torch tensor of shape
#         y_mix: Torch tensor of shape
#     '''
#     # Sample a minibatch, B1, of train_data
#
#     X1 = features
#     Y1 = labels
#     X2 = torch.zeros(X1.shape).cuda()
#     Y2 = torch.zeros(Y1.shape, dtype=torch.long).cuda()
#
#     class_ids = torch.unique(Y1)
#     # Intra-label LISA: same label, different domains
#     for eachclass in class_ids:
#         sel_idx = torch.nonzero(Y1 == eachclass)[:,0]
#         sel_x2, sel_y2 = dataset.get_lisa_new_sample(time_idx, eachclass.item(), sel_idx.shape[0])
#         if sel_x2 is None or sel_y2 is None:
#             return X1, Y1
#         if embedding is not None:
#             sel_x2 = embedding(sel_x2)
#         X2[sel_idx] = sel_x2
#         Y2[sel_idx] = sel_y2
#
#     Y1 = torch.cat([F.one_hot(Y1[i], num_classes) for i in range(len(Y1))], dim=0)
#     Y2 = torch.cat([F.one_hot(Y2[i], num_classes) for i in range(len(Y2))], dim=0)
#     if cut_mix:
#         x_mix, y_mix = cut_mix_up(X1, X2, Y1, Y2, mix_alpha)
#     else:
#         x_mix, y_mix = mix_up(X1, X2, Y1, Y2, mix_alpha)
#
#     return x_mix, y_mix