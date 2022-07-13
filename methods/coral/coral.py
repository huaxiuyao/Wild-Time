'''
Adapted from https://github.com/p-lambda/wilds/blob/main/examples/algorithms/deepCORAL.py.
'''

import ipdb
import numpy as np
import os
import torch

from dataloaders import FastDataLoader, InfiniteDataLoader
from data.utils import Mode
from methods.base_trainer import BaseTrainer
from methods.utils import prepare_data, forward_pass, split_into_groups

class DeepCORAL(BaseTrainer):

    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.coral_lambda = args.coral_lambda
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        return f'DeepCORAL-coral_lambda={self.coral_lambda}-{self.base_trainer_str}'

    def coral_penalty(self, x, y):
        if x.dim() > 2:
            # featurizers output Tensors of size (batch_size, ..., feature dimensionality).
            # we flatten to Tensors of size (*, feature dimensionality)
            x = x.view(-1, x.size(-1))
            y = y.view(-1, y.size(-1))

        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def train_step(self, dataloader):
        self.network.train()
        loss_all = []
        for step, (x, y, g) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            g = g.squeeze(1).cuda()
            unique_groups, group_indices, _ = split_into_groups(g)
            n_groups_per_batch = unique_groups.numel()

            classification_loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup, self.cut_mix, self.mix_alpha)
            coral_loss = torch.zeros(1).cuda()
            for i_group in range(n_groups_per_batch):
                for j_group in range(i_group + 1, n_groups_per_batch):
                    coral_loss += self.coral_penalty(logits[group_indices[i_group]].squeeze(0), logits[group_indices[j_group]].squeeze(0))
            if n_groups_per_batch > 1:
                coral_loss /= (n_groups_per_batch * (n_groups_per_batch-1) / 2) # get the mean penalty

            loss = classification_loss + self.coral_lambda * coral_loss
            loss_all.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 200 == 0 and step != 0:
                print('step: {}, loss: {}'.format(step, np.mean(loss_all)))

            if step == self.train_update_iter:
                if self.scheduler is not None:
                    self.scheduler.step()
                break