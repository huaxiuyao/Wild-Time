import numpy as np
import os
import torch
import torch.nn.functional as F

from dataloaders import FastDataLoader, InfiniteDataLoader
from data.utils import Mode
from methods.utils import prepare_data, forward_pass
from methods.lisa import lisa
from methods.groupdro.loss import LossComputer
from methods.base_trainer import BaseTrainer


class GroupDRO(BaseTrainer):
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.num_groups = args.num_groups
        self.group_size = args.group_size
        dataset.current_time = dataset.ENV[0]
        self.loss_computer = LossComputer(dataset, criterion, is_robust=True)
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        return f'GroupDRO-num_groups={self.num_groups}-group_size={self.group_size}-{self.base_trainer_str}'

    def train_step(self, dataloader):
        self.network.train()
        loss_all = []
        for step, (x, y, g) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            g = g.squeeze(1).cuda()
            _, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup, self.cut_mix, self.mix_alpha)
            if str(self.train_dataset) in ['drug']:
                logits = logits.unsqueeze(1)
            loss = self.loss_computer.loss(logits, y, g, is_training=True)
            loss_all.append(loss.item())

            # if step % 100 == 0 and step != 0:
            #     print('step: {}, loss: {}'.format(step, np.mean(loss_all)))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step == self.train_update_iter:
                if self.scheduler is not None:
                    self.scheduler.step()
                break