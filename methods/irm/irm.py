'''
Adapted from https://github.com/p-lambda/wilds/blob/main/examples/algorithms/IRM.py.
'''

import os
import numpy as np
import torch

from dataloaders import InfiniteDataLoader
from methods.base_trainer import BaseTrainer
from methods.groupdro.loss import LossComputer
from methods.utils import prepare_data, forward_pass, split_into_groups


class IRM(BaseTrainer):

    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.update_count = 0
        self.irm_lambda = args.irm_lambda
        self.irm_penalty_anneal_iters = args.irm_penalty_anneal_iters
        self.scale = torch.tensor(1.).requires_grad_()
        dataset.current_time = dataset.ENV[0]
        self.loss_computer = LossComputer(self.train_dataset, criterion, is_robust=True)
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        return f'IRM-irm_lambda={self.irm_lambda}-irm_penalty_anneal_iters={self.irm_penalty_anneal_iters}' \
               f'-{self.base_trainer_str}'

    def irm_penalty(self, losses):
        grad_1 = torch.autograd.grad(losses[0::2].mean(), [self.scale], create_graph=True)[0]
        grad_2 = torch.autograd.grad(losses[1::2].mean(), [self.scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def train_step(self, dataloader):
        self.network.train()
        loss_all = []
        for step, (x, y, g) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            g = g.squeeze(1).cuda()

            self.network.zero_grad()
            unique_groups, group_indices, _ = split_into_groups(g)
            n_groups_per_batch = unique_groups.numel()
            avg_loss = 0.
            penalty = 0.
            _, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                        self.cut_mix, self.mix_alpha)
            for i_group in group_indices:
                group_losses = self.criterion(self.scale * logits[i_group], y[i_group])
                if group_losses.numel() > 0:
                    avg_loss += group_losses.mean()
                penalty += self.irm_penalty(group_losses)
            avg_loss /= n_groups_per_batch
            penalty /= n_groups_per_batch

            if self.update_count >= self.irm_penalty_anneal_iters:
                penalty_weight = self.irm_lambda
            else:
                penalty_weight = 1.0

            loss = avg_loss + penalty * penalty_weight
            loss_all.append(loss.item())
            loss.backward()
            self.optimizer.step()

            if step % 100 == 0 and step != 0:
                print('step: {}, loss: {}'.format(step, np.mean(loss_all)))

            if step == self.train_update_iter:
                if self.scheduler is not None:
                    self.scheduler.step()
                break

    def run_online(self):
        for i, t in enumerate(self.train_dataset.ENV[:-1]):
            if self.args.load_model:
                self.load_model(t)
            else:
                self.train_dataset.update_current_timestamp(t)
                self.train_dataset.train = True
                time_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                 num_workers=self.num_workers, collate_fn=self.collate_fn)
                self.train_step(time_dataloader)
                self.save_model(t)
                self.train_dataset.update_historical(i + 1)

            self.eval_dataset.train = False
            avg_acc, worst_acc, best_acc = self.evaluate(i + 1)
            self.task_accuracies[t] = avg_acc
            self.worst_time_accuracies[t] = worst_acc
            self.best_time_accuracies[t] = best_acc

        self.save()