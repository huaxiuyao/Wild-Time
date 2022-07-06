'''
Adapted from https://github.com/GMvandeVen/continual-learning.
'''

import numpy as np
import os
import torch.utils.data

from dataloaders import FastDataLoader, InfiniteDataLoader
from methods.base_trainer import BaseTrainer
from methods.utils import prepare_data, forward_pass


class SI(BaseTrainer):
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.si_c = args.si_c            #-> hyperparam: how strong to weigh SI-loss ("regularisation strength")
        self.epsilon = args.epsilon      #-> dampening parameter: bounds 'omega' when squared parameter-change goes to 0
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        str_all = f'SI-si_c={self.si_c}-epsilon={self.epsilon}-{self.base_trainer_str}'
        return str_all

    def _device(self):
        return next(self.network.parameters()).device

    def _is_on_cuda(self):
        return next(self.network.parameters()).is_cuda

    def update_omega(self, W, epsilon):
        '''After completing training on a task, update the per-parameter regularization strength.
        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

            # Find/calculate new values for quadratic penalty on parameters
            p_prev = getattr(self.network, '{}_SI_prev_task'.format(n))
            p_current = p.detach().clone()
            p_change = p_current - p_prev
            omega_add = W[n] / (p_change ** 2 + epsilon)
            try:
                omega = getattr(self.network, '{}_SI_omega'.format(n))
            except AttributeError:
                omega = p.detach().clone().zero_()
            omega_new = omega + omega_add

            # Store these new values in the model
            self.network.register_buffer('{}_SI_prev_task'.format(n), p_current)
            self.network.register_buffer('{}_SI_omega'.format(n), omega_new)

    def surrogate_loss(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for n, p in self.network.named_parameters():
                # print(n)
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self.network, '{}_SI_prev_task'.format(n))
                    omega = getattr(self.network, '{}_SI_omega'.format(n))
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p - prev_values) ** 2).sum())
            return sum(losses)

        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())

    def train_step(self, dataloader):
        # Prepare <dicts> to store running importance estimates and parameter-values before update
        W = {}
        p_old = {}
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()

        self.network.train()
        loss_all = []

        for step, (x, y) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                           self.cut_mix, self.mix_alpha)
            loss = loss + self.si_c * self.surrogate_loss()
            loss_all.append(loss.item())

            # if step % 50 == 0:
                # print('step: {}, loss: {}'.format(step, np.mean(loss_all)))
                # print(self.surrogate_loss())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step == self.train_update_iter:
                if self.scheduler is not None:
                    self.scheduler.step()

                # Update running parameter importance estimates in W
                for n, p in self.network.named_parameters():
                    if p.requires_grad:
                        # n = "network." + n
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            W[n].add_(-p.grad * (p.detach() - p_old[n]))
                        p_old[n] = p.detach().clone()
                self.update_omega(W, self.epsilon)
                break

    def train_online(self):
        # Register starting param-values (needed for "intelligent synapses").
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.network.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())

        for i, t in enumerate(self.train_dataset.ENV[:-1]):
            if self.args.offline and t == self.split_time:
                break
            if self.args.load_model:
                self.load_model(t)
            else:
                self.train_dataset.update_current_timestamp(t)
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                 num_workers=self.num_workers, collate_fn=self.collate_fn)
                self.train_step(train_dataloader)
                self.save_model(t)

    def run_online(self):
        # Register starting param-values (needed for "intelligent synapses").
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.network.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())

        end = len(self.eval_dataset.ENV) - self.eval_next_timesteps
        for i, t in enumerate(self.train_dataset.ENV[:end]):
            if self.args.load_model:
                self.load_model(t)
            else:
                self.train_dataset.update_current_timestamp(t)
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                      batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.collate_fn)
                self.train_step(train_dataloader)
                self.save_model(t)

            avg_acc, worst_acc, best_acc = self.evaluate(i + 1)
            self.task_accuracies[t] = avg_acc
            self.worst_time_accuracies[t] = worst_acc
            self.best_time_accuracies[t] = best_acc

            self.save()
