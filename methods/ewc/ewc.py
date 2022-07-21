'''
Adapted from https://github.com/GMvandeVen/continual-learning.
'''

import copy
import os

import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from methods.base_trainer import BaseTrainer
from methods.utils import prepare_data, forward_pass


def get_data_loader(dataset, batch_size, cuda=False, collate_fn=None, drop_last=False, augment=False):
    '''Return <DataLoader>-object for the provided <DataSet>-object [dataset].'''

    # If requested, make copy of original dataset to add augmenting transform (without altering original dataset)
    if augment:
        dataset_ = copy.deepcopy(dataset)
        dataset_.transform = transforms.Compose([dataset.transform, *data.AVAILABLE_TRANSFORMS['augment']])
    else:
        dataset_ = dataset

    rand_sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=min(batch_size, len(dataset_)))
    # Create and return the <DataLoader>-object
    return DataLoader(
        dataset_, sampler=rand_sampler,
        collate_fn=(collate_fn or default_collate), drop_last=drop_last,
        **({'num_workers': 2, 'pin_memory': True} if cuda else {})
    )


class EWC(BaseTrainer):
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.ewc_lambda = args.ewc_lambda   #-> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = args.gamma             #-> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = args.online           #-> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = args.fisher_n       #-> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = args.emp_FI           #-> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = 0             #-> keeps track of number of quadratic loss terms (for "offline EWC")
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        str_all = f'EWC-lambda={self.ewc_lambda}-gamma={self.gamma}-online={self.online}-fisher_n={self.fisher_n}' \
                  f'-emp_FI={self.emp_FI}-{self.base_trainer_str}'
        return str_all

    def _device(self):
        return next(self.network.parameters()).device

    def _is_on_cuda(self):
        return next(self.network.parameters()).is_cuda

    def estimate_fisher(self):
        '''After completing training on a task, estimate diagonal of Fisher Information matrix.
        [dataset]:          <DataSet> to be used to estimate FI-matrix'''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        self.network.eval()

        # Create data-loader to give batches of size 1
        # This samples the current time step
        data_loader = get_data_loader(self.train_dataset, batch_size=self.mini_batch_size, collate_fn=self.collate_fn)

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        ind = 0
        for index, (x, y) in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            # run forward pass of model
            x, y = prepare_data(x, y, str(self.train_dataset))
            loss, output, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup, self.cut_mix, self.mix_alpha)
            if self.emp_FI:
                # -use provided label to calculate loglikelihood --> "empirical Fisher":
                label = torch.LongTensor([y]) if type(y) == int else y
                label = label.to(self._device())
                negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
            else:
                # calculate negative log-likelihood
                if str(self.train_dataset) == 'drug':
                    negloglikelihood = F.mse_loss(output, y)
                elif self.args.regression:
                    negloglikelihood = F.mse_loss(output, y)
                else:
                    label = output.max(1)[1]
                    negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            # Calculate gradient of negative loglikelihood
            self.network.zero_grad()
            negloglikelihood.backward()

            # Square gradients and keep running sum
            for n, p in self.network.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2
            ind = index

        # Normalize by sample size used for estimation
        # print('num samples', (ind + 1))
        est_fisher_info = {n: p / (ind + 1) for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.network.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count + 1),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and self.EWC_task_count == 1:
                    existing_values = getattr(self.network, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.network.register_buffer(
                    '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count + 1),
                    est_fisher_info[n])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        # Set model back to its initial mode
        self.network.train()

    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count > 0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.EWC_task_count + 1):
                for n, p in self.network.named_parameters():
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(self.network, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
                        fisher = getattr(self.network, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma * fisher if self.online else fisher
                        # Calculate EWC-loss
                        losses.append((fisher * (p - mean) ** 2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1. / 2) * sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=self._device())

    def train_step(self, dataloader):
        self.network.train()
        loss_all = []
        for step, (x, y) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup, self.cut_mix, self.mix_alpha)
            loss = loss + self.ewc_lambda * self.ewc_loss()
            loss_all.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step == self.train_update_iter:
                if self.scheduler is not None:
                    self.scheduler.step()
                self.estimate_fisher()
                break
