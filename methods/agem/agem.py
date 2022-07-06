'''
Adapted from https://github.com/aimagelab/mammoth.
'''

import os

import numpy as np
import torch

from methods.agem.buffer import Buffer
from methods.base_trainer import BaseTrainer
from methods.utils import prepare_data, forward_pass


def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class AGEM(BaseTrainer):

    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)

        self.buffer = Buffer(self.args.buffer_size, self._device())
        self.grad_dims = []
        for param in self.network.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self._device())
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self._device())
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        return f'AGEM-buffer_size={self.args.buffer_size}-{self.base_trainer_str}'

    def _device(self):
        return next(self.network.parameters()).device

    def end_task(self, dataloader):
        sample = next(iter(dataloader))
        cur_x, cur_y = sample
        cur_x, cur_y = prepare_data(cur_x, cur_y, str(self.train_dataset))
        if str(self.train_dataset) == 'weather':
            cur_x = [cur_x['categorical'], cur_x['continuous']]
        elif str(self.train_dataset) == 'mimic':
            cur_x = [(code, type) for (code, type) in zip(cur_x[0], cur_x[1])]
        self.buffer.add_data(
            examples=cur_x,
            labels=cur_y
        )

    def train_step(self, dataloader):
        self.network.train()
        loss_all = []
        for step, (x, y) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))

            self.network.zero_grad()
            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                           self.cut_mix, self.mix_alpha)
            loss_all.append(loss.item())
            loss.backward()

            if not self.buffer.is_empty():
                store_grad(self.network.parameters, self.grad_xy, self.grad_dims)

                buf_data = self.buffer.get_data(self.mini_batch_size, transform=None)
                if str(self.train_dataset) == 'mimic':
                    buf_inputs_codes, buf_inputs_types, buf_labels = [], [], []
                    for elt in buf_data:
                        if torch.is_tensor(elt):
                            buf_labels.append(elt)
                        else:
                            buf_inputs_codes.append(elt[0][0])
                            buf_inputs_types.append(elt[0][1])

                    buf_inputs = (buf_inputs_codes, buf_inputs_types)
                elif len(buf_data) > 2:
                    # Drug dataset has 2 inputs
                    buf_inputs = [buf_data[0], buf_data[1]]
                    buf_labels = buf_data[2]
                else:
                    buf_inputs, buf_labels = buf_data
                if str(self.train_dataset) == 'weather':
                    categorical = buf_inputs[0].cuda()
                    continuous = buf_inputs[1].cuda()
                    buf_inputs = {}
                    buf_inputs['categorical'] = categorical
                    buf_inputs['continuous'] = continuous
                    buf_labels = buf_labels.cuda()
                else:
                    buf_inputs, buf_labels = prepare_data(buf_inputs, buf_labels, str(self.train_dataset))
                self.network.zero_grad()
                penalty, buff_outputs, buf_labels = forward_pass(buf_inputs, buf_labels, self.train_dataset, self.network,
                                                                 self.criterion, self.lisa, self.mixup, self.cut_mix, self.mix_alpha)
                penalty.backward()
                store_grad(self.network.parameters, self.grad_er, self.grad_dims)

                dot_prod = torch.dot(self.grad_xy, self.grad_er)
                if dot_prod.item() < 0:
                    g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                    overwrite_grad(self.network.parameters, g_tilde, self.grad_dims)
                else:
                    overwrite_grad(self.network.parameters, self.grad_xy, self.grad_dims)

            self.optimizer.step()

            if step % 100 == 0 and step != 0:
                print('step: {}, loss: {}'.format(step, np.mean(loss_all)))

            if step == self.train_update_iter:
                if self.scheduler is not None:
                    self.scheduler.step()
                self.end_task(dataloader)
                break
