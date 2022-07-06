import os

import numpy as np

from data.utils import Mode
from dataloaders import FastDataLoader, InfiniteDataLoader
from methods.base_trainer import BaseTrainer


class TOE(BaseTrainer):
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        if self.args.lisa:
            return f'TOE-LISA-no-domainid-{self.base_trainer_str}'
        elif self.args.mixup:
            return f'TOE-Mixup-no-domainid-{self.base_trainer_str}'
        return f'TOE-{self.base_trainer_str}'