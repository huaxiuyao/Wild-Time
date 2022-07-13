import os
from methods.base_trainer import BaseTrainer


class ERM(BaseTrainer):
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        if self.args.lisa:
            return f'ERM-LISA-no-domainid-{self.base_trainer_str}'
        elif self.args.mixup:
            return f'ERM-Mixup-no-domainid-{self.base_trainer_str}'
        return f'ERM-{self.base_trainer_str}'