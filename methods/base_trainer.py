import copy
import os
import pickle
from sklearn import metrics

import ipdb
import numpy as np
import torch
import torch.nn.functional as F
from torch import cuda

from data.utils import Mode
from dataloaders import FastDataLoader, InfiniteDataLoader
from methods.utils import prepare_data, forward_pass, plot_accuracy

device = 'cuda' if cuda.is_available() else 'cpu'

def collate_fn_mimic(batch):
    codes = [item[0][0] for item in batch]
    types = [item[0][1] for item in batch]
    target = [item[1] for item in batch]
    if len(batch[0]) == 2:
        return [(codes, types), target]
    else:
        groupid = torch.cat([item[2] for item in batch], dim=0).unsqueeze(1)
        return [(codes, types), target, groupid]

def rev_min_max_func(x, arr):
    max_val = np.max(arr)
    min_val = np.min(arr)
    og_val = (x *(max_val - min_val)) + min_val
    return og_val


class BaseTrainer:
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        # Dataset settings
        self.train_dataset = dataset
        self.train_dataset.mode = Mode.TRAIN
        self.eval_dataset = copy.deepcopy(dataset)
        self.eval_dataset.mode = Mode.TEST_OOD
        self.num_classes = dataset.num_classes
        self.num_tasks = dataset.num_tasks
        self.collate_fn = collate_fn_mimic if args.dataset == 'mimic' else None

        # Training hyperparameters
        self.args = args
        self.train_update_iter = args.train_update_iter
        self.lisa = args.lisa
        self.mixup = args.mixup
        self.cut_mix = args.cut_mix
        self.mix_alpha = args.mix_alpha
        self.mini_batch_size = args.mini_batch_size
        self.num_workers = args.num_workers
        self.test_update_iter = args.test_update_iter
        self.test_batch_size = args.test_batch_size
        self.base_trainer_str = self.get_base_trainer_str()

        # Evaluation and metrics
        self.split_time = args.split_time
        self.eval_next_timesteps = args.eval_next_timesteps
        self.eval_worst_time = args.eval_worst_time
        self.task_accuracies = {}
        self.worst_time_accuracies = {}
        self.best_time_accuracies = {}
        self.eval_metric = self.args.eval_metric

    def __str__(self):
        pass

    def get_base_trainer_str(self):
        base_trainer_str = f'train_update_iter={self.train_update_iter}-lr={self.args.lr}-' \
                                f'mini_batch_size={self.args.mini_batch_size}-seed={self.args.random_seed}'
        if self.args.lisa:
            base_trainer_str += f'-lisa-mix_alpha={self.mix_alpha}'
        elif self.mixup:
            base_trainer_str += f'-mixup-mix_alpha={self.mix_alpha}'
        if self.cut_mix:
            base_trainer_str += f'-cut_mix'
        base_trainer_str += f'-offline'
        return base_trainer_str

    def train_step(self, dataloader):
        self.network.train()
        loss_all = []

        for step, (x, y) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                           self.cut_mix, self.mix_alpha)
            loss_all.append(loss.item())

            # if step % 200 == 0 and step != 0:
            #     print(f'step: {step}, loss: {np.mean(loss_all)}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step == self.train_update_iter:
                if self.scheduler is not None:
                    self.scheduler.step()
                break

    def train_online(self):
        for i, t in enumerate(self.train_dataset.ENV[:-1]):
            if self.args.offline and t == self.split_time:
                break
            if self.args.load_model and self.model_path_exists(t):
                self.load_model(t)
            else:
                if self.args.lisa and i == self.args.lisa_start_time:
                    self.lisa = True
                self.train_dataset.update_current_timestamp(t)
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.collate_fn)
                self.train_step(train_dataloader)
                self.save_model(t)
                if self.args.method in ['coral', 'groupdro', 'irm', 'erm']:
                    self.train_dataset.update_historical(i + 1, data_del=True)

    def network_evaluation(self, test_time_dataloader):
        self.network.eval()
        pred_all = []
        y_all = []
        for _, sample in enumerate(test_time_dataloader):
            if len(sample) == 3:
                x, y, _ = sample
            else:
                x, y = sample
            x, y = prepare_data(x, y, str(self.eval_dataset))
            if self.args.regression and self.args.dataset == 'precipitation':
                y = y.to(torch.float)

            with torch.no_grad():
                logits_test = self.network(x)

                if self.args.dataset in ['drug']:
                    pred_test = logits_test.reshape(-1, )
                else:
                    pred_test = F.softmax(logits_test, dim=1).argmax(dim=1)

                pred_all = list(pred_all) + pred_test.detach().cpu().numpy().tolist()

                y_all = list(y_all) + y.cpu().numpy().tolist()

        if self.args.dataset == 'drug':
            from tdc import Evaluator
            evaluator = Evaluator(name='PCC')
            acc = evaluator(y_all, pred_all)
        elif self.args.regression:
            logits_test = np.array(logits_test.squeeze(1).detach().cpu().numpy())
            y = y.detach().cpu().numpy()

            if self.eval_metric == 'mae':
                acc = metrics.mean_absolute_error(y, logits_test)
            else:
                acc = metrics.mean_squared_error(y, logits_test, squared=False)
        else:
            pred_all = np.array(pred_all)
            y_all = np.array(y_all)
            if self.args.dataset == 'precipitation':
                y_all = y_all.squeeze()
            if self.args.dataset == 'precipitation' and self.eval_metric == 'f1':
                acc = metrics.f1_score(y_all, pred_all, average='macro')
            elif self.args.dataset == 'mimic' and self.args.prediction_type in ['mortality']:
                acc = metrics.roc_auc_score(y_all, pred_all)
            else:
                correct = (pred_all == y_all).sum().item()
                acc = correct / float(y_all.shape[0])

        self.network.train()

        return acc

    def evaluate(self, start):
        self.network.eval()
        accs = []
        for i in range(start, min(start + self.eval_next_timesteps, len(self.eval_dataset.ENV))):
            test_time = self.eval_dataset.ENV[i]
            self.eval_dataset.update_current_timestamp(test_time)
            test_time_dataloader = FastDataLoader(dataset=self.eval_dataset, batch_size=self.mini_batch_size,
                                                  num_workers=self.num_workers, collate_fn=self.collate_fn)
            acc = self.network_evaluation(test_time_dataloader)
            accs.append(acc)

        avg_acc = np.mean(accs)
        worst_acc = np.min(accs)
        best_acc = np.max(accs)

        print(
            f'current task is {start - 1}, eval next {self.eval_next_timesteps} timesteps, avg {self.eval_metric}'
            f' is {avg_acc}, worst {self.eval_metric} is {worst_acc}, best {self.eval_metric} is {best_acc}')
        print('current task is, all accs ', accs)
        self.network.train()

        return avg_acc, worst_acc, best_acc

    def evaluate_online(self):
        end = len(self.eval_dataset.ENV) - self.eval_next_timesteps
        for i, t in enumerate(self.eval_dataset.ENV[:end]):
            self.load_model(t)
            avg_acc, worst_acc, best_acc = self.evaluate(i + 1)
            self.task_accuracies[t] = avg_acc
            self.worst_time_accuracies[t] = worst_acc
            self.best_time_accuracies[t] = best_acc

    def evaluate_offline(self):
        timesteps = self.train_dataset.ENV
        acc_all = []
        for i, t in enumerate(timesteps):
            if t < self.split_time:
                # Collate test ID data from all time steps 1, ..., m
                self.train_dataset.mode = Mode.TEST_ID
                self.train_dataset.update_current_timestamp(t)
                self.train_dataset.update_historical(i + 1, data_del=True)
            elif t == self.split_time:
                # Evaluate in-distribution
                self.train_dataset.mode = Mode.TEST_ID
                self.train_dataset.update_current_timestamp(t)
                test_id_dataloader = FastDataLoader(dataset=self.train_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.collate_fn)
                acc = self.network_evaluation(test_id_dataloader)
                print('ID accuracy:', acc)
            else:
                # Evaluate out-of-distribution
                self.eval_dataset.mode = Mode.TEST_OOD
                self.eval_dataset.update_current_timestamp(t)
                test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.collate_fn)
                acc = self.network_evaluation(test_ood_dataloader)
                print(f'time is {t}, accuracy is {acc}')

                acc_all.append(acc)

        print('OOD avg accuracy:', np.mean(acc_all))
        print('OOD worst accuracy:', np.min(acc_all))
        print('all OOD accuracies', acc_all)

    def run_offline(self):
        if self.args.method in ['agem', 'ewc', 'ft', 'si']:
            self.train_online()
            self.evaluate_offline()
        else:
            timesteps = self.train_dataset.ENV
            acc_all = []
            for i, t in enumerate(timesteps):
                if t < self.split_time:
                    # Collate data from all time steps 1, ..., m-1
                    self.train_dataset.mode = Mode.TRAIN
                    self.train_dataset.update_current_timestamp(t)
                    self.train_dataset.update_historical(i + 1)
                    self.train_dataset.mode = Mode.TEST_ID
                    self.train_dataset.update_current_timestamp(t)
                    self.train_dataset.update_historical(i + 1, data_del=True)
                elif t == self.split_time:
                    # Collate data from all time steps 1, ..., m
                    self.train_dataset.mode = Mode.TRAIN
                    self.train_dataset.update_current_timestamp(t)

                    # Train
                    train_id_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                            batch_size=self.mini_batch_size,
                                                            num_workers=self.num_workers, collate_fn=self.collate_fn)

                    if self.args.load_model:
                        self.load_model(t)
                    else:
                        self.train_step(train_id_dataloader)
                        self.save_model(t)

                    # Evaluate in-distribution
                    self.train_dataset.mode = Mode.TEST_ID
                    self.train_dataset.update_current_timestamp(t)
                    test_id_dataloader = FastDataLoader(dataset=self.train_dataset,
                                                        batch_size=self.mini_batch_size,
                                                        num_workers=self.num_workers, collate_fn=self.collate_fn)
                    acc = self.network_evaluation(test_id_dataloader)
                    print('ID accuracy:', acc)
                else:
                    # Evaluate out-of-distribution
                    self.eval_dataset.mode = Mode.TEST_OOD
                    self.eval_dataset.update_current_timestamp(t)
                    test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                        batch_size=self.mini_batch_size,
                                                        num_workers=self.num_workers, collate_fn=self.collate_fn)
                    acc = self.network_evaluation(test_ood_dataloader)
                    print(f'time is {t}, accuracy is {acc}')

                    acc_all.append(acc)

            print('OOD avg accuracy:', np.mean(acc_all))
            print('OOD worst accuracy:', np.min(acc_all))
            print('all OOD accuracies', acc_all)

    def run_offline_task_difficulty(self):
        timesteps = self.train_dataset.ENV
        acc_all = []
        l=len(timesteps)

        for i, t in enumerate(timesteps):
            if i == l-1:
                self.train_dataset.mode = Mode.TRAIN
                self.train_dataset.update_current_timestamp(t)

                train_id_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                         batch_size=self.mini_batch_size,
                                                         num_workers=self.num_workers, collate_fn=self.collate_fn)

                if self.args.load_model:
                    self.load_model(t)
                else:
                    self.train_step(train_id_dataloader)
                    self.save_model(t)
            else:
                self.train_dataset.mode = Mode.TRAIN
                self.train_dataset.update_current_timestamp(t)
                self.train_dataset.update_historical(i + 1)
        for i, t in enumerate(timesteps):
            if t <= self.split_time:
                continue
            else:
                # Evaluate out-of-distribution
                self.eval_dataset.mode = Mode.TEST_ID
                self.eval_dataset.update_current_timestamp(t)
                test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.collate_fn)
                acc = self.network_evaluation(test_ood_dataloader)
                print(f'time is {t}, accuracy is {acc}')

                acc_all.append(acc)

        print('OOD avg accuracy:', np.mean(acc_all))
        print('OOD worst accuracy:', np.min(acc_all))
        print('all OOD accuracies', acc_all)

    def run_online(self):
        if not self.args.load_model:
            self.train_online()
        self.evaluate_online()

    def run(self):
        torch.cuda.empty_cache()
        import time
        start_time = time.time()
        if self.args.offline:
            if self.args.difficulty:
                self.run_offline_task_difficulty()
            else:
                self.run_offline()
        else:
            self.run_online()
        print('run time:', time.time() - start_time)

    def get_model_path(self, timestep):
        model_str = f'{str(self.train_dataset)}_{str(self)}_time={timestep}'
        path = os.path.join(self.args.log_dir, model_str)
        return path

    def model_path_exists(self, timestep):
        return os.path.exists(self.get_model_path(timestep))

    def save_model(self, timestep):
        path = self.get_model_path(timestep)
        torch.save(self.network.state_dict(), path)

    def load_model(self, timestep):
        path = self.get_model_path(timestep)
        self.network.load_state_dict(torch.load(path), strict=False)

    def save(self) -> None:
        '''
        Saves and plots task accuracies.
        '''
        print('avg worst time accuracy:', np.mean(list(self.worst_time_accuracies.values())))
        print('avg task accuracy:', np.mean(list(self.task_accuracies.values())))
        print('avg best time accuracy:', np.mean(list(self.best_time_accuracies.values())))

        metrics = [self.worst_time_accuracies, self.task_accuracies, self.best_time_accuracies]
        pickle.dump(metrics, open(self.results_file[:-4], 'wb'))

        plot_title1 = f'{str(self)}-worst_time'
        plot_title2 = f'{str(self)}-avg_time'
        plot_accuracy(title=plot_title1, tasks=np.array(list(self.worst_time_accuracies.keys())),
                      accuracies=list(self.worst_time_accuracies.values()), save_path=f'{self.results_file[:-4]}.png')
        plot_accuracy(title=plot_title2, tasks=np.array(list(self.task_accuracies.keys())),
                      accuracies=list(self.task_accuracies.values()), save_path=f'{self.results_file[:-4]}.png')