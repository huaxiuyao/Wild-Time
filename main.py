import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch import cuda

from get_dataset import get_dataset
from methods.agem.agem import AGEM
from methods.coral.coral import DeepCORAL
from methods.erm.erm import ERM
from methods.ewc.ewc import EWC
from methods.ft.ft import FT
from methods.groupdro.groupdro import GroupDRO
from methods.irm.irm import IRM
from methods.si.si import SI
from networks.article import ArticleNetwork
from networks.drug import DTI_Encoder, DTI_Classifier
from networks.fmow import FMoWNetwork
from networks.mimic import Transformer
from networks.precipitation import PrecipitationNetwork
from networks.yearbook import YearbookNetwork

parser = argparse.ArgumentParser(description='Wild-Time')

parser.add_argument('--dataset', default='yearbook', choices=['arxiv', 'drug', 'huffpost', 'mimic', 'fmow', 'precipitation', 'yearbook'])
parser.add_argument('--regression', dest='regression', action='store_true', help='regression task for mimic datasets')
parser.add_argument('--prediction_type', type=str, help='MIMIC: "mortality" or "readmission", "precipitation"')
parser.add_argument('--method', default='ft', choices=['agem', 'coral', 'ensemble', 'ewc', 'ft', 'groupdro', 'irm', 'si', 'erm'])
parser.add_argument('--device', default=0, type=int, help='gpu id')
parser.add_argument('--random_seed', default=1, type=int, help='random seed number')

# Training hyperparameters
parser.add_argument('--train_update_iter', default=10, type=int, help='train update iter')
parser.add_argument('--lr', default=0.01, type=float, help='the base learning rate of the generator')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--mini_batch_size', type=int, default=32, help='mini batch size for SGD')
parser.add_argument('--test_update_iter', type=int, default=10, help='test-time adaptation iterations')
parser.add_argument('--test_batch_size', type=int, default=32, help='batch size for test-time adaptation')

# Evaluation
parser.add_argument('--offline', dest='offline', action='store_true', help='evaluate offline at a single time step split')
parser.add_argument('--difficulty', dest='difficulty', action='store_true', help='task difficulty')
parser.add_argument('--split_time', type=int, help='timestep to split ID vs OOD')
parser.add_argument('--eval_next_timesteps', default=1, type=int, help='number of future timesteps to evaluate on')
parser.add_argument('--eval_worst_time', dest='eval_worst_time', action='store_true', help='evaluate worst timestep accuracy')
parser.add_argument('--load_model', dest='load_model', action='store_true', help='load trained model for evaluation only')
parser.add_argument('--eval_metric', default='acc', choices=['acc', 'f1', 'rmse'])

# FT
parser.add_argument('--K', default=1, type=int, help='number of previous timesteps to finetune on')

# LISA and Mixup
parser.add_argument('--lisa', dest='lisa', action='store_true', help='train with LISA')
parser.add_argument('--lisa_intra_domain', dest='lisa_intra_domain', action='store_true', help='train with LISA intra domain')
parser.add_argument('--mixup', dest='mixup', action='store_true', help='train with vanilla mixup')
parser.add_argument('--lisa_start_time', type=int, default=0, help='lisa_start_time')
parser.add_argument('--mix_alpha', type=float, default=2.0, help='mix alpha for LISA')
parser.add_argument('--cut_mix', dest='cut_mix', action='store_true', help='use cut mix up')

# GroupDRO
parser.add_argument('--num_groups', type=int, default=4, help='number of windows for Group DRO')
parser.add_argument('--group_size', type=int, default=4, help='window size for Group DRO')

# EWC
parser.add_argument('--ewc_lambda', type=float, default=1.0, help='how strong to weigh EWC-loss ("regularisation strength")')
parser.add_argument('--gamma', type=float, default=1.0, help='decay-term for old tasks (contribution to quadratic term)')
parser.add_argument('--online', dest='online', action='store_true', help='"online" (=single quadratic term) or "offline" (=quadratic term per task) EWC')
parser.add_argument('--fisher_n', type=int, default=None, help='sample size for estimating FI-matrix (if "None", full pass over dataset)')
parser.add_argument('--emp_FI', dest='emp_FI', action='store_true', help='if True, use provided labels to calculate FI ("empirical FI"); else predicted labels')

# A-GEM
parser.add_argument('--buffer_size', type=int, default=100, help='buffer size for A-GEM')

# DeepCORAL
parser.add_argument('--coral_lambda', type=float, default=1.0, help='how strong to weigh CORAL loss')

# IRM
parser.add_argument('--irm_lambda', type=float, default=1.0, help='how strong to weigh IRM penalty loss')
parser.add_argument('--irm_penalty_anneal_iters', type=int, default=0, help='number of iterations after which we anneal IRM penalty loss')

# SI
parser.add_argument('--si_c', type=float, default=0.1, help='SI: regularisation strength')
parser.add_argument('--epsilon', type=float, default=0.001, help='dampening parameter: bounds "omega" when squared parameter-change goes to 0')

## Logging, saving, and testing options
parser.add_argument('--data_dir', default='./Data', type=str, help='directory for datasets.')
parser.add_argument('--log_dir', default='./checkpoints', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--results_dir', default='./results', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers in data generator')


args = parser.parse_args()
print(args)

random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.manual_seed(args.random_seed)

device = 'cuda' if cuda.is_available() else 'cpu'


if __name__ == '__main__':
    torch.cuda.empty_cache()

    if not os.path.isdir(args.data_dir):
        raise ValueError(f'Data directory {args.data_dir} does not exist!')
    if args.load_model and not os.path.isdir(args.log_dir):
        raise ValueError(f'Model checkpoint directory {args.log_dir} does not exist!')
    if not os.path.isdir(args.results_dir):
        raise ValueError(f'Results directory {args.results_dir} does not exist!')

    if args.method in ['groupdro', 'local', 'irm']:
        reduction = 'none'
    else:
        reduction = 'mean'

    # Datasets
    dataset = get_dataset(dataset=args.dataset, args=args)
    print(dataset.mode)

    # Criterions, backbone networks, optimizers, schedulers
    scheduler = None
    if args.dataset in ['yearbook']:
        criterion = nn.CrossEntropyLoss(reduction=reduction).cuda()
        network = YearbookNetwork(
            args, num_input_channels=3, num_classes=dataset.num_classes).cuda()
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.dataset in ['fmow']:
        criterion = nn.CrossEntropyLoss(reduction=reduction).cuda()
        network = FMoWNetwork().cuda()
        optimizer = torch.optim.Adam((network.parameters()), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True,
                                     betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

    elif args.dataset in ['drug']:
        criterion = nn.MSELoss(reduction=reduction).cuda()
        featurizer = DTI_Encoder()
        classifier = DTI_Classifier(featurizer.n_outputs, 1)
        network = nn.Sequential(featurizer, classifier).cuda()
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.dataset in ['precipitation']:
        criterion = nn.CrossEntropyLoss(reduction=reduction).cuda()
        network = PrecipitationNetwork(args, num_classes=9).cuda()
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    elif args.dataset in ['mimic']:
        network = Transformer(embedding_size=128, dropout=0.5, layers=2, heads=2).cuda()
        if args.prediction_type == 'readmission':
            class_weight = torch.FloatTensor(np.array([0.26, 0.74])).cuda()
        elif args.prediction_type == 'mortality':
            if args.lisa:
                class_weight = torch.FloatTensor(np.array([0.03, 0.97])).cuda()
            else:
                class_weight = torch.FloatTensor(np.array([0.05, 0.95])).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weight, reduction=reduction).cuda()
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)  # use lr = 5e-4

    elif args.dataset in ['arxiv', 'huffpost']:
        criterion = nn.CrossEntropyLoss(reduction=reduction).cuda()
        network = ArticleNetwork(num_classes=dataset.num_classes).cuda()
        optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.method == 'ft':
        trainer = FT(args, dataset, network, criterion, optimizer, scheduler)
    elif args.method == 'erm':
        trainer = ERM(args, dataset, network, criterion, optimizer, scheduler)
    elif args.method == 'groupdro':
        trainer = GroupDRO(args, dataset, network, criterion, optimizer, scheduler)
    elif args.method == 'ewc':
        trainer = EWC(args, dataset, network, criterion, optimizer, scheduler)
    elif args.method == 'agem':
        trainer = AGEM(args, dataset, network, criterion, optimizer, scheduler)
    elif args.method == 'si':
        trainer = SI(args, dataset, network, criterion, optimizer, scheduler)
    elif args.method == 'coral':
        trainer = DeepCORAL(args, dataset, network, criterion, optimizer, scheduler)
    elif args.method == 'irm':
        trainer = IRM(args, dataset, network, criterion, optimizer, scheduler)

    trainer.run()