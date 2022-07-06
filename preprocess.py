'''
Script to preprocess all datasets used in Temporal Robustness and save to data_dir.
'''

import argparse
from archived.mnist.rotated_mnist import preprocess_rotated_mnist
from archived.mnist import preprocess_rainbow_mnist, preprocess_temporal_rainbow_mnist
from archived.mnist import preprocess_temporal_rainbow_mnist
from archived.waterbirds.preprocess import preprocess_waterbirds
from data.yearbook.preprocess import preprocess_yearbook
from archived.data_config import get_config

parser = argparse.ArgumentParser(description='data_preprocess')
parser.add_argument('--data_dir', default='/iris/u/cchoi1/Data', type=str, help='directory for datasets.')
parser.add_argument('--dataset', default='yearbook', type=str, help='datasets to preprocess')
parser.add_argument('--land_domains_sizes', nargs="+", type=int, help='list of land domain sizes, i.e. 1 1 1 1 1')
parser.add_argument('--water_domains_sizes', nargs="+", type=int, help='list of water domain sizes, i.e. 1 1 1 1')

args = parser.parse_args()
print(args)

if __name__ == '__main__':
    data_config = get_config(args.dataset, args.data_dir)
    if args.dataset == 'rainbow_mnist':
        preprocess_temporal_rainbow_mnist(data_config)
    elif args.dataset == 'cub':
        preprocess_waterbirds(data_config, list(args.land_domains_sizes), list(args.water_domains_sizes))
    elif args.dataset == 'yearbook':
        preprocess_yearbook(data_config)
