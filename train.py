import os
import yaml
import time
import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from datetime import datetime
from runner import IterRunner
from utils import fill_config
from builder import build_dataloader, build_model


def parse_args():
    parser = argparse.ArgumentParser(
            description='A PyTorch project for face recognition.')
    parser.add_argument('--config', 
            help='train config file path')
    parser.add_argument('--proj_dir', 
            help='the dir to save logs and models')
    parser.add_argument('--start_time', 
            help='time to start training')
    args = parser.parse_args()

    return args

def main_worker(rank, world_size, config):
    # init processes
    backend = config['parallel']['backend']
    dist_url = config['parallel']['dist_url']
    dist.init_process_group(
            backend=backend, init_method=dist_url,
            world_size=world_size, rank=rank)

    # init dataloader
    train_loader = build_dataloader(config['data']['train'])
    val_loaders = build_dataloader(config['data']['val'])

    # init model
    torch.cuda.set_device(rank)
    feat_dim = config['model']['backbone']['net']['out_channel']
    config['model']['head']['net']['feat_dim'] = feat_dim
    num_class = len(train_loader.dataset.classes)
    config['model']['head']['net']['num_class'] = num_class
    model = build_model(config['model'])
    if rank==0:
        print(model)

    # init runner and run
    runner = IterRunner(config, train_loader, val_loaders, model)
    runner.run()

    # clean up
    dist.destroy_process_group()


if __name__ == '__main__':
    # get arguments and config
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    config['data'] = fill_config(config['data'])
    config['model'] = fill_config(config['model'])

    # override config
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise KeyError('Devices IDs have to be specified.'
                'CPU mode is not supported yet')
    else:
        device_ids = os.environ['CUDA_VISIBLE_DEVICES']
        world_size = len(device_ids.split(','))
        config['parallel']['device_ids'] = device_ids
        config['parallel']['world_size'] = world_size

    if args.proj_dir:
        config['project']['proj_dir'] = arg.proj_dir

    if args.start_time:
        yy, mm, dd, h, m, s = args.start_time.split('-')
        yy, mm, dd = int(yy), int(mm), int(dd)
        h, m, s = int(h), int(m), int(s)
        start_time = datetime(yy, mm, dd, h, m, s)
        while datetime.now() < start_time:
            print(datetime.now())
            time.sleep(600)

    # start multiple processes
    mp.spawn(
        main_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True,
    )
