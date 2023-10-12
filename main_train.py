import os
import os.path as osp
import time
import yaml
import torch

import socket
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from glob import glob
from opensphere.utils.build_helper import build_from_args

#import random
#import numpy
#random.seed(0)
#numpy.random.seed(0)
#torch.manual_seed(0)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def get_config_from_args():
    # get arguments
    parser = argparse.ArgumentParser(
            description='A PyTorch framework for deep metric learning.')
    parser.add_argument('--cfg_path', default='',
            help='path of training config')
    parser.add_argument('--proj_dir', default='./project',
            help='the dir to save logs and models')
    parser.add_argument('--start_time', default='20220927_160047',
            help='time to start training')
    args = parser.parse_args()

    # check if resume from a project
    ckpt_paths = glob(osp.join(args.proj_dir, 'checkpoint/ckpt_*.pth'))
    if len(ckpt_paths) > 0:
        # if resume, project directory should be provided
        cfg_path = osp.join(args.proj_dir, 'config.yml')
    else:
        # if not, path to config file should be provided
        cfg_path = args.cfg_path

    # get config
    with open(cfg_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    # update config with args
    trainer_cfg = config['trainer']
    trainer_cfg['proj_dir'] = args.proj_dir

    # check gpu ids
    trainer_cfg['device_ids'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    if trainer_cfg['device_ids'] is None:
        raise KeyError('Please specify GPU IDs.')

    # time to start
    start_time = time.strptime(args.start_time, '%Y%m%d_%H%M%S')
    while time.localtime() < start_time:
        print(args.start_time)
        time.sleep(666)
    print('start...')

    return config

def main_worker(rank, world_size, port, config):
    # init processes
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{port}',
        rank=rank, world_size=world_size,
    )

    # init trainer and run
    trainer_args = {'type': config['trainer']['type'], 'config': config}
    trainer = build_from_args(trainer_args, 'opensphere')
    trainer.run()
    
    # clean up
    dist.destroy_process_group()

if __name__ == '__main__':
    # get arguments and config
    config = get_config_from_args()

    # find an available port 
    with socket.socket() as sock:
        sock.bind(('', 0))
        port = sock.getsockname()[1]

    # start multiple processes
    world_size = len(config['trainer']['device_ids'].split(','))
    mp.spawn(
        main_worker, args=(world_size, port, config),
        nprocs=world_size, join=True,
    )
