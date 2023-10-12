import os
import os.path as osp
import time
import yaml
import socket
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from opensphere.tester import Tester

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def get_config_from_args():
    parser = argparse.ArgumentParser(
            description='A PyTorch project for face recognition.')
    parser.add_argument('--cfg_path', 
            help='path of testing config')
    parser.add_argument('--proj_dir',
            help='the directory to load model')
    parser.add_argument('--start_time', default='20220927_160047',
            help='time to start testing')
    args = parser.parse_args()

    # config of test data
    with open(args.cfg_path, 'r') as f:
        test_cfg = yaml.load(f, yaml.SafeLoader)

    # config of trained model
    cfg_path = osp.join(args.proj_dir, 'config.yml')
    with open(cfg_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    # config for main_test
    config['data'] = test_cfg['data']
    config['trainer']['proj_dir'] = args.proj_dir

        # check gpu ids
    config['trainer']['device_ids'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    if config['trainer']['device_ids'] is None:
        raise KeyError('Please specify GPU IDs.')

    start_time = time.strptime(args.start_time, '%Y%m%d_%H%M%S')
    while time.localtime() < start_time:
        print(args.start_time)
        time.sleep(666)

    return config

def main_worker(rank, world_size, port, config):
    # init processes
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{port}',
        rank=rank, world_size=world_size,
    )

    tester = Tester(config)
    tester.run()

    # clean up
    dist.destroy_process_group()


if __name__ == '__main__':
    # get arguments
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
