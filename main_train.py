import os
import time
import yaml
import socket
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
            description='A PyTorch project for face recognition.')
    parser.add_argument('--config',
            help='train config file path')
    parser.add_argument('--proj_dir', default='project',
            help='the dir to save logs and models')
    parser.add_argument('--start_time', default='20220927_160047',
            help='time to start training')
    args = parser.parse_args()

    return args

def main_worker(rank, world_size, port, config):
    # init processes
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{port}',
        rank=rank, world_size=world_size,
    )

    # init runner and run
    trainer = Trainer(config)
    trainer.run()

    # clean up
    dist.destroy_process_group()

if __name__ == '__main__':
    # get arguments and config
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    proj_cfg = config['project']
    proj_cfg['proj_dir'] = args.proj_dir
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        device_ids = os.environ['CUDA_VISIBLE_DEVICES']
        world_size = len(device_ids.split(','))
        proj_cfg['device_ids'] = device_ids
    else:
        raise KeyError('GPU ID is missing.')

    start_time = time.strptime(args.start_time, '%Y%m%d_%H%M%S')
    while time.localtime() < start_time:
        print(args.start_time)
        time.sleep(600)

    # find an available port and start multiple processes
    with socket.socket() as sock:
        sock.bind(('', 0))
        port = sock.getsockname()[1]
    mp.spawn(
        main_worker, args=(world_size, port, config),
        nprocs=world_size, join=True,
    )