import os
import os.path as osp
import yaml
import time
import argparse

import torch
import torch.nn as nn

from tabulate import tabulate
from datetime import datetime

from utils import fill_config
from builder import build_dataloader, build_from_cfg


def parse_args():
    parser = argparse.ArgumentParser(
            description='A PyTorch project for face recognition.')
    parser.add_argument('--config', 
            help='config files for testing datasets')
    parser.add_argument('--proj_dirs', '--list', nargs='+',
            help='the project directories to be tested')
    parser.add_argument('--start_time', 
            help='time to start training')
    args = parser.parse_args()

    return args

@torch.no_grad()
def get_feats(net, data, flip=True):
    # extract features from the original 
    # and horizontally flipped data
    feats = net(data)
    if flip:
        data = torch.flip(data, [3])
        feats += net(data)

    return feats.data.cpu()

@torch.no_grad()
def test_run(net, checkpoints, dataloaders):
    tables = {}
    for n_ckpt, checkpoint in enumerate(checkpoints):
        # load model parameters
        net.load_state_dict(torch.load(checkpoint))
        for n_loader, dataloader in enumerate(dataloaders):
            # get feats from test_loader
            dataset_feats = []
            dataset_indices = []
            for n_batch, (data, indices) in enumerate(dataloader):
                # collect feature and indices
                data = data.cuda()
                indices = indices.tolist()
                feats = get_feats(net, data)
                dataset_feats.append(feats)
                dataset_indices.extend(indices)
                # progress
                print('feature extraction:',
                      'checkpoint: {}/{}'.format(n_ckpt+1, len(checkpoints)),
                      'dataset: {}/{}'.format(n_loader+1, len(dataloaders)),
                      'batch: {}/{}'.format(n_batch+1, len(dataloader)),
                      end='\r')
            print('')
            # eval
            dataset_feats = torch.cat(dataset_feats, dim=0)
            dataset_feats = dataset_feats[dataset_indices]
            results = dataloader.dataset.evaluate(dataset_feats)
            # save
            name = dataloader.dataset.name
            if name not in tables:
                tables[name] = []
            tables[name].append(results)

    return tables

def show_save_results(tables, table_paths, save_iters):
    # reorganize tables for showing and saving
    for (name, table), table_path in zip(tables.items(), table_paths):
        # insert #_ckpt
        assert len(table) == len(save_iters)
        for row, save_iter in zip(table, save_iters):
            row.insert(0, ('{}\n#ckpt'.format(name), str(save_iter)))
        # get all headers by iterating the table
        headers = []
        for row in table:
            for header, _ in row:
                if header in headers:
                    continue
                headers.append(header)
        # get content of each row based on the headers
        content = []
        for row in table:
            results = dict(row)
            content.append([results.get(header) for header in headers])
        # append avg results
        avg_row = ['avg',]
        for idx in range(1, len(headers)):
            cells = [row[idx] for row in content if row[idx] is not None]
            avg_row.append(sum(cells) / len(cells))
        content.append(avg_row)

        # print and save
        headers = [header.replace('=', '\n') for header in headers]
        table = tabulate(content, headers=headers,
            floatfmt='6.3f', stralign='center', numalign='center')
        print('\n', table)
        with open(table_path, 'w') as f:
            f.write(table)

def main_worker(config):
    # parallel setting
    device_ids = os.environ['CUDA_VISIBLE_DEVICES']
    device_ids = list(range(len(device_ids.split(','))))

    # build dataloader
    test_loaders = build_dataloader(config['data']['test'])

    # eval projects one by one
    for proj_dir in config['project']['proj_dirs']:
        print(proj_dir)
        # load config
        config_path = osp.join(proj_dir, 'config.yml')
        with open(config_path, 'r') as f:
            test_config = yaml.load(f, yaml.SafeLoader)
    
        # build model
        bkb_net = build_from_cfg(
            test_config['model']['backbone']['net'],
            'model.backbone',
        )
        bkb_net = nn.DataParallel(bkb_net, device_ids=device_ids)
        bkb_net = bkb_net.cuda()
        bkb_net.eval()

        # model paths and run test
        model_dir = test_config['project']['model_dir']
        save_iters = test_config['project']['save_iters']
        bkb_paths = [
            osp.join(model_dir, 'backbone_{}.pth'.format(save_iter))
            for save_iter in save_iters
        ]
        tables = test_run(bkb_net, bkb_paths, test_loaders)
        
        # paths to save tables
        table_paths = [
            osp.join(proj_dir, name + '.txt') for name in tables]
        show_save_results(tables, table_paths, save_iters)


if __name__ == '__main__':
    # get arguments and config
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    config['data'] = fill_config(config['data'])
 
    # override config
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise KeyError('Devices IDs have to be specified.'
                'CPU mode is not supported yet')

    if args.proj_dirs:
        config['project']['proj_dirs'] = args.proj_dirs

    if args.start_time:
        yy, mm, dd, h, m, s = args.start_time.split('-')
        yy, mm, dd = int(yy), int(mm), int(dd)
        h, m, s = int(h), int(m), int(s)
        start_time = datetime(yy, mm, dd, h, m, s)
        while datetime.now() < start_time:
            print(datetime.now())
            time.sleep(600)

    main_worker(config)
