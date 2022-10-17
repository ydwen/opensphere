import os
import os.path as osp
import math
import time
import yaml
import torch

from opensphere import OpenSphere
from utils import fill_config, get_rank
from utils import IterLoader, MeterLogger
from builder import build_dataloader, build_model

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


class Trainer():
    def __init__(self, config):
        # build train and test dataloaders
        data_cfg = config['data']
        data_cfg = fill_config(data_cfg)
        train_loader = build_dataloader(data_cfg['train'])
        self.val_loaders = build_dataloader(data_cfg['val'])

        # build OpenSphere model
        model_cfg = config['model']
        feat_dim = model_cfg['backbone']['out_channel']
        model_cfg['head']['feat_dim'] = feat_dim
        num_class = len(set(train_loader.dataset.classes))
        model_cfg['head']['num_class'] = num_class
        self.opensphere = OpenSphere(**build_model(model_cfg))

        # has to be after the model initalization and IDK why
        self.train_loader = IterLoader(train_loader)

        # project info of a new runner
        proj_cfg = config['project']
        self.iter = 0
        self.val_intvl = proj_cfg['val_intvl']
        self.save_iters = proj_cfg['save_iters']
        self.max_iter = max(proj_cfg['save_iters'])

        self.rank = get_rank()
        if self.rank != 0:
            return

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        proj_dir = osp.join(proj_cfg['proj_dir'], timestamp)
        self.ckpt_dir = osp.join(proj_dir, 'checkpoint')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # loggers
        train_log_path = osp.join(proj_dir, 'train.log')
        val_log_path = osp.join(proj_dir, 'val.log')
        self.train_logger = MeterLogger(
            name='train', path=train_log_path, **proj_cfg['train_log'])
        self.val_logger = MeterLogger(
            name='val', path=val_log_path, **proj_cfg['val_log'])
        # save config file
        cfg_path = osp.join(proj_dir, 'config.yml')
        with open(cfg_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=None)        


    def train(self):
        # load data
        data, labels = next(self.train_loader)
        data, labels = data.to(self.rank), labels.to(self.rank)
        # fit
        self.opensphere.set_mode('train')
        msg = self.opensphere.fit(data, labels)

        if self.rank == 0:
            # logging and update meters
            msg['Iter'] = self.iter
            self.train_logger.update(msg)

    @torch.no_grad()
    def val(self):
        # switch to test mode
        self.opensphere.set_mode('eval')
        msg = {'Iter': self.iter}
        for val_loader in self.val_loaders:
            dataset = val_loader.dataset
            # print(f'validating {dataset.name:^10s},',
            #       f'checkpoint: {self.iter:5d}', end='\r')
            feats = self.opensphere.get_feature_dataset(val_loader)
            result = dataset.evaluate(
                self.opensphere.scoring, feats.cpu())
            result = dict(result)
            msg[dataset.name] = result[dataset.metrics[0]]

        if self.rank == 0:
            self.val_logger.update(msg)

    def run(self):
        while self.iter <= self.max_iter:
            # train step
            if self.iter % self.val_intvl == 0 and self.iter > 0:
                self.val()

            if self.iter in self.save_iters:
                if self.rank == 0:
                    ckpt_path = osp.join(
                        self.ckpt_dir, f'checkpoint_{self.iter}.pth')
                    self.opensphere.save_checkpoint(ckpt_path)

            self.train()
            self.iter += 1
