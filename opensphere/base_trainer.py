import os
import os.path as osp
import re
import time
import yaml
import json
import torch

from glob import glob
from copy import deepcopy
from .model import Model
from .dataloader import build_dataloader, IterLoader
from .utils.build_helper import build_from_args
from .utils.dist_helper import get_rank, get_world_size
from .utils.logger import PythonLogger

class BaseTrainer():
    def __init__(self, config):
        # get config
        self.config = config
        # get dist info
        self.rank = get_rank()
        self.world_size = get_world_size()
        # init
        self.init_settings()
        self.init_dataloader()
        self.init_model()
        self.init_optimizer()
        self.init_scheduler()

        if self.rank != 0:
            return

        self.init_workplace()
        self.init_loggers()

    def init_settings(self):
        args = self.config['trainer']
        self.step = 0
        self.amp = args['amp']
        self.val_intvl = args['val_intvl']
        self.ckpt_steps = args['ckpt_steps']
        self.save_steps = args['save_steps']
        self.max_step = max(args['scheduler']['milestones'])
        self.max_grad_norm = args.get('max_grad_norm', 1e5)
        # check if resume
        ckpt_paths = glob(osp.join(args['proj_dir'], 'checkpoint/ckpt_*.pth'))
        if len(ckpt_paths) > 0:
            # resume
            pattern = r'ckpt_(\d+).pth'
            ckpt_paths.sort(key=lambda x: int(re.search(pattern, x).group(1)))
            self.proj_dir = args['proj_dir']
            self.ckpt_dir = osp.dirname(ckpt_paths[-1])
            self.ckpt_path = ckpt_paths[-1]
        else:
            # from scratch
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            self.proj_dir = osp.join(args['proj_dir'], timestamp)
            self.ckpt_path = None
            self.ckpt_dir = osp.join(self.proj_dir, 'checkpoint')

    def init_dataloader(self):
        # build train and test dataloaders
        args = self.config['data']
        train_loader = build_dataloader(args['train'])
        self.train_loader = IterLoader(train_loader)
        self.val_loaders = build_dataloader(args['val'])

    def init_model(self):
        model_cfg = self.config['model']
        self.model = Model(model_cfg)

    def init_optimizer(self):
        args = deepcopy(self.config['trainer']['optimizer'])
        # lrs and params
        base_lr = args['lr']
        base_wd = args['weight_decay']
        bkb_params = {
            'lr': args.pop('bkb_lr', base_lr),
            'weight_decay': args.pop('bkb_wd', base_wd),
            'params': self.model.backbone.parameters(),
        }
        head_params = {
            'lr': args.pop('head_lr', base_lr),
            'weight_decay': args.pop('head_wd', base_wd),
            'params': self.model.head.parameters(),
        }
        args['params'] = [bkb_params, head_params]
        self.optimizer = build_from_args(args, 'torch.optim')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def init_scheduler(self):
        args = deepcopy(self.config['trainer']['scheduler'])
        args['optimizer'] = self.optimizer
        self.scheduler = build_from_args(args, 'torch.optim.lr_scheduler')

    def init_workplace(self):
        # make ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        print(f'saving to {self.ckpt_dir}...')
        # save config file
        cfg_path = osp.join(self.proj_dir, 'config.yml')
        with open(cfg_path, 'w') as f:
            yaml.dump(self.config, f, sort_keys=False, default_flow_style=None)

    def init_loggers(self):
        self.train_logger = PythonLogger(
            path=osp.join(self.proj_dir, 'train.log'),
            **self.config['trainer']['train_log'],
        )
        self.val_logger = PythonLogger(
            path=osp.join(self.proj_dir, 'val.log'),
            **self.config['trainer']['val_log'],
        )
 
    def state_dict(self):
        return {
            'step': self.step,
            'epoch': self.epoch + 1, # to avoid repeat?
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict['step']
        self.epoch = state_dict['epoch']
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.scaler.load_state_dict(state_dict['scaler'])

    def load_checkpoint(self):
        # load checkpoint
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        ckpt_dict = torch.load(self.ckpt_path, map_location=map_location)
        self.load_state_dict(ckpt_dict)
        # load dataloader
        self.train_loader.set_epoch(self.epoch)

        if self.rank != 0:
            return
        print(f'=> load a checkpoint from {self.ckpt_path}')

    def train_step(self):
        raise NotImplementedError

    @torch.no_grad()
    def val_step(self):
        # switch to eval_mode
        self.model.eval_mode()
        for val_loader in self.val_loaders:
            name = val_loader.dataset.name
            feats = self.model.get_feature_dataset(val_loader)
            results, scores, labels = \
                self.model.evaluate_dataset(val_loader.dataset, feats.cpu())

            if self.rank != 0:
                continue

            screen_msgs = {f'{name}_{k}': v for (k, v) in results}
            self.val_logger.add(screen_msgs, step=self.step)

    def run(self):
        # resume if needed
        if self.ckpt_path is not None:
            self.load_checkpoint()
        # main loop
        while self.step <= self.max_step:
            # update step
            self.step += 1
            self.epoch = self.train_loader.epoch
            # train & val
            self.train_step()
            if self.step % self.val_intvl == 0 or self.step == 666:
                self.val_step()

            if self.rank != 0:
                continue

            # log
            # save checkpint
            if self.step in self.ckpt_steps:
                ckpt_path = osp.join(self.ckpt_dir, f'ckpt_{self.step}.pth')
                torch.save(self.state_dict(), ckpt_path)
                print(f'save checkpoint to {ckpt_path}')
            # save model
            if self.step in self.save_steps:
                model_path = osp.join(self.ckpt_dir, f'model_{self.step}.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f'save model to {model_path}')
