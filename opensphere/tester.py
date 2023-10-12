import os.path as osp
import torch

from .dataloader import build_dataloader
from .model import Model
from .utils.dist_helper import get_rank
from .utils.logger import Meter
from tabulate import tabulate

class Tester():
    def __init__(self, config):
        # 
        self.config = config
        # init dataloader
        self.test_loaders = build_dataloader(config['data']['test'])
        # init model
        self.model = Model(config['model'])

        # models to load
        self.save_steps = config['trainer']['save_steps'][-1:]
        self.proj_dir = config['trainer']['proj_dir']
        self.ckpt_dir = osp.join(self.proj_dir, 'checkpoint')

        self.rank = get_rank()

    @torch.no_grad()
    def test_step(self, test_loader):
        meter = Meter()
        for save_step in self.save_steps:
            # load checkpoint and switch to test mode
            model_path = osp.join(self.ckpt_dir, f'model_{save_step}.pth')
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            model_dict = torch.load(model_path, map_location=map_location)
            self.model.load_state_dict(model_dict)
            self.model.eval_mode()

            feats = self.model.get_feature_dataset(test_loader)
            results, _, _ = self.model.evaluate_dataset(test_loader.dataset, feats.cpu())
            meter.add(dict(results), step=int(save_step))
        return meter

    def parse(self, meter):
        content, headers = meter.summary()
        headers = [h.replace('=', '\n') for h in headers]
        table = tabulate(content, headers=headers, stralign='center')
        return table

    def run(self):
        for test_loader in self.test_loaders:
            name = test_loader.dataset.name
            meter = self.test_step(test_loader)
            table = self.parse(meter)

            if self.rank != 0:
                continue
            print(self.proj_dir, name, '\n', table)
            table_path = osp.join(self.proj_dir, f'{name}.txt')
            with open(table_path, 'w') as f:
                f.write(table)
