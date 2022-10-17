import os
import os.path as osp
import yaml
import torch

from opensphere import OpenSphere
from utils import fill_config, get_rank, Meter
from builder import build_dataloader, build_model
from tabulate import tabulate

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


class Tester():
    def __init__(self, config):
        # build train and test dataloaders
        data_cfg = config['data']
        data_cfg = fill_config(data_cfg)
        self.test_loaders = build_dataloader(data_cfg['test'])

        # get config of the testing project
        self.proj_dir = config['project']['proj_dir']
        cfg_path = osp.join(self.proj_dir, 'config.yml')
        with open(cfg_path, 'r') as f:
            test_cfg = yaml.load(f, yaml.SafeLoader)

        # build OpenSphere model
        model_cfg = test_cfg['model']
        self.opensphere = OpenSphere(**build_model(model_cfg))

        # project info of a new runner
        self.save_iters = test_cfg['project']['save_iters']
        self.ckpt_dir = osp.join(self.proj_dir, 'checkpoint')
        self.rank = get_rank()


    @torch.no_grad()
    def test(self, test_loader):
        meter = Meter()
        for save_iter in self.save_iters:
            # load checkpoint and switch to test mode
            ckpt_path = osp.join(
                self.ckpt_dir, f'checkpoint_{save_iter}.pth')
            self.opensphere.load_checkpoint(ckpt_path)
            self.opensphere.set_mode('eval')

            dataset = test_loader.dataset
            if self.rank == 0:
                print(f'evaluating dataset {dataset.name},',
                      f'checkpoint: {save_iter:5d}')
            feats = self.opensphere.get_feature_dataset(test_loader)
            record = dataset.evaluate(
                self.opensphere.scoring, feats.cpu())
            record.insert(0, ('Iter', int(save_iter)))
            meter.add(dict(record))
        
        return dataset.name, meter

    def parse(self, meter):
        content = [[rcd[h] for h in meter.headers]
                   for rcd in meter.summary()]
        headers = [h.replace('=', '\n') for h in meter.headers]
        table = tabulate(content, headers=headers,
            floatfmt='6.3f', stralign='center', numalign='center')

        return table

    def run(self):
        for test_loader in self.test_loaders:
            name, meter = self.test(test_loader)
            table = self.parse(meter)

            if self.rank != 0:
                continue
            print(self.proj_dir, name, '\n', table)
            table_path = osp.join(self.proj_dir, f'{name}.txt')
            with open(table_path, 'w') as f:
                f.write(table)
