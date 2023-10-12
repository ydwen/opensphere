import torch

from tqdm import tqdm
from copy import deepcopy
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils.dist_helper import is_dist, get_rank, get_world_size
from .utils.build_helper import build_from_args


class Model():
    def __init__(self, cfg):
        # get dist info
        self.rank = get_rank()
        self.world_size = get_world_size()

        # check
        if 'backbone' not in cfg:
            raise KeyError('Missing key "backbone" in `cfg`', cfg)
        if 'head' not in cfg:
            raise KeyError('Missing key "head" in `cfg`', cfg)
        # build modules
        args = deepcopy(cfg)
        self.backbone = self.build_module(args['backbone']).to(self.rank)
        self.head = self.build_module(args['head']).to(self.rank)
        if is_dist():
            self.backbone = DDP(self.backbone, device_ids=[self.rank])
            self.head = DDP(self.head, device_ids=[self.rank])
        self.embed_dim = args['backbone']['embed_dim']

    def build_module(self, args):
        return build_from_args(args, 'opensphere.module')

    def train_mode(self):
        self.backbone.train()
        self.head.train()

    def eval_mode(self):
        self.backbone.eval()
        self.head.eval()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def state_dict(self):
        if is_dist():
            backbone = self.backbone.module
            head = self.head.module
        else:
            backbone = self.backbone
            head = self.head
        return {
            'backbone': backbone.state_dict(),
            'head': head.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if is_dist():
            self.backbone.module.load_state_dict(state_dict['backbone'])
            self.head.module.load_state_dict(state_dict['head'])
        else:
            self.backbone.load_state_dict(state_dict['backbone'])
            self.head.load_state_dict(state_dict['head'])


    @torch.no_grad()
    def get_feature(self, data):
        '''
        Args:
            data: images with the size of (B, 3, H, W)
        Returns:
            feats: (B, 2, embed_dim), 2 for the original and flipped images
        '''
        feats = self.backbone(data)
        data_flip = torch.flip(data, [3])
        feats_flip = self.backbone(data_flip)
        return torch.stack([feats, feats_flip], dim=1)

    @torch.no_grad()
    def get_feature_dataset(self, dataloader):
        '''
        Args:
            dataloader: a dataloader for a dataset
        Returns:
            feats: (N, 2, embed_dim)
        '''
        dataset_feats = torch.zeros(
            len(dataloader.dataset), 2, self.embed_dim,
            dtype=torch.float32, device=self.rank,
        )
        desc = f'feature extraction for dataset `{dataloader.dataset.name}`'
        disable = (self.rank != 0)
        for data, indices in tqdm(dataloader, desc=desc, disable=disable):
            data = data.to(self.rank)
            indices = indices.tolist()
            feats = self.get_feature(data)
            dataset_feats[indices] = feats
        dist.all_reduce(dataset_feats, op=dist.ReduceOp.SUM)

        return dataset_feats

    def evaluate_dataset(self, dataset, feats):
        '''
        Args:
            feats: (N, 2, embed_dim)
        Returns:
            results: a dict of evaluation results,
                e.g. {'ACC': 96.1%, 'EER': 4.3%, 'TPR@FPR=0.0001': 58.2%, ...}
        '''
        f_wrapping = self.head.module.f_wrapping
        f_fusing = self.head.module.f_fusing
        f_scoring = self.head.module.f_scoring

        return dataset.evaluate(feats, f_wrapping, f_fusing, f_scoring)
