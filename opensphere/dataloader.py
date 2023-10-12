import warnings
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

from .utils.dist_helper import get_rank, get_world_size
from .utils.build_helper import build_from_args


class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, batch_size, total_size, seed=0,
                 weights=None, shuffle=True, drop_last=False):
        # Weighted sampler for :class:`torch.nn.parallel.DistributedDataParallel`.
        assert shuffle is True
        assert drop_last is True

        self.dataset = dataset
        self.batch_size = batch_size
        self.total_size = total_size
        self.seed = seed
        if weights is None:
            self.weights = np.ones(len(dataset))
        else:
            self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()
        assert len(self.weights) == len(dataset)

        self.epoch = 0
        self.rank = get_rank()
        self.num_replicas = get_world_size()
        assert self.batch_size % self.num_replicas == 0

        self.num_batch = self.total_size // self.batch_size
        self.total_size = self.num_batch * self.batch_size
        self.num_samples = self.total_size // self.num_replicas


    def __iter__(self):
        # deterministically shuffle based on epoch and seed
        indices = []
        desc = f'sampling the following {self.num_batch} mini-batches'
        disable = (self.rank != 0)
        np.random.seed(self.seed + self.epoch)
        for idx in tqdm(range(self.num_batch), desc=desc, disable=disable):
            # It's slow to use torch.multinomial and IDKW
            sub_indices = np.random.choice(
                len(self.weights), size=self.batch_size,
                replace=False, p=self.weights,
            )
            indices.extend(sub_indices.tolist())
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def build_dataloader(cfg):
    """ build a dataloader or a list of dataloaders
    Args:
        cfg: a dict for a dataloader, 
             or a list or a tuple of dicts for multiple dataloaders.
    Returns:
        dataloader(s): PyTorch dataloader(s)
    """
    if isinstance(cfg, (list, tuple)):
        return [build_dataloader(c) for c in cfg]

    if 'dataset' not in cfg:
        raise KeyError('Missing key "dataset" in `cfg`', cfg)
    if 'dataloader' not in cfg:
        raise KeyError('Missing key "dataloader" in `cfg`', cfg)
    args = deepcopy(cfg)

    # dataset
    dataset = build_from_args(args['dataset'], 'opensphere.dataset')
    
    # recompute the batch_size for each gpu
    batch_size = args['dataloader']['batch_size']
    world_size = get_world_size()
    sample_per_gpu = batch_size // world_size
    if batch_size % world_size != 0:
        warnings.warn(f'change batch_size to {sample_per_gpu*world_size}')

    args['dataloader']['dataset'] = dataset
    args['dataloader']['batch_size'] = sample_per_gpu
    # sampler for ddp
    sampler_args = args['dataloader'].pop('sampler', {})
    if sampler_args.get('type') == 'DistributedWeightedSampler':
        sampler = DistributedWeightedSampler(
            dataset, batch_size=sample_per_gpu*world_size,
            total_size=sampler_args['total_size'],
            weights=getattr(dataset, 'weights', None),
            shuffle=True, drop_last=True,
        )
    else:
        sampler = DistributedSampler(
            dataset, shuffle=args['dataloader']['shuffle'],
        )
    args['dataloader']['sampler'] = sampler
    args['dataloader']['shuffle'] = False # shuffle is now done by sampler

    dataloader = build_from_args(args['dataloader'], 'torch.utils.data')

    return dataloader


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._epoch = 0
        self.shuffle()

    @property
    def epoch(self):
        return self._epoch

    def set_epoch(self, epoch):
        self._epoch = epoch

    def shuffle(self):
        if hasattr(self._dataloader.sampler, 'set_epoch'):
            self._dataloader.sampler.set_epoch(self._epoch)
        self.iter_loader = iter(self._dataloader)

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            self.shuffle()
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)
