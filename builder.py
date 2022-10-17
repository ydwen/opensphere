import warnings

import torch
import torch.nn as nn

from copy import deepcopy
from importlib import import_module
from utils import get_world_size, get_rank

def build_from_cfg(cfg, module):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError('Missing key "type" in `cfg`', cfg)

    args = cfg.copy()
    obj_type = args.pop('type')
    if not isinstance(obj_type, str):
        raise TypeError(f'type must be a str, but got {type(obj_type)}')
    else:
        obj_cls = getattr(import_module(module), obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {module} module')

    return obj_cls(**args)

def build_dataloader(cfg):
    """
    Args:
        the `cfg` could be a dict for a dataloader,
        or a list or a tuple of dicts for multiple dataloaders.
    Returns:
        PyTorch dataloader(s)
    """
    if isinstance(cfg, (list, tuple)):
        return [build_dataloader(c) for c in cfg]

    if 'dataset' not in cfg:
        raise KeyError('Missing key "dataset" in `cfg`', cfg)
    if 'dataloader' not in cfg:
        raise KeyError('Missing key "dataloader" in `cfg`', cfg)

    args = deepcopy(cfg)
    dataset = build_from_cfg(args['dataset'], 'dataset')

    # sampler
    sampler = None
    world_size = get_world_size()
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=args['dataloader']['shuffle'])
        args['dataloader']['shuffle'] = False

    # recompute the batch_size for each gpu
    batch_size = args['dataloader']['batch_size']
    sample_per_gpu = batch_size // world_size
    if batch_size % world_size != 0:
        warnings.warn(f'change batch_size to {sample_per_gpu * world_size}')

    args['dataloader']['dataset'] = dataset
    args['dataloader']['sampler'] = sampler
    args['dataloader']['batch_size'] = sample_per_gpu
    dataloader = build_from_cfg(args['dataloader'], 'torch.utils.data')

    return dataloader


def build_module(cfg):
    rank = get_rank()
    net = build_from_cfg(cfg, 'model').to(rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[rank])

    return net

def build_model(cfg):
    # check
    if 'backbone' not in cfg:
        raise KeyError('Missing key "backbone" in `cfg`', cfg)
    if 'head' not in cfg:
        raise KeyError('Missing key "head" in `cfg`', cfg)
    if 'optimizer' not in cfg:
        raise KeyError('Missing key "optimizer" in `cfg`', cfg)
    if 'scheduler' not in cfg:
        raise KeyError('Missing key "scheduler" in `cfg`', cfg)

    args = deepcopy(cfg)
    backbone = build_module(args['backbone'])
    head = build_module(args['head'])
    feat_dim = args['head']['feat_dim']

    args['optimizer']['params'] = [
        {'params': backbone.parameters()},
        {'params': head.parameters()},
    ]
    optimizer = build_from_cfg(args['optimizer'], 'torch.optim')

    args['scheduler']['optimizer'] = optimizer
    scheduler = build_from_cfg(
        args['scheduler'], 'torch.optim.lr_scheduler')

    if 'max_grad_norm' not in args:
        max_grad_norm = 1e5
        warnings.warn('Set `max_grad_norm` to 1e5')
    else:
        max_grad_norm = args['max_grad_norm']

    return {'backbone': backbone,
            'head': head,
            'feat_dim': feat_dim,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'max_grad_norm': max_grad_norm}
