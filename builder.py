import os
import copy
import warnings

import torch
import torch.nn as nn

from utils import is_dist, get_world_size, get_rank
from importlib import import_module


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
        raise KeyError(f'`cfg` must contain the key "type", but got {cfg}')

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
        the type of `cfg` could also be a dict for a dataloader,
        or a list or a tuple of dicts for multiple dataloaders.
    Returns:
        PyTorch dataloader(s)
    """
    if isinstance(cfg, (list, tuple)):
        return [build_dataloader(c) for c in cfg]
    else:
        if 'dataset' not in cfg:
            raise KeyError(f'`cfg` must contain the key "dataset", but got {cfg}')
        dataset = build_from_cfg(cfg['dataset'], 'dataset')
        world_size = get_world_size()
        if world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, shuffle=cfg['dataloader']['shuffle'])
        else:
            sampler = None
 
        if 'dataloader' not in cfg:
            raise KeyError(f'`cfg` must contain the key "dataloader", but got {cfg}')
        loader_cfg = copy.deepcopy(cfg['dataloader'])
        loader_cfg['dataset'] = dataset
        loader_cfg['sampler'] = sampler 
        loader_cfg['shuffle'] = (sampler is None) and loader_cfg['shuffle']
        # recompute the batch_size for each gpu
        sample_per_gpu = loader_cfg['batch_size'] // world_size
        if loader_cfg['batch_size'] % world_size != 0:
            warnings.warn('the batch size is changed '
            'to {}'.format(sample_per_gpu * world_size))
        loader_cfg['batch_size'] = sample_per_gpu

        worker_per_gpu = loader_cfg['num_workers']
        loader_cfg['num_workers'] = worker_per_gpu
 
        dataloader = build_from_cfg(loader_cfg, 'torch.utils.data')
    
        return dataloader


def build_module(cfg, module):
    if 'net' not in cfg:
        raise KeyError(f'`cfg` must contain the key "net", but got {cfg}')
    rank = get_rank()
    net = build_from_cfg(cfg['net'], module)
    net = net.to(rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    if 'pretrained' in cfg:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        net.load_state_dict(
                torch.load(cfg['pretrained'], map_location=map_location))

    if 'optimizer' not in cfg:
        raise KeyError(f'`cfg` must contain the key "optimizer", but got {cfg}')
    optim_cfg = copy.deepcopy(cfg['optimizer'])
    optim_cfg['params'] = net.parameters()
    optimizer = build_from_cfg(optim_cfg, 'torch.optim')

    if 'clip_grad_norm' not in cfg:
        cfg['clip_grad_norm'] = 1e5
        warnings.warn('`clip_grad_norm` is not set. The default is 1e5')
    clip_grad_norm = cfg['clip_grad_norm']

    if 'scheduler' not in cfg:
        raise KeyError(f'`cfg` must contain the key "scheduler", but got {cfg}')
    sched_cfg = copy.deepcopy(cfg['scheduler'])
    sched_cfg['optimizer'] = optimizer
    scheduler = build_from_cfg(sched_cfg, 'torch.optim.lr_scheduler')
    
    return {'net': net, 'clip_grad_norm': clip_grad_norm,
            'optimizer': optimizer, 'scheduler': scheduler}


def build_model(cfg):
    if 'backbone' not in cfg:
        raise KeyError(f'`cfg` must contain the key "backbone", but got {cfg}')
    if 'head' not in cfg:
        raise KeyError(f'`cfg` must contain the key "head", but got {cfg}')

    model = {}
    for module in cfg:
        model[module] = build_module(cfg[module], f'model.{module}')
    return model

