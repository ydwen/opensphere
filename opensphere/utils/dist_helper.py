import torch
from torch import distributed as dist


def is_dist() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

@torch.no_grad()
def gather_concat(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)

@torch.no_grad()
def gather_sum(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

@torch.no_grad()
def gather_max(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor
