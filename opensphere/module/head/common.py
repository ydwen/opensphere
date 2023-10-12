import torch
import torch.nn.functional as F

# neck: feature interact network


def softplus_wrapping(raw_feats):
    '''
    Args:
        raw_feats: (B, L) or (B, C, L)
    Returns:
        out: (B, L) or (B, C, L)
    '''
    mags = F.softplus(raw_feats[..., :1], beta=1)
    feats = mags * F.normalize(raw_feats[..., 1:], dim=-1)
    return feats

def get_wrapping(mode):
    if mode == 'cartesian':
        f_wrapping = lambda raw_feats: raw_feats
    elif mode == 'polarlike':
        f_wrapping = softplus_wrapping
    else:
        raise ValueError('mode must be `cartesian` or `polarlike`')
    return f_wrapping


def weighted_average(feats, weights=None, pre_norm=False, post_norm=True):
    '''
    Args:
        feats: (B, C, L) or (C, L)
        B: batch size
        C: the number of observations from the same subject
        L: feature dimension
        weights: None or (C, ), weights for each observation
        pre_norm: normalize the features before fusion
        post_norm: normalize the fused feature
    Returns:
        fuse_feat: (B, L) or (L, )
    '''
    if weights is None:
        weights = torch.ones(feats.size(-2))
    weights = weights / weights.sum()
    weights = weights.to(feats.device)

    assert weights.ndim == 1
    if feats.ndim == 2:
        weights = weights.view(-1, 1)
    elif feats.ndim == 3:
        weights = weights.view(1, -1, 1)
    else:
        raise ValueError("Invalid shape for `feats`")

    if pre_norm:
        feats = F.normalize(feats, dim=-1)
    fuse_feat = torch.sum(weights * feats, dim=-2)
    if post_norm:
        fuse_feat = F.normalize(fuse_feat, dim=-1)

    return fuse_feat

def get_fusing(pre_norm, post_norm):
    f_fusing = lambda feats, weights=None: \
        weighted_average(feats, weights, pre_norm, post_norm)
    return f_fusing


def generalized_inner_product(feats_1, feats_2, b_theta=0., all_pairs=False):
    '''
    Args:
        feats_1: (B1, L)
        feats_2: (B2, L)
    Returns:
        scores: (B1, )   if all_pairs is False and B1==B2 
                (B1, B2) if all_pairs is True
    '''
    assert feats_1.ndim == feats_2.ndim == 2
    assert feats_1.size(1) == feats_2.size(1)

    mags_1 = torch.norm(feats_1, p=2, dim=1, keepdim=True)
    mags_2 = torch.norm(feats_2, p=2, dim=1, keepdim=True)
    if all_pairs:
        scores = feats_1.mm(feats_2.t()) - (b_theta * mags_1).mm(mags_2.t())
    else:
        scores = (feats_1 * feats_2).sum(dim=1) - (b_theta * mags_1 * mags_2).flatten()
    return scores

def get_scoring(b_theta):
    f_scoring = lambda feats_1, feats_2, all_pairs=False: \
        generalized_inner_product(feats_1, feats_2, b_theta, all_pairs)
    return f_scoring
