import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from .common import get_wrapping, get_fusing, get_scoring

class SphereFace2(nn.Module):
    """ reference: <SphereFace2: Binary Classification is All You Need
                    for Deep Face Recognition> SphereFace2-C
    """
    def __init__(self, feat_dim, subj_num,
            alpha=0.7, r=40., m=0.4, t=3., lw=50.):
        super().__init__()
        #
        self.f_wrapping = get_wrapping(mode='cartesian')
        self.f_fusing = get_fusing(pre_norm=False, post_norm=True)
        self.f_scoring = get_scoring(b_theta=0.)

        self.feat_dim = feat_dim
        self.subj_num = subj_num
        # alpha is the lambda in paper Eqn. 5
        self.alpha = alpha
        self.r = r
        self.m = m
        self.t = t
        self.lw = lw

        # init weights
        self.w = nn.Parameter(torch.Tensor(subj_num, feat_dim))
        nn.init.xavier_normal_(self.w)

        # init bias
        z = alpha / ((1. - alpha) * (subj_num - 1.))
        ay = r * (2. * 0.5**t - 1. - m)
        ai = r * (2. * 0.5**t - 1. + m)

        temp = (1. - z)**2 + 4. * z * math.exp(ay - ai)
        b = math.log(2. * z) - ai - math.log(1. - z +  math.sqrt(temp))
        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.b, b)

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=-1)

        #delta theta with margin
        x = self.f_wrapping(x)
        x = self.f_fusing(x[:, None, :])
        cos_theta = self.f_scoring(x, self.w, all_pairs=True)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, y.view(-1, 1), 1.)
        with torch.no_grad():
            g_cos_theta = 2. * ((cos_theta + 1.) / 2.).pow(self.t) - 1.
            g_cos_theta = g_cos_theta - self.m * (2. * one_hot - 1.)
            d_theta = g_cos_theta - cos_theta

        logits = self.r * (cos_theta + d_theta) + self.b
        weight = self.alpha * one_hot + (1. - self.alpha) * (1. - one_hot)
        weight = self.lw * self.subj_num / self.r * weight
        loss = F.binary_cross_entropy_with_logits(logits, one_hot, weight=weight)

        return loss
