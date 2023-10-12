import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from .common import get_wrapping, get_fusing, get_scoring

class SphereFace(nn.Module):
    """ reference: <SphereFace: Deep Hypersphere Embedding for Face Recognition>"
        It also used characteristic gradient detachment tricks proposed in
        <SphereFace Revived: Unifying Hyperspherical Face Recognition>.
    """
    def __init__(self, feat_dim, subj_num, s=30., m=1.7):
        super().__init__()
        #
        self.f_wrapping = get_wrapping(mode='cartesian')
        self.f_fusing = get_fusing(pre_norm=False, post_norm=True)
        self.f_scoring = get_scoring(b_theta=0.)

        self.feat_dim = feat_dim
        self.subj_num = subj_num
        self.s = s
        self.m = m

        self.w = nn.Parameter(torch.Tensor(subj_num, feat_dim))
        nn.init.xavier_normal_(self.w)

    def forward(self, x, y):
        # weight normalization
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=-1)

        # cos_theta and d_theta
        x = self.f_wrapping(x)
        x = self.f_fusing(x[:, None, :])
        cos_theta = self.f_scoring(x, self.w, all_pairs=True)
        with torch.no_grad():
            m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
            m_theta.scatter_(1, y.view(-1, 1), self.m, reduce='multiply')
            k = (m_theta / math.pi).floor()
            sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
            phi_theta = sign * torch.cos(m_theta) - 2. * k
            d_theta = phi_theta - cos_theta

        logits = self.s * (cos_theta + d_theta)
        loss = F.cross_entropy(logits, y)

        return loss
