import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class SphereFace(nn.Module):
    """ reference: <SphereFace: Deep Hypersphere Embedding for Face Recognition>"
        It also used characteristic gradient detachment tricks proposed in
        <SphereFace Revived: Unifying Hyperspherical Face Recognition>.
    """
    def __init__(self, feat_dim, num_class, s=30., m=1.5):
        super(SphereFace, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

    def forward(self, x, y):
        # weight normalization
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        # cos_theta and d_theta
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        with torch.no_grad():
            m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
            m_theta.scatter_(
                1, y.view(-1, 1), self.m, reduce='multiply',
            )
            k = (m_theta / math.pi).floor()
            sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
            phi_theta = sign * torch.cos(m_theta) - 2. * k
            d_theta = phi_theta - cos_theta

        logits = self.s * (cos_theta + d_theta)
        loss = F.cross_entropy(logits, y)

        return loss

    def scoring(self, x0, x1, n2m=False):
        x0 = F.normalize(x0, dim=1)
        x1 = F.normalize(x1, dim=1)
        if n2m:
            return x0.mm(x1.t())
        return torch.sum(x0 * x1, dim=1)
