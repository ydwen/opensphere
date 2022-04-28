import torch
import torch.nn as nn
import torch.nn.functional as F

import math


@torch.no_grad()
def get_d_theta(cos_theta, y, magn_type, m):
    # m * theta
    m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
    m_theta.scatter_(1, y.view(-1, 1), m, reduce='multiply')

    # delta
    if magn_type == 'v0':
        k = (m_theta / math.pi).floor()
        sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
        phi_theta = sign * torch.cos(m_theta) - 2. * k
        d_theta = phi_theta - cos_theta
    elif magn_type == 'v1':
        m_theta.clamp_(1e-5, 3.14159)
        phi_theta = torch.cos(m_theta)
        d_theta = phi_theta - cos_theta
    elif magn_type == 'v2':
        eta_theta = torch.cos(m_theta / m)
        d_theta = eta_theta - cos_theta
    else:
        raise NotImplementedError

    return d_theta

# Warning: Both SphereFaceR_N and SphereFaceR_S are not fully tested.
#          Please use SphereFaceR_H for now.

class SphereFaceR_N(nn.Module):
    """ reference: <SphereFace Revived: Unifying Hyperspherical Face Recognition>
    """
    def __init__(self, feat_dim, num_class, magn_type='v0',
            dm=0.1, steps=[], lw=1.):
        super(SphereFace_N, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.magn_type = magn_type
        self.dm = dm
        self.steps = steps
        self.lw = lw
        self.iter = 0

        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

    def forward(self, x, y):
        # adaptive margin for No FN
        self.iter += 1
        n_step = len([step for step in self.steps if step < self.iter])
        m = 1. + self.dm * n_step

        # weight normalization
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, 2, 0)

        # forward
        magnitude = torch.norm(x, p=2, dim=1, keepdim=True)
        cos_theta = x.mm(self.w) / magnitude
        d_theta = get_d_theta(cos_theta, y, self.magn_type, m)

        logits = magnitude * (cos_theta + d_theta)
        loss = self.lw * F.cross_entropy(logits, y)

        return loss


class SphereFaceR_H(nn.Module):
    """ reference: <SphereFace Revived: Unifying Hyperspherical Face Recognition>
    """
    def __init__(self, feat_dim, num_class, magn_type='v0',
            s=30., m=1.5, lw=50.):
        super(SphereFaceR_H, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.magn_type = magn_type
        self.s = s
        self.m = m
        self.lw = lw

        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

    def forward(self, x, y):
        # weight normalization
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, 2, 0)

        # forward
        magnitude = torch.norm(x, p=2, dim=1, keepdim=True)
        cos_theta = x.mm(self.w) / magnitude
        d_theta = get_d_theta(cos_theta, y, self.magn_type, self.m)

        logits = self.s * (cos_theta + d_theta)
        loss = self.lw * F.cross_entropy(logits, y) / self.s

        return loss


class SphereFaceR_S(nn.Module):
    """ reference: <SphereFace Revived: Unifying Hyperspherical Face Recognition>
    """
    def __init__(self, feat_dim, num_class, magn_type='v0',
            s=30., dm=1.5, steps=[], t=0.01, lw=50.):
        super(SphereFaceR_S, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.magn_type = magn_type
        self.s = s
        self.dm = dm
        self.steps = steps
        self.t = t
        self.lw = lw
        self.iter = 0

        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

    def forward(self, x, y):
        # adaptive margin for soft FN
        self.iter += 1
        n_step = len([step for step in self.steps if step < self.iter])
        m = 1. + self.dm * n_step

        # weight normalization
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, 2, 0)

        # forward
        magnitude = torch.norm(x, p=2, dim=1, keepdim=True)
        cos_theta = x.mm(self.w) / magnitude
        d_theta = get_d_theta(cos_theta, y, self.magn_type, m)

        logits = magnitude * (cos_theta + d_theta)
        loss = self.lw * F.cross_entropy(logits, y) / self.s
        loss += (self.t * torch.abs(magnitude - self.s)).mean()

        return loss
