import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class GAsoftmax(nn.Module):
    """ reference: <Deep Hyperspherical Learning> in NIPS 2017"
        Section 3 in the paper: https://arxiv.org/abs/1711.03189
    """
    def __init__(self, feat_dim, num_class, s=30., m=1.5):
        super(GAsoftmax, self).__init__()
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
        #with torch.no_grad():
        m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
        m_theta_ori = m_theta
        with torch.no_grad():
            m_theta.scatter_(
                1, y.view(-1, 1), self.m, reduce='multiply',
            )
            m_theta_offset = m_theta - m_theta_ori

        confid = -0.63662 * (m_theta_ori + m_theta_offset ) + 1.
        logits = self.s * (confid)
        loss = F.cross_entropy(logits, y)

        return loss
