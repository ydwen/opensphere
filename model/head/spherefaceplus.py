import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class SphereFacePlus(nn.Module):
    """ reference: <Learning towards Minimum Hyperspherical Energy>"
    """
    def __init__(self, feat_dim, num_class, s=30., m=1.5, lambda_MHE=1.):
        super(SphereFacePlus, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.s = s
        self.m = m
        self.lambda_MHE = lambda_MHE
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

        # mini-batch MHE loss for classifiers
        sel_w = self.w[:,torch.unique(y)]
        gram_mat = torch.acos(torch.matmul(torch.transpose(sel_w, 0, 1), sel_w).clamp(-1.+1e-5, 1.-1e-5))
        shape_gram = gram_mat.size()
        MHE_loss = torch.sum(torch.triu(torch.pow(gram_mat, -2), diagonal=1))
        MHE_loss = MHE_loss / (shape_gram[0] * (shape_gram[0] - 1) * 0.5)

        loss = F.cross_entropy(logits, y) + self.lambda_MHE * MHE_loss

        return loss
