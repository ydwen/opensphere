import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class SphereFace2M(nn.Module):
    """ reference: <SphereFace2: Binary Classification is All You Need
                    for Deep Face Recognition>
    """
    def __init__(self, feat_dim, num_class,
            alpha=0.7, r=40., m=0.4, t=3., lw=50.):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class

        # alpha is the lambda in paper Eqn. 5
        self.alpha = alpha
        self.r = r
        self.m = m
        self.t = t
        self.lw = lw

        # init weights
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

        # init bias
        z = alpha / ((1. - alpha) * (num_class - 1.))
        theta_y = min(math.pi, m * math.pi/2.)
        ay = r * (2. * ((math.cos(theta_y) + 1.) / 2.)**t - 1.)
        ai = r * (2. * 0.5**t - 1.)
        tmp = (1. - z)**2 + 4. * z * math.exp(ay - ai)
        b = math.log(2. * z) - ai \
            - math.log(1. - z +  math.sqrt(tmp))
        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.b, b)

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        #delta theta with margin
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, y.view(-1, 1), 1.)
        with torch.no_grad():
            eps = 1e-8
            m_theta = torch.acos(cos_theta.clamp(-1.+eps, 1.-eps))
            m_theta.scatter_(1, y.view(-1, 1), self.m, reduce='multiply')
            g_cos_theta = torch.cos(m_theta.clamp_(eps, 3.14))
            g_cos_theta = 2. * ((g_cos_theta + 1.) / 2.).pow(self.t) - 1.
            d_theta = g_cos_theta - cos_theta
        
        logits = self.r * (cos_theta + d_theta) + self.b
        weight = self.alpha * one_hot + (1. - self.alpha) * (1. - one_hot)
        weight = self.lw * self.num_class / self.r * weight
        loss = F.binary_cross_entropy_with_logits(
                logits, one_hot, weight=weight)

        return loss

    def scoring(self, x0, x1, n2m=False):
        x0 = F.normalize(x0, dim=1)
        x1 = F.normalize(x1, dim=1)
        if n2m:
            return x0.mm(x1.t())
        return torch.sum(x0 * x1, dim=1)
