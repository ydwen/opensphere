import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_wrapping, get_fusing, get_scoring

class CosFace(nn.Module):
    """reference1: <CosFace: Large Margin Cosine Loss for Deep Face Recognition>
       reference2: <Additive Margin Softmax for Face Verification>
    """
    def __init__(self, feat_dim, subj_num, s=64., m=0.35):
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
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=-1)

        x = self.f_wrapping(x)
        x = self.f_fusing(x[:, None, :])
        cos_theta = self.f_scoring(x, self.w, all_pairs=True)
        with torch.no_grad():
            d_theta = torch.zeros_like(cos_theta)
            d_theta.scatter_(1, y.view(-1, 1), -self.m, reduce='add')

        logits = self.s * (cos_theta + d_theta)
        loss = F.cross_entropy(logits, y)

        return loss
