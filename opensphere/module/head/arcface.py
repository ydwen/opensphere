import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_wrapping, get_fusing, get_scoring

class ArcFace(nn.Module):
    """ reference: <Additive Angular Margin Loss for Deep Face Recognition>
    """
    def __init__(self, feat_dim, subj_num, s=64., m=0.5):
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
            theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
            theta_m.scatter_(1, y.view(-1, 1), self.m, reduce='add')
            theta_m.clamp_(1e-5, 3.14159)
            d_theta = torch.cos(theta_m) - cos_theta

        logits = self.s * (cos_theta + d_theta)
        loss = F.cross_entropy(logits, y)

        return loss
