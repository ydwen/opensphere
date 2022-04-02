import torch
import torch.nn as nn
import torch.nn.functional as F


class CocoLoss(nn.Module):
    """reference1: <Learning Deep Features via
                    Congenerous Cosine Loss for Person Recognition>
       reference2: <NormFace: L2 Hypersphere Embedding for Face Verification>
    """
    def __init__(self, feat_dim, num_class, s=30.):
        super(CocoLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.s = s
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(x, dim=1).mm(self.w)
        logits = self.s * cos_theta

        loss = F.cross_entropy(logits, y)

        return loss
