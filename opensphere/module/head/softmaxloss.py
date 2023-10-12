import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxLoss(nn.Module):
    def __init__(self, feat_dim, num_class):
        super(SoftmaxLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class

        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

    def forward(self, x, y):
        logits = x.mm(self.w)
        loss = F.cross_entropy(logits, y)
        
        return loss
