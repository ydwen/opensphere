import torch

from torch import distributed as dist
from torch.nn.utils import clip_grad_norm_

from utils import get_rank

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

class OpenSphere():
    def __init__(self, backbone, head,
                 optimizer=None, scheduler=None,
                 max_grad_norm=1e5):
        self.backbone = backbone
        self.head = head
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.rank = get_rank()
        self.feat_dim = head.module.feat_dim

    def set_mode(self, mode):
        if mode == 'train':
            self.backbone.train()
            self.head.train()
        elif mode == 'eval':
            self.backbone.eval()
            self.head.eval()
        else:
            raise ValueError(f'unknown mode {mode}')

    def update_param(self):
        b_params = list(self.backbone.parameters())
        h_params = list(self.head.parameters())
        grad_norm = clip_grad_norm_(
            b_params + h_params, self.max_grad_norm, norm_type=2)
        self.optimizer.step()
        self.scheduler.step()

        return grad_norm

    def fit(self, data, labels):
        # clean grad
        self.optimizer.zero_grad()
        # forward
        feats = self.backbone(data)
        loss = self.head(feats, labels)
        # backward
        loss.backward()
        # update
        grad_norm = self.update_param()

        with torch.no_grad():
            mags = torch.norm(feats, 2, 1)
        msg = {
            'Loss': loss.item(),
            'grad_norm': grad_norm,
            'Mag_mean': mags.mean().item(),
            'Mag_std': mags.std().item(),
        }

        return msg

    @torch.no_grad()
    def extract_feature(self, data, flip=True):
        feats = self.backbone(data)
        if flip:
            data = torch.flip(data, [3])
            feats = 0.5 * feats + 0.5 * self.backbone(data)
        return feats
    
    @torch.no_grad()
    def get_feature_dataset(self, dataloader):
        # create a placeholder `dataset_feats`,
        # compute _feats in different GPUs and collect 
        dataset = dataloader.dataset
        dataset_feats = torch.zeros(len(dataset), self.feat_dim)
        dataset_feats = dataset_feats.type(torch.float32).to(self.rank)
        for idx, (data, indices) in enumerate(dataloader):
            data = data.to(self.rank)
            indices = indices.tolist()
            feats = self.extract_feature(data, flip=True)
            dataset_feats[indices, :] = feats
            # print(f'feature extraction: [{dataset.name:^10s}],',
            #       f'batch: {idx+1:5d}/{len(dataloader):<5d}', end='\r')
        dist.all_reduce(dataset_feats, op=dist.ReduceOp.SUM)

        return dataset_feats
    
    @torch.no_grad()
    def scoring(self, feats0, feats1, n2m):
        return self.head.module.scoring(feats0, feats1, n2m)

    def save_checkpoint(self, path):
        checkpoint = {
            'backbone': self.backbone.state_dict(),
            'head': self.head.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        checkpoint = torch.load(path, map_location=map_location)
        self.backbone.load_state_dict(checkpoint['backbone'])
        self.head.load_state_dict(checkpoint['head'])
