import torch

from .model import Model
from torch.nn.utils import clip_grad_norm_
from .base_trainer import BaseTrainer

class ProxyBasedTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def init_model(self):
        # build model
        proxy_cfg = {
            'feat_dim': self.config['model']['backbone']['embed_dim'],
            'subj_num': len(self.train_loader._dataloader.dataset.sorted_subjs),
        }
        self.config['model']['head'].update(proxy_cfg)
        super().init_model()

    def train_step(self):
        # load data
        data, labels = next(self.train_loader)
        data, labels = data.to(self.rank), labels.to(self.rank)

        self.model.train_mode()

        self.optimizer.zero_grad()
        # forward
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp):
            feats = self.model.backbone(data)
            loss = self.model.head(feats, labels)

        # backward
        self.scaler.scale(loss).backward()
        # gradient clip
        self.scaler.unscale_(self.optimizer)
        b_params = list(self.model.backbone.parameters())
        h_params = list(self.model.head.parameters())
        grad_norm = clip_grad_norm_(b_params + h_params, self.max_grad_norm, norm_type=2)
        # update
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        if self.rank != 0:
            return

        # log
        with torch.no_grad():
            wrapped_feats = self.model.head.module.f_wrapping(feats)
            mags = torch.norm(wrapped_feats, dim=-1)
            mag_std, mag_mean = torch.std_mean(mags)
        screen_msgs = {
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'mag_mean': mag_mean.item(),
            'mag_std': mag_std.item(),
        }
        self.train_logger.add(screen_msgs, step=self.step)
