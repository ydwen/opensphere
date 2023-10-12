import torch
import torch.nn.functional as F

from copy import deepcopy
from torch import distributed as dist
from torch.nn.utils import clip_grad_norm_

from .model import Model
from .base_trainer import BaseTrainer
from .utils.dist_helper import gather_concat

class ProxyFreeTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        # settings
        self.num_bank = self.config['trainer']['num_bank']
        self.encoder_momentum = self.config['trainer']['encoder_momentum']
        self.shuffle_across_gpus = self.config['trainer']['shuffle_across_gpus']
        # bank
        self.ema_backbone = deepcopy(self.model.backbone)
        self.ptr = 0
        self.feat_bank = 0.1 * torch.randn(
            (self.num_bank, self.model.embed_dim),
            dtype=torch.float32, device=self.rank,
        )
        self.label_bank = torch.randint(
            0, int(1e8), (self.num_bank, ),
            dtype=torch.long, device=self.rank,
        )

    def state_dict(self):
        return {
            **super().state_dict(),
            'ema_backbone': self.ema_backbone.state_dict(),
            'bank_ptr': self.ptr,
            'feat_bank': self.feat_bank,
            'label_bank': self.label_bank,
        }

    def load_state_dict(self, state_dict):
        # loading for base_trainer
        super().load_state_dict(state_dict)
        # bank
        self.ema_backbone.load_state_dict(state_dict['ema_backbone'])
        self.ptr = state_dict['bank_ptr']
        self.feat_bank = state_dict['feat_bank']
        self.label_bank = state_dict['label_bank']

    def shuffle_minibatch(self, data, labels):
        # shuffle across GPUs
        data_gather = gather_concat(data)
        labels_gather = gather_concat(labels)
        idx_shuffle = torch.randperm(data_gather.shape[0], device=self.rank)
        dist.broadcast(idx_shuffle, src=0)
        idx_this = idx_shuffle.view(self.world_size, -1)[self.rank]

        return data_gather[idx_this], labels_gather[idx_this]

    @torch.no_grad()
    def update_ema_backbone(self):
        # update ema backbone
        params = self.model.backbone.parameters()
        ema_params = self.ema_backbone.parameters()
        for param, ema_param in zip(params, ema_params):
            ema_param.data = ema_param.data * self.encoder_momentum \
                            + param.data * (1. - self.encoder_momentum)

    @torch.no_grad()
    def update_bank(self, data, labels):
        ''' This is adapted from MoCo repository
            (https://github.com/facebookresearch/moco)
        '''
        # get ema features
        ema_feats = self.ema_backbone(data)

        # gather keys before updating queue
        all_feats = gather_concat(ema_feats)
        all_labels = gather_concat(labels)

        batch_size = all_feats.shape[0]
        assert self.num_bank % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.feat_bank[self.ptr:self.ptr + batch_size] = all_feats
        self.label_bank[self.ptr:self.ptr + batch_size] = all_labels
        self.ptr = (self.ptr + batch_size) % self.num_bank  # move pointer

    def train_step(self):
        # load data
        data, labels = next(self.train_loader)
        data = data.view(-1, *data.shape[2:]).to(self.rank)
        labels = labels.view(-1).to(self.rank)
        # shuffle bn
        if self.shuffle_across_gpus:
            data, labels = self.shuffle_minibatch(data, labels)

        # train mode
        self.model.train_mode()
        self.ema_backbone.train()

        # update bank
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp):
            self.update_ema_backbone()
            self.update_bank(data, labels)

        self.optimizer.zero_grad()
        # forward
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp):
            feats = self.model.backbone(data)
            loss = self.model.head(feats, labels, self.feat_bank, self.label_bank)

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
            'bias': self.model.head.module.bias.data.item(),
        }
        self.train_logger.add(screen_msgs, step=self.step)
