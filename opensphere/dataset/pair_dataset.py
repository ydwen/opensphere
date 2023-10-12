import os.path as osp
import torch
from torch.utils.data import Dataset
from .utils import image_pipeline, get_results

class PairDataset(Dataset):
    def __init__(self, name, data_dir, ann_path,
                 metrics=['ACC'], test_mode=True):
        super().__init__()

        self.name = name
        self.data_dir = data_dir
        self.ann_path = ann_path
        self.metrics = metrics
        self.test_mode = test_mode

        self.init_dataset()

    def init_dataset(self):
        """ Get data from an annotation file.
        """
        with open(self.ann_path, 'r') as f:
            lines = f.readlines()

        paths = set()
        for line in lines:
            _, path0, path1 = line.rstrip().split(' ')
            paths.add(path0)
            paths.add(path1)
        self.paths = list(paths)
        self.paths.sort()

        if len(self.paths) == 0:
            raise (RuntimeError('Found 0 files.'))

        path2index = {path: idx for idx, path in enumerate(self.paths)}
        self.indices0 = []
        self.indices1 = []
        self.labels = []
        for line in lines:
            label, path0, path1 = line.rstrip().split(' ')
            self.indices0.append(path2index[path0])
            self.indices1.append(path2index[path1])
            self.labels.append(int(label))

    def fuse_features(self, feats, f_wrapping, f_fusing):
        # fuse features
        fused_feats = []
        for idx in range(feats.size(0)):
            wrapped_feats = f_wrapping(feats[idx:idx+1])
            wrapped_feats = wrapped_feats.view(-1, wrapped_feats.size(-1))
            fuse_feat = f_fusing(wrapped_feats)
            fused_feats.append(fuse_feat)
        fused_feats = torch.stack(fused_feats, dim=0)

        return fused_feats

    def compute_scores(self, fused_feats, f_scoring):
        # 1:1 scoring
        # split `indices` into minibatches
        num_minibatch = 100000
        scores = []
        for t in range(0, len(self.indices0), num_minibatch):
            _indices0 = self.indices0[t:t+num_minibatch]
            _indices1 = self.indices1[t:t+num_minibatch]
            _scores = f_scoring(
                fused_feats[_indices0], fused_feats[_indices1], all_pairs=False,
            )
            scores.extend(_scores.tolist())

        return scores

    def evaluate(self, feats, f_wrapping, f_fusing, f_scoring):
        fused_feats = self.fuse_features(feats, f_wrapping, f_fusing)
        scores = self.compute_scores(fused_feats, f_scoring)

        # get results and miscs
        FPRs = [m.split('=')[-1] for m in self.metrics if 'FPR' in m]
        results = get_results(scores, self.labels, FPRs)
        results = [(m, results[m]) for m in self.metrics]

        return results, scores, self.labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # load image and pre-process (pipeline) from path
        path = self.paths[idx]
        item = {'path': osp.join(self.data_dir, path)}
        image = image_pipeline(item, self.test_mode)

        return image, idx
