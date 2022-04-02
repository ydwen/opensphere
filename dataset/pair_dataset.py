import torch
import os.path as osp

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from .utils import image_pipeline, get_metrics


class PairDataset(Dataset):
    def __init__(self, name, data_dir, ann_path, metrics,
            test_mode=True):
        super().__init__()

        self.name = name
        self.data_dir = data_dir
        self.ann_path = ann_path
        self.metrics = metrics
        self.test_mode = test_mode

        self.get_data()
        self.get_label()

    def get_data(self):
        """Get data from an annotation file.
        """
        with open(self.ann_path, 'r') as f:
            lines = f.readlines()

        paths = set()
        for line in lines:
            _, path1, path2 = line.rstrip().split(' ')
            paths.add(path1)
            paths.add(path2)
        paths = list(paths)
        paths.sort()
        self.data_items = [{'path': path} for path in paths]

        if len(self.data_items) == 0:
            raise (RuntimeError('Found 0 files.'))

    def get_label(self):
        """Get labels from an annoation file
        """
        with open(self.ann_path, 'r') as f:
            lines = f.readlines()

        path2index = {item['path']: idx 
                for idx, item in enumerate(self.data_items)}

        self.indices0 = []
        self.indices1 = []
        self.labels = []
        for line in lines:
            label, path0, path1 = line.rstrip().split(' ')
            self.indices0.append(path2index[path0])
            self.indices1.append(path2index[path1])
            self.labels.append(int(label))

    def prepare(self, idx):
        # load image and pre-process (pipeline) from path
        path = self.data_items[idx]['path']
        item = {'path': osp.join(self.data_dir, path)}
        image = image_pipeline(item, self.test_mode)

        return image, idx

    def evaluate(self, feats, 
            FPRs=['1e-4', '5e-4', '1e-3', '5e-3', '5e-2']):
        # pair-wise scores
        feats = F.normalize(feats, dim=1)
        feats0 = feats[self.indices0, :]
        feats1 = feats[self.indices1, :]
        scores = torch.sum(feats0 * feats1, dim=1).tolist()

        return get_metrics(self.labels, scores, FPRs)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        return self.prepare(idx)
