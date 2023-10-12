import numpy as np
import os.path as osp

from torch.utils.data import Dataset
from .utils import image_pipeline
from .augmenter import Augmenter

class GroupClassDataset(Dataset):
    def __init__(
            self, data_dir, lst_path,
            sample_per_subj=4,
            unbalance_factor=1.0,
            test_mode=False,
            augment_params={
                'horizontal_flip': False, 'lowres_prob': 0.,
                'crop_prob': 0., 'photo_prob': 0.,
            },
            name='barfoo',
        ):
        super().__init__()
        self.data_dir = data_dir
        self.lst_path = lst_path
        self.sample_per_subj = sample_per_subj
        self.unbalance_factor = unbalance_factor
        self.augmenter = Augmenter(**augment_params)
        self.test_mode = test_mode
        self.name = name
        self.init_dataset()

    def init_dataset(self):
        # parse
        with open(self.lst_path, 'r') as f:
            lines = f.readlines()

        self.subj2paths = {}
        for line in lines:
            path = line.rstrip()
            subj = osp.dirname(path)
            if subj not in self.subj2paths:
                self.subj2paths[subj] = []
            self.subj2paths[subj].append(path)

        # remove subjects with less than (2 * sample_per_subj)
        min_samples = 2 * self.sample_per_subj
        self.subj2paths = {
            k: v for k, v in self.subj2paths.items() if len(v) >= min_samples
        }

        self.sorted_subjs = sorted(list(self.subj2paths.keys()))
        self.subj2label = {
            subj: idx for idx, subj in enumerate(self.sorted_subjs)
        }

        # compute sampling weights
        self.weights = [len(self.subj2paths[subj]) for subj in self.sorted_subjs]
        self.weights = [w**self.unbalance_factor for w in self.weights]

        if len(self.subj2paths) == 0:
            raise RuntimeError('Found 0 files in {}'.format(self.lst_path))

    def __len__(self):
        return len(self.sorted_subjs)

    def __getitem__(self, idx):
        # load image and pre-process (pipeline)
        subj = self.sorted_subjs[idx]
        paths = np.random.choice(
            self.subj2paths[subj], self.sample_per_subj, replace=False,
        )

        images = []
        labels = []
        for path in paths.tolist():
            item = {
                'path': osp.join(self.data_dir, path),
                'augmenter': self.augmenter,
            }
            images.append(image_pipeline(item, self.test_mode))

            subj = osp.dirname(path)
            labels.append(self.subj2label[subj])

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)
