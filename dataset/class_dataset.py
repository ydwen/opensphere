import random
import os.path as osp

from .utils import image_pipeline
from torch.utils.data import Dataset


class ClassDataset(Dataset):
    def __init__(self, name, data_dir, ann_path,
            test_mode=False, noise_ratio=None, seed=None):
        super().__init__()

        self.name = name
        self.data_dir = data_dir
        self.ann_path = ann_path
        self.test_mode = test_mode
        self.noise_ratio = noise_ratio
        self.seed = seed

        self.get_data()
        self.get_label()

    def get_data(self):
        """Get data from a provided annotation file.
        """
        with open(self.ann_path, 'r') as f:
            lines = f.readlines()

        self.data_items = []
        for line in lines:
            path, name= line.rstrip().split()
            item = {'path': path, 'name': name}
            self.data_items.append(item)

        if len(self.data_items) == 0:
            raise (RuntimeError('Found 0 files.'))

    def corrupt_label(self):
        random.seed(self.seed)
        labels = list({item['label'] for item in self.label_items})
        for item in self.label_items:
            if random.random() > self.noise_ratio:
                continue
            item['label'] = random.choice(labels)

    def get_label(self):
        """ convert name to label,
            and optionally permutate some labels
        """
        names = {item['name'] for item in self.data_items}
        names = sorted(list(names))
        self.classes = names
        name2label = {name: idx for idx, name in enumerate(names)}

        self.label_items = []
        for item in self.data_items:
            label = name2label[item['name']]
            self.label_items.append({'label': label})

        if self.noise_ratio:
            self.corrupt_label()

    def prepare(self, idx):
        # load image and pre-process (pipeline)
        path = self.data_items[idx]['path']
        item = {'path': osp.join(self.data_dir, path)}
        image = image_pipeline(item, self.test_mode)
        label = self.label_items[idx]['label']

        return image, label

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        return self.prepare(idx)
