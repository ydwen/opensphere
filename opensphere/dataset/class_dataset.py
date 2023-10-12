import os.path as osp

from .utils import image_pipeline
from torch.utils.data import Dataset
from .augmenter import Augmenter

class ClassDataset(Dataset):
    def __init__(
            self, data_dir, lst_path,
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
        self.test_mode = test_mode
        self.augmenter = Augmenter(**augment_params)
        self.name = name
        self.init_dataset()

    def init_dataset(self):
        """Get data from a provided annotation file.
        """
        with open(self.lst_path, 'r') as f:
            lines = f.readlines()

        self.paths = []
        subjects = set()
        for line in lines:
            path = line.rstrip()
            name = osp.dirname(path)
            self.paths.append(path)
            subjects.add(name)
        self.sorted_subjs = sorted(list(subjects))
        self.name2label = {name: idx for idx, name in enumerate(self.sorted_subjs)}

        if len(self.paths) == 0:
            raise RuntimeError('Found 0 files in {}'.format(self.lst_path))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # load image and pre-process (pipeline)
        path = self.paths[idx]
        item = {
            'path': osp.join(self.data_dir, path),
            'augmenter': self.augmenter,
        }
        image = image_pipeline(item, self.test_mode)

        name = osp.dirname(path)
        label = self.name2label[name]

        return image, label
