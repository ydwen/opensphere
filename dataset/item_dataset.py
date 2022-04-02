import os.path as osp

from .utils import image_pipeline
from torch.utils.data import Dataset


class ItemDataset(Dataset):
    """ An example of creating a dataset from a given data_items.
    """
    def __init__(self, name, data_items, test_mode=True):
        super().__init__()

        self.name = name
        self.data_items = data_items
        self.test_mode = test_mode

    def prepare(self, idx):
        # load image and pre-process (pipeline)
        image = image_pipeline(self.data_items[idx], self.test_mode)

        return image, idx

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        return self.prepare(idx)
