import sys
import logging
import collections

from copy import deepcopy
from torch import distributed as dist


def is_dist() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def merge(dict1, dict2):
    ''' Return a new dictionary by merging
        two dictionaries recursively.
    '''
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if isinstance(value, collections.abc.Mapping):
            result[key] = merge(result.get(key, {}), value)
        else:
            result[key] = deepcopy(dict2[key])
    return result

def fill_config(config):
    base_cfg = config.pop('base', {})
    for sub, sub_cfg in config.items():
        if isinstance(sub_cfg, dict):
            config[sub] = merge(base_cfg, sub_cfg)
        elif isinstance(sub_cfg, list):
            config[sub] = [merge(base_cfg, c) for c in sub_cfg]
    return config


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


class Meter():
    def __init__(self, fmt='7.3f'):
        self.fmt = fmt
        self.clean()

    def get_avg_record(self):
        h = 'Iter'
        v = '{:5d}~{:<5d}'.format(self.history[0][h], self.history[-1][h])
        avg_record = {h: f'{v}'}
        for h in self.headers:
            if h == 'Iter':
                continue
            vals = [record[h] for record in self.history]
            avg_record[h] = sum(vals) / len(vals)

        return avg_record
    
    def summary(self):
        avg_record = self.get_avg_record()
        return self.history + [avg_record]

    def record2message(self, record):
        h = 'Iter'
        v = record[h]
        if isinstance(v, int):
            message = [f'{h}: {v:5d}']
        elif isinstance(v, str):
            message = [f'{h}: {v}']
        else:
            raise ValueError(f'Unkown type of {v}: {type(v)}')

        for h, v in record.items():
            if h == 'Iter':
                continue
            message.append(f'{h}: {v:{self.fmt}}')

        return ', '.join(message)

    def get_avg_message(self):
        avg_record = self.get_avg_record()
        return self.record2message(avg_record)

    def clean(self):
        self.history = []
        self.headers = []

    def add(self, record):
        # check headers
        if len(self.headers) == 0:
            self.headers = list(record.keys())
        assert set(record.keys()) == set(self.headers)
        # add record
        self.history.append(record)


class MeterLogger():
    def __init__(self, name, path, fmt, info_intvl=1):
        self.logger = self.get_logger(name, path)
        self.meter = Meter(fmt)
        self.info_intvl = info_intvl

    def get_logger(self, name, path):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # set format
        msg_fmt = '[%(levelname)s] %(asctime)s, %(message)s'
        time_fmt = '%Y%m%d_%H%M%S'
        formatter = logging.Formatter(msg_fmt, time_fmt)

        # define file (DEBUG) and steam (INFO) handler and set formatter
        file_handler = logging.FileHandler(path, 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

        # to avoid duplicated logging info
        if len(logger.root.handlers) == 0:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            stream_handler.setLevel(logging.WARNING)
            logger.root.addHandler(stream_handler)

        return logger

    def update(self, record):
        self.meter.add(record)
        # construct message to show on screen every `self.info_intvl` iters
        if len(self.meter.history) >= self.info_intvl:
            msg = self.meter.get_avg_message()
            self.logger.info(msg)
            self.meter.clean()
