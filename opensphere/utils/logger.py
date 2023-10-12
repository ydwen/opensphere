import os
import sys
import math
import torch
import logging

from sklearn import metrics
from scipy.interpolate import interp1d

class Meter():
    '''
    steps = [step0, step1, ...]
    records = [record0, record1, ...]
    '''
    def __init__(self, fmt='7.3f'):
        self.fmt = fmt
        self.clean()

    def clean(self):
        self.headers = []
        self.records = []
        self.steps = []

    def add(self, record, step):
        # check headers
        if len(self.headers) == 0:
            self.headers = list(record.keys())
        assert set(record.keys()) == set(self.headers)
        # add record
        self.records.append(record)
        self.steps.append(step)

    @property
    def avg_record(self):
        avg_record = {}
        for h in self.headers:
            vals = [record[h] for record in self.records]
            avg_record[h] = sum(vals) / len(vals)
        return avg_record

    def apply_format(self, record, has_header=False):
        if has_header:
            return [f'{h}: {record[h]:{self.fmt}}' for h in self.headers]
        else:
            return [f'{record[h]:{self.fmt}}' for h in self.headers]

    def summary(self):
        assert len(self.records) == len(self.steps)
        headers = ['Step'] + self.headers
        content = []
        for record, step in zip(self.records, self.steps):
            content.append([step] + self.apply_format(record))
        content.append(['avg'] + self.apply_format(self.avg_record))
        return content, headers

    def get_message(self):
        assert len(self.records) == len(self.steps)
        message = [f'Step: {self.steps[0]:6d}~{self.steps[-1]:<6d}']
        message.extend(self.apply_format(self.avg_record, has_header=True))
        return ', '.join(message)


class PythonLogger():
    def __init__(self, name, path, fmt, info_intvl=1):
        self.logger = self.get_logger(name, path)
        self.meter = Meter(fmt)
        self.info_intvl = info_intvl

    def get_logger(self, name, path):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # set format
        msg_fmt = '%(asctime)s, %(message)s'
        time_fmt = '%Y%m%d_%H%M%S'
        formatter = logging.Formatter(msg_fmt, time_fmt)

        # define file and steam handler and set formatter
        if os.path.exists(path):
            file_handler = logging.FileHandler(path, 'a')
        else:
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

    def add(self, record, step):
        self.meter.add(record, step)
        # construct message to show on screen every `self.info_intvl` iters
        if len(self.meter.steps) >= self.info_intvl:
            msg = self.meter.get_message()
            self.logger.info(msg)
            self.meter.clean()
