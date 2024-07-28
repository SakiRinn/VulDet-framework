import csv
from dataclasses import dataclass, field
from typing import Any
import os.path as osp
import json
from functools import partial
import warnings

import torch

from .base import BaseDataset


@dataclass(frozen=True)
class TextEntry:
    idx: int
    text: str
    label: int
    vul_type: 'str | None' = None
    info: 'dict[str, Any]' = field(default_factory={})


class TextDataset(BaseDataset):

    SUPPORTED_FORMATS = [
        'json',
        'csv'
    ]

    def __init__(self, file_path, data_format: 'str | None' = None,
                 code_field='code', label_field='label', type_field=None):
        file_path = osp.realpath(file_path)
        if data_format is None:
            data_format = file_path.split("/")[-1].split(".")[1]
        data_format = data_format.strip()
        if data_format not in self.SUPPORTED_FORMATS:
            raise TypeError(f"`{data_format}` is an unsupported format for text dataset.")

        self.code_field = code_field
        self.label_field = label_field
        self.type_field = type_field
        super().__init__(file_path)

    def __getitem__(self, idx):
        if idx != self.data[idx].idx:
            warnings.warn('The dataset index is inconsistent with the entry index, '
                          'the order of data may be disrupted!')
        return self.data[idx].text, torch.tensor(self.data[idx].label)

    def load(self, file_path):
        load_func = getattr(self, 'from_' + self.data_format)
        return load_func(self, file_path)

    def from_json(self, file_path):
        # Read
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        # Process
        data = []
        for i, e in enumerate(raw_data):
            code = e.pop(self.code_field)
            label = int(e.pop(self.label_field))
            vul_type = e.pop(self.type_field) if self.type_field is not None else None
            entry = TextEntry(i, code, label, vul_type, e)
            data.append(entry)
        return data

    def from_csv(self, file_path, delimiter=','):
        # Read
        with open(file_path, 'r') as f:
            csv_reader = csv.DictReader(f, delimiter=delimiter)
            raw_data = [row for row in csv_reader]
        # Process
        data = []
        for i, e in enumerate(raw_data):
            code = e.pop(self.code_field)
            label = int(e.pop(self.label_field))
            vul_type = e.pop(self.type_field) if self.type_field is not None else None
            entry = TextEntry(i, code, label, vul_type, e)
            data.append(entry)
        return data
