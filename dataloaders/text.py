import os.path as osp
import json
import h5py
import numpy as np
import torch
from functools import partial

from dataloaders.base import BaseDataset, DataEntry


def hdf5_to_dict(group):
    def dataset_to_dict(dataset):
        data = dataset[()]
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return data

    result = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            result[k] = dataset_to_dict(v)
        elif isinstance(v, h5py.Group):
            result[k] = hdf5_to_dict(v)
    for k, v in group.attrs.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        result[k] = v
    return result


class TextEntry(DataEntry):

    def __init__(self, index, text, label: 'int | None' = None):
        super().__init__(index, label)
        self.text = text


class TextDataset(BaseDataset):

    SUPPORTED_TYPES = [
        'json',
        'draper',
    ]

    def __init__(self, file_path, file_type=None, validate_split=1.,
                 code_field='code', label_field='label', is_train=False):
        file_path = osp.realpath(file_path)
        if file_type is None:
            file_type = file_path.split("/")[-1].split(".")[1]
        if file_type not in self.SUPPORTED_TYPES:
            raise TypeError(f"`{file_type}` is an unsupported file type for text dataset.")

        self.load = partial(getattr(self, 'load_' + file_type),
                            code_field=code_field, label_field=label_field)
        super().__init__(file_path, is_train, validate_split)

    def __getitem__(self, idx):
        return self.data[idx].text, torch.tensor(self.data[idx].label)

    @staticmethod
    def load_json(file_path, code_field='code', label_field='label'):
        # Read
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        # Preprocess
        data = []
        for i, e in enumerate(raw_data):
            # code = ' '.join(e[code_tag].split())
            code = e[code_field]
            label = e[label_field]
            entry = TextEntry(i, code, label)
            data.append(entry)
        return data

    @staticmethod
    def load_draper(file_path, code_field='functionSource'):
        # Read
        with h5py.File(file_path, 'r') as f:
            raw_data = hdf5_to_dict(f)
        # Preprocess
        ...
