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

    def __init__(self, file_path, is_train=False, validate_split=1.,
                 file_type=None, code_tag='func', label_tag='target'):
        file_path = osp.realpath(file_path)
        if file_type is None:
            file_type = file_path.split("/")[-1].split(".")[1]
        self.load = partial(getattr(self, 'load_' + file_type),
                            code_tag=code_tag, label_tag=label_tag)
        super().__init__(file_path, is_train, validate_split)

    def __getitem__(self, idx):
        return self.data[idx].text, torch.tensor(self.data[idx].label)

    @staticmethod
    def load_json(file_path, code_tag='code', label_tag='label'):
        # Read
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        # Preprocess
        data = []
        for i, e in enumerate(raw_data):
            # code = ' '.join(e[code_tag].split())
            code = e[code_tag]
            label = e[label_tag]
            entry = TextEntry(i, code, label)
            data.append(entry)
        return data

    @staticmethod
    def load_hdf5(file_path, code_tag='code', label_tag='label'):
        # Read
        with h5py.File(file_path, 'r') as f:
            raw_data = hdf5_to_dict(f)
        # Preprocess
        data = []
        codes, labels = raw_data[code_tag], labels[label_tag]
        for i, (code, label) in enumerate(zip(codes, labels)):
            entry = TextEntry(i, code, label)
            data.append(entry)
